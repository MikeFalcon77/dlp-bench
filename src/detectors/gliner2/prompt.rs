//! GLiNER2 prompt construction.
//!
//! The encoder input is *not* a plain tokenization of the text. It is a schema
//! prompt followed by the text, with no `[CLS]`/`[SEP]`:
//!
//! ```text
//! (  [P]  entities  (  [E] label0  [E] label1 ... )  )  [SEP_TEXT]  w0 w1 ...
//! ```
//!
//! Each *word* in that sequence is tokenized independently with
//! `add_special_tokens = false` and the sub-token id lists are concatenated.
//! Two asymmetries from the reference that are easy to get backwards:
//!
//! * schema label strings are **not** lowercased;
//! * text words **are** lowercased.
//!
//! `schema_special_indices` is `[position of [P]] + [position of each [E]]`,
//! hence its length is `num_labels + 1` — the graph reads the `[P]` embedding
//! for the count head and the `[E]` embeddings as label representations.
//!
//! # Why the prefix is precomputed
//!
//! Every token position in the schema section depends only on preceding schema
//! tokens, so the prefix ids and `schema_special_indices` are text-independent
//! and are built once at load. The graph consumes raw ids plus index tensors
//! and never re-derives the schema, so this is exact, not an approximation.
//!
//! It is also a smaller win than it looks, and the benchmark should say so:
//! the prefix is ~4 tokens per label plus overhead, and all 12 encoder layers
//! reprocess it on **every** call because the encoder is bidirectional. For a
//! short prompt the label schema, not the document, dominates the FLOPs. That
//! is why schema size is a first-class perf dimension.

use std::collections::HashMap;

use tokenizers::Tokenizer;

use super::words::Word;
use crate::error::DetectorError;

/// Special token ids, verified against the shipped `tokenizer.json`.
pub mod special {
    pub const SEP_STRUCT: &str = "[SEP_STRUCT]";
    pub const SEP_TEXT: &str = "[SEP_TEXT]";
    pub const P: &str = "[P]";
    pub const E: &str = "[E]";

    /// Ids asserted at load; a tokenizer that disagrees is the wrong tokenizer.
    pub const SEP_STRUCT_ID: u32 = 250102;
    pub const SEP_TEXT_ID: u32 = 250103;
    pub const P_ID: u32 = 250104;
    pub const E_ID: u32 = 250106;
}

/// The precomputed, text-independent part of the encoder input.
#[derive(Debug, Clone)]
pub struct PromptPrefix {
    /// Token ids for `( [P] entities ( [E] L0 ... ) ) [SEP_TEXT]`.
    pub ids: Vec<i64>,
    /// `[pos([P])] + [pos([E]_i) for each label]`; length `num_labels + 1`.
    pub schema_special_indices: Vec<i64>,
    pub num_labels: usize,
}

impl PromptPrefix {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Caches sub-token ids per lowercased word.
///
/// The reference uses an LRU of 50k; correctness does not depend on the
/// eviction policy because `encode` is pure, so a plain map with a size cap is
/// equivalent and simpler.
#[derive(Debug, Default)]
pub struct WordIdCache {
    map: HashMap<Box<str>, Vec<i64>>,
    cap: usize,
}

impl WordIdCache {
    pub fn new(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            cap,
        }
    }

    fn get_or_insert(&mut self, word: &str, tok: &Tokenizer) -> Result<&[i64], DetectorError> {
        if !self.map.contains_key(word) {
            if self.map.len() >= self.cap {
                self.map.clear(); // cheap bounded reset; hit rate recovers immediately
            }
            let ids = encode_word(tok, word)?;
            self.map.insert(word.into(), ids);
        }
        Ok(&self.map[word])
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Tokenize one word to ids, with no special tokens added.
///
/// This is the exact analogue of the reference's
/// `convert_tokens_to_ids(tokenizer.tokenize(word))`: the `TemplateProcessing`
/// post-processor that would wrap the sequence in `[CLS] ... [SEP]` is skipped
/// because `add_special_tokens` is false.
pub fn encode_word(tok: &Tokenizer, word: &str) -> Result<Vec<i64>, DetectorError> {
    let enc = tok
        .encode(word, false)
        .map_err(|e| DetectorError::Tokenizer(format!("encoding {word:?}: {e}")))?;
    Ok(enc.get_ids().iter().map(|&i| i64::from(i)).collect())
}

/// Build the schema prefix for a fixed label set.
pub fn build_prefix(tok: &Tokenizer, labels: &[&str]) -> Result<PromptPrefix, DetectorError> {
    if labels.is_empty() {
        return Err(DetectorError::config(
            "GLiNER2 schema must have at least one label",
        ));
    }

    // Word sequence, mirroring `_transform_schema` + `_format_input_with_mapping`.
    // Note the single schema means the `[SEP_STRUCT]` the reference appends is
    // immediately popped, so it never appears.
    let mut ids: Vec<i64> = Vec::with_capacity(labels.len() * 4 + 8);
    let mut schema_special_indices: Vec<i64> = Vec::with_capacity(labels.len() + 1);

    let push = |ids: &mut Vec<i64>, word: &str| -> Result<(), DetectorError> {
        ids.extend(encode_word(tok, word)?);
        Ok(())
    };

    push(&mut ids, "(")?;
    // `[P]` marks the parent; its embedding feeds the count head.
    schema_special_indices.push(ids.len() as i64);
    push(&mut ids, special::P)?;
    push(&mut ids, "entities")?;
    push(&mut ids, "(")?;
    for label in labels {
        // `[E]` marks a child field; its embedding is that label's query vector.
        schema_special_indices.push(ids.len() as i64);
        push(&mut ids, special::E)?;
        // Label strings keep their case — the reference does not lowercase them.
        push(&mut ids, label)?;
    }
    push(&mut ids, ")")?;
    push(&mut ids, ")")?;
    push(&mut ids, special::SEP_TEXT)?;

    debug_assert_eq!(schema_special_indices.len(), labels.len() + 1);

    Ok(PromptPrefix {
        ids,
        schema_special_indices,
        num_labels: labels.len(),
    })
}

/// The tokenizer's `[PAD]` id.
///
/// Asserted against the shipped tokenizer at load, like the other special ids —
/// padding with the wrong token would not error, it would just quietly change
/// what the encoder sees.
pub const PAD_ID: i64 = 0;

/// Pin the encoder input to a fixed size by right-padding.
///
/// Exists so a session built with ONNX Runtime dimension overrides gets inputs
/// of exactly the size it was told to expect. Every document then costs what
/// the *bucket* costs rather than what the document costs, which is the trade
/// static shapes ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedShape {
    pub seq_len: usize,
    pub num_words: usize,
}

/// Per-document encoder input.
#[derive(Debug, Default)]
pub struct EncodedInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    /// Absolute position of each text word's first sub-token.
    pub text_word_indices: Vec<i64>,
    /// Tokens before padding. Equals `input_ids.len()` when unpadded.
    pub real_seq_len: usize,
    /// Words before padding. Equals `text_word_indices.len()` when unpadded.
    pub real_num_words: usize,
}

impl EncodedInput {
    pub fn seq_len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn num_words(&self) -> usize {
        self.text_word_indices.len()
    }
}

/// Append the text words to the precomputed prefix, optionally padding to
/// `pad`.
///
/// Buffers are reused across documents; the caller owns them.
pub fn build_input(
    prefix: &PromptPrefix,
    words: &[Word],
    tok: &Tokenizer,
    cache: &mut WordIdCache,
    pad: Option<FixedShape>,
    out: &mut EncodedInput,
) -> Result<(), DetectorError> {
    out.input_ids.clear();
    out.input_ids.extend_from_slice(&prefix.ids);
    out.text_word_indices.clear();
    out.text_word_indices.reserve(words.len());

    for w in words {
        // Recorded *before* extending, which reproduces the reference's
        // behaviour for a word that tokenizes to nothing: it inherits the next
        // word's first-subtoken position.
        out.text_word_indices.push(out.input_ids.len() as i64);
        let sub = cache.get_or_insert(&w.lower, tok)?;
        out.input_ids.extend_from_slice(sub);
    }

    if out.input_ids.is_empty() {
        return Err(DetectorError::inference("empty encoder input"));
    }

    // The reference has a latent out-of-bounds here: a *trailing* zero-subtoken
    // word records an index equal to `seq_len`, and the gather then reads past
    // the end. Clamp rather than reproduce a crash.
    let last = (out.input_ids.len() - 1) as i64;
    for v in &mut out.text_word_indices {
        if *v > last {
            *v = last;
        }
    }

    out.real_seq_len = out.input_ids.len();
    out.real_num_words = out.text_word_indices.len();

    out.attention_mask.clear();
    out.attention_mask.resize(out.real_seq_len, 1);

    if let Some(fixed) = pad {
        // Overflow is an error, never a truncation. Silently dropping the tail
        // of a document would make a DLP detector miss whatever was in it and
        // report success, which is the worst failure this code could have.
        if out.real_seq_len > fixed.seq_len {
            return Err(DetectorError::inference(format!(
                "document needs {} tokens but the session was built for {}; \
                 use a larger bucket or split the document",
                out.real_seq_len, fixed.seq_len
            )));
        }
        if out.real_num_words > fixed.num_words {
            return Err(DetectorError::inference(format!(
                "document has {} words but the session was built for {}; \
                 use a larger bucket or split the document",
                out.real_num_words, fixed.num_words
            )));
        }

        out.input_ids.resize(fixed.seq_len, PAD_ID);
        // Zero mask over the pad region is the entire correctness argument for
        // this feature; `tests/static_shapes.rs` checks that the graph honours
        // it rather than assuming a traced export kept the mask at all.
        out.attention_mask.resize(fixed.seq_len, 0);

        // Pad words point at the *last real token*, never into the pad region
        // and never into the schema prefix. Their `span_probs` rows come out as
        // duplicates of the last real word's and are discarded by `decode`,
        // which clamps `num_words` to `words.len()`.
        out.text_word_indices.resize(fixed.num_words, last);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detectors::gliner2::test_support::{TOKENIZER_PATH, load_tokenizer, model_present};
    use crate::detectors::gliner2::words;

    #[test]
    fn pad_id_is_the_tokenizers_own_pad_token() {
        // Padding with the wrong id would not error anywhere; it would just
        // feed the encoder a different token and change the outputs slightly.
        if !model_present() {
            eprintln!("skipping: run scripts/fetch-models.sh");
            return;
        }
        let tok = load_tokenizer();
        assert_eq!(
            tok.token_to_id("[PAD]").map(i64::from),
            Some(PAD_ID),
            "PAD_ID no longer matches the shipped tokenizer"
        );
    }

    #[test]
    fn padding_extends_the_input_without_disturbing_the_real_part() {
        if !model_present() {
            eprintln!("skipping: run scripts/fetch-models.sh");
            return;
        }
        let tok = load_tokenizer();
        let prefix = build_prefix(&tok, &["email", "person"]).unwrap();
        let mut cache = WordIdCache::new(64);
        let prepared = words::prepare("contact bob at bob@example.com");

        let mut plain = EncodedInput::default();
        build_input(&prefix, &prepared.words, &tok, &mut cache, None, &mut plain).unwrap();

        let fixed = FixedShape {
            seq_len: plain.seq_len() + 40,
            num_words: plain.num_words() + 9,
        };
        let mut padded = EncodedInput::default();
        build_input(
            &prefix,
            &prepared.words,
            &tok,
            &mut cache,
            Some(fixed),
            &mut padded,
        )
        .unwrap();

        assert_eq!(padded.seq_len(), fixed.seq_len);
        assert_eq!(padded.num_words(), fixed.num_words);
        assert_eq!(padded.real_seq_len, plain.seq_len());
        assert_eq!(padded.real_num_words, plain.num_words());

        let real = plain.seq_len();
        assert_eq!(
            &padded.input_ids[..real],
            &plain.input_ids[..],
            "padding must not perturb the real tokens"
        );
        assert!(
            padded.input_ids[real..].iter().all(|&v| v == PAD_ID),
            "tail must be [PAD]"
        );
        assert!(
            padded.attention_mask[..real].iter().all(|&v| v == 1)
                && padded.attention_mask[real..].iter().all(|&v| v == 0),
            "mask must be 1 over the real tokens and 0 over the pad"
        );

        // Every word index must address a real token — a pad-region index
        // would gather a [PAD] embedding, and a prefix index would gather a
        // schema token, both of which decode into plausible garbage.
        let last_real = (real - 1) as i64;
        assert!(
            padded.text_word_indices.iter().all(|&i| i <= last_real),
            "a word index points past the real tokens"
        );
        assert!(
            padded.text_word_indices[plain.num_words()..]
                .iter()
                .all(|&i| i == last_real),
            "pad words must all point at the last real token"
        );
    }

    #[test]
    fn a_document_too_large_for_its_bucket_errors_rather_than_truncating() {
        if !model_present() {
            eprintln!("skipping: run scripts/fetch-models.sh");
            return;
        }
        let tok = load_tokenizer();
        let prefix = build_prefix(&tok, &["email"]).unwrap();
        let mut cache = WordIdCache::new(64);
        let prepared = words::prepare("one two three four five six seven eight nine ten");
        let mut enc = EncodedInput::default();

        // Truncation here would silently drop the tail of a document, and a
        // DLP detector that misses what it never saw reports success.
        let err = build_input(
            &prefix,
            &prepared.words,
            &tok,
            &mut cache,
            Some(FixedShape {
                seq_len: 4,
                num_words: 64,
            }),
            &mut enc,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("tokens"), "{err}");
        assert!(err.contains("larger bucket"), "unhelpful error: {err}");

        let err = build_input(
            &prefix,
            &prepared.words,
            &tok,
            &mut cache,
            Some(FixedShape {
                seq_len: 4096,
                num_words: 2,
            }),
            &mut enc,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("words"), "{err}");
    }

    #[test]
    fn special_token_ids_match_the_shipped_tokenizer() {
        if !model_present() {
            eprintln!("skipping: run scripts/fetch-models.sh");
            return;
        }
        let tok = load_tokenizer();
        for (name, want) in [
            (special::P, special::P_ID),
            (special::E, special::E_ID),
            (special::SEP_TEXT, special::SEP_TEXT_ID),
            (special::SEP_STRUCT, special::SEP_STRUCT_ID),
        ] {
            let ids = encode_word(&tok, name).unwrap();
            assert_eq!(
                ids,
                vec![i64::from(want)],
                "{name} must encode to exactly one id ({want}), got {ids:?} from {}",
                TOKENIZER_PATH
            );
        }
    }

    #[test]
    fn prefix_layout_matches_the_reference_recipe() {
        if !model_present() {
            return;
        }
        let tok = load_tokenizer();
        let labels = ["person", "email", "phone_number"];
        let p = build_prefix(&tok, &labels).unwrap();

        assert_eq!(p.num_labels, 3);
        assert_eq!(p.schema_special_indices.len(), 4, "num_labels + 1");

        // Index 0 points at [P]; the rest point at each [E].
        assert_eq!(
            p.ids[p.schema_special_indices[0] as usize],
            i64::from(special::P_ID)
        );
        for i in 1..=3 {
            assert_eq!(
                p.ids[p.schema_special_indices[i] as usize],
                i64::from(special::E_ID),
                "index {i} must point at an [E] marker"
            );
        }
        // The prefix ends with [SEP_TEXT].
        assert_eq!(*p.ids.last().unwrap(), i64::from(special::SEP_TEXT_ID));
        // A single schema means [SEP_STRUCT] is never emitted.
        assert!(
            !p.ids.contains(&i64::from(special::SEP_STRUCT_ID)),
            "[SEP_STRUCT] must not appear for a single schema"
        );
        // Indices are strictly increasing.
        assert!(p.schema_special_indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn prefix_grows_with_the_label_count() {
        if !model_present() {
            return;
        }
        let tok = load_tokenizer();
        let small = build_prefix(&tok, &["person"]).unwrap();
        let big = build_prefix(&tok, &crate::labels::GLINER2_PII_FULL.labels()).unwrap();
        assert!(big.len() > small.len());
        // The quantity that dominates short-prompt latency; asserted so the
        // README's claim stays honest if the schema changes.
        assert!(
            big.len() > 100,
            "42-label prefix is only {} tokens; the perf story assumes ~4/label",
            big.len()
        );
    }

    #[test]
    fn empty_schema_is_rejected() {
        if !model_present() {
            return;
        }
        let tok = load_tokenizer();
        assert!(build_prefix(&tok, &[]).is_err());
    }

    #[test]
    fn text_word_indices_point_at_first_subtokens_after_the_prefix() {
        if !model_present() {
            return;
        }
        let tok = load_tokenizer();
        let prefix = build_prefix(&tok, &["person", "email"]).unwrap();
        let prepared = words::prepare("Contact John Smith at john@example.com");
        let mut cache = WordIdCache::new(1024);
        let mut enc = EncodedInput::default();
        build_input(&prefix, &prepared.words, &tok, &mut cache, None, &mut enc).unwrap();

        assert_eq!(enc.num_words(), prepared.words.len());
        assert_eq!(enc.attention_mask.len(), enc.seq_len());
        assert!(enc.attention_mask.iter().all(|&m| m == 1));

        // The prefix is an exact leading slice of every input.
        assert_eq!(&enc.input_ids[..prefix.len()], prefix.ids.as_slice());

        // Every word index lands in the text region, in increasing order,
        // and within bounds.
        assert!(enc.text_word_indices[0] >= prefix.len() as i64);
        assert!(enc.text_word_indices.windows(2).all(|w| w[0] <= w[1]));
        assert!(
            enc.text_word_indices
                .iter()
                .all(|&i| (i as usize) < enc.seq_len())
        );
    }

    #[test]
    fn the_prefix_is_identical_across_documents() {
        if !model_present() {
            return;
        }
        let tok = load_tokenizer();
        let prefix = build_prefix(&tok, &["person", "email"]).unwrap();
        let mut cache = WordIdCache::new(1024);

        let mut a = EncodedInput::default();
        let mut b = EncodedInput::default();
        build_input(
            &prefix,
            &words::prepare("one").words,
            &tok,
            &mut cache,
            None,
            &mut a,
        )
        .unwrap();
        build_input(
            &prefix,
            &words::prepare("something else").words,
            &tok,
            &mut cache,
            None,
            &mut b,
        )
        .unwrap();

        assert_eq!(&a.input_ids[..prefix.len()], &b.input_ids[..prefix.len()]);
        // ...which is exactly what makes precomputing it legitimate.
        assert_eq!(&a.input_ids[..prefix.len()], prefix.ids.as_slice());
    }

    #[test]
    fn word_cache_returns_the_same_ids_as_a_direct_encode() {
        if !model_present() {
            return;
        }
        let tok = load_tokenizer();
        let mut cache = WordIdCache::new(4); // tiny, to force the reset path
        for w in ["alpha", "beta", "gamma", "delta", "epsilon", "alpha"] {
            let cached = cache.get_or_insert(w, &tok).unwrap().to_vec();
            assert_eq!(
                cached,
                encode_word(&tok, w).unwrap(),
                "cache diverged for {w}"
            );
        }
    }

    #[test]
    fn unicode_documents_encode_without_panicking_and_stay_in_bounds() {
        if !model_present() {
            return;
        }
        let tok = load_tokenizer();
        let prefix = build_prefix(&tok, &["person", "email"]).unwrap();
        let mut cache = WordIdCache::new(1024);
        let mut enc = EncodedInput::default();

        for text in [
            "Михаил Ястребцов",
            "Jörg Müller",
            "José García",
            "François Dupont",
            "王小明",
            "",
            "   ",
            "emoji 👋 test",
        ] {
            let prepared = words::prepare(text);
            build_input(&prefix, &prepared.words, &tok, &mut cache, None, &mut enc).unwrap();
            assert_eq!(
                enc.num_words(),
                prepared.words.len(),
                "word count mismatch for {text:?}"
            );
            assert!(
                enc.text_word_indices
                    .iter()
                    .all(|&i| (i as usize) < enc.seq_len()),
                "out-of-bounds word index for {text:?}"
            );
        }
    }
}
