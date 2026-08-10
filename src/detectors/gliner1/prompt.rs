//! GLiNER1 prompt construction.
//!
//! The encoder input is a pre-tokenized sequence:
//!
//! ```text
//! <<ENT>> label0  <<ENT>> label1  …  <<SEP>>  w0 w1 w2 …
//! ```
//!
//! fed to the tokenizer as *already split words* so that each element keeps its
//! own word id. Special tokens (`[CLS]`/`[SEP]`) are added normally.
//!
//! `words_mask` is how the model recovers word representations from sub-tokens:
//! it holds the **1-based text-word index at the first sub-token of each text
//! word**, and 0 everywhere else — including every prompt position. The model
//! gathers on it, so an off-by-one here shifts every span the model can return.
//!
//! We deliberately do **not** use the tokenizer's offsets. With pre-tokenized
//! input they are relative to each element rather than to the document, which
//! would be useless for span reporting. Instead we split the text ourselves
//! with byte offsets ([`crate::detectors::words`]) and index back into that.

use std::sync::LazyLock;

use regex::Regex;
use tokenizers::Tokenizer;

use crate::detectors::words::{Word, split_with};
use crate::error::DetectorError;

pub const ENT_TOKEN: &str = "<<ENT>>";
pub const SEP_TOKEN: &str = "<<SEP>>";
/// Asserted at load; a tokenizer that disagrees is not this model's.
pub const ENT_TOKEN_ID: u32 = 50368;
pub const SEP_TOKEN_ID: u32 = 50369;

/// GLiNER1's `whitespace` word splitter.
///
/// Simpler than GLiNER2's — no URL or email awareness — so `john@example.com`
/// becomes five words here and one there. Matching the model's own splitter is
/// mandatory: it defines what a "word" index means.
static WORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\w+(?:[-_]\w+)*|\S").expect("gliner1 word regex"));

pub fn split_words(text: &str, out: &mut Vec<Word>) {
    split_with(&WORD_RE, text, false, out);
}

/// Per-document encoder input.
#[derive(Debug, Default)]
pub struct EncodedInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub words_mask: Vec<i64>,
    pub num_words: usize,
}

impl EncodedInput {
    pub fn seq_len(&self) -> usize {
        self.input_ids.len()
    }
}

/// Verify the tokenizer really is this model's.
pub fn assert_special_tokens(tok: &Tokenizer) -> Result<(), DetectorError> {
    for (name, want) in [(ENT_TOKEN, ENT_TOKEN_ID), (SEP_TOKEN, SEP_TOKEN_ID)] {
        let enc = tok
            .encode(name, false)
            .map_err(|e| DetectorError::Tokenizer(format!("encoding {name}: {e}")))?;
        if enc.get_ids() != [want] {
            return Err(DetectorError::Tokenizer(format!(
                "special token {name} encodes to {:?}, expected [{want}]; \
                 this is not the tokenizer this detector was built against",
                enc.get_ids()
            )));
        }
    }
    Ok(())
}

/// Build the encoder input for one document.
pub fn build_input(
    tok: &Tokenizer,
    labels: &[&str],
    words: &[Word],
    text: &str,
    out: &mut EncodedInput,
) -> Result<(), DetectorError> {
    // `<<ENT>> label` per class, then `<<SEP>>`, then the text words.
    let mut seq: Vec<&str> = Vec::with_capacity(labels.len() * 2 + 1 + words.len());
    for l in labels {
        seq.push(ENT_TOKEN);
        seq.push(l);
    }
    seq.push(SEP_TOKEN);
    // Index of the first *text* word within `seq`; everything before it is
    // prompt and must stay masked out.
    let text_start = seq.len();
    for w in words {
        seq.push(w.text(text));
    }

    let enc = tok
        .encode(seq, true)
        .map_err(|e| DetectorError::Tokenizer(format!("encoding failed: {e}")))?;

    out.input_ids.clear();
    out.input_ids
        .extend(enc.get_ids().iter().map(|&i| i64::from(i)));
    out.attention_mask.clear();
    out.attention_mask.resize(out.input_ids.len(), 1);
    out.num_words = words.len();

    out.words_mask.clear();
    out.words_mask.resize(out.input_ids.len(), 0);
    let word_ids = enc.get_word_ids();
    let mut prev: Option<u32> = None;
    for (i, wid) in word_ids.iter().enumerate() {
        if let Some(w) = *wid
            && Some(w) != prev
            && (w as usize) >= text_start
        {
            // 1-based, and only for text words — prompt positions stay 0.
            out.words_mask[i] = (w as usize - text_start + 1) as i64;
        }
        prev = *wid;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIR: &str = "models/gliner-pii-edge";

    fn present() -> bool {
        std::path::Path::new(DIR).join("tokenizer.json").is_file()
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_file(std::path::Path::new(DIR).join("tokenizer.json")).unwrap()
    }

    fn words_of(text: &str) -> Vec<Word> {
        let mut v = Vec::new();
        split_words(text, &mut v);
        v
    }

    #[test]
    fn special_token_ids_match_the_shipped_tokenizer() {
        if !present() {
            eprintln!("SKIP: run `bash scripts/fetch-models.sh gliner-pii-edge`");
            return;
        }
        assert_special_tokens(&tokenizer()).unwrap();
    }

    #[test]
    fn words_mask_is_one_based_and_only_marks_first_subtokens_of_text_words() {
        if !present() {
            return;
        }
        let tok = tokenizer();
        let text = "Contact John Smith at john@example.com";
        let words = words_of(text);
        let labels = ["person", "email address", "credit card"];
        let mut enc = EncodedInput::default();
        build_input(&tok, &labels, &words, text, &mut enc).unwrap();

        assert_eq!(enc.num_words, words.len());
        assert_eq!(enc.attention_mask.len(), enc.seq_len());
        assert_eq!(enc.words_mask.len(), enc.seq_len());

        // Exactly one non-zero per text word, and they are 1..=num_words.
        let nonzero: Vec<i64> = enc.words_mask.iter().copied().filter(|&x| x != 0).collect();
        assert_eq!(nonzero.len(), words.len(), "one marker per word");
        assert_eq!(nonzero, (1..=words.len() as i64).collect::<Vec<_>>());

        // Markers are strictly increasing in position — a gather relies on it.
        let positions: Vec<usize> = enc
            .words_mask
            .iter()
            .enumerate()
            .filter(|&(_, &m)| m != 0)
            .map(|(i, _)| i)
            .collect();
        assert!(positions.windows(2).all(|p| p[0] < p[1]));
    }

    #[test]
    fn prompt_positions_are_never_marked() {
        if !present() {
            return;
        }
        let tok = tokenizer();
        let text = "alpha";
        let words = words_of(text);
        let labels = ["person", "email address"];
        let mut enc = EncodedInput::default();
        build_input(&tok, &labels, &words, text, &mut enc).unwrap();

        // The single text word must be marked at the *last* marker position,
        // after all the prompt tokens.
        let marked: Vec<usize> = enc
            .words_mask
            .iter()
            .enumerate()
            .filter(|&(_, &m)| m != 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(marked.len(), 1);

        // The prompt contains one <<ENT>> per label and exactly one <<SEP>>.
        let ent = enc
            .input_ids
            .iter()
            .filter(|&&i| i == i64::from(ENT_TOKEN_ID))
            .count();
        let sep = enc
            .input_ids
            .iter()
            .filter(|&&i| i == i64::from(SEP_TOKEN_ID))
            .count();
        assert_eq!(ent, labels.len());
        assert_eq!(sep, 1);
        // ...and the marker sits after the <<SEP>>.
        let sep_at = enc
            .input_ids
            .iter()
            .position(|&i| i == i64::from(SEP_TOKEN_ID))
            .unwrap();
        assert!(marked[0] > sep_at);
    }

    #[test]
    fn adding_labels_lengthens_the_prompt_but_not_the_word_count() {
        if !present() {
            return;
        }
        let tok = tokenizer();
        let text = "Contact John Smith";
        let words = words_of(text);
        let mut a = EncodedInput::default();
        let mut b = EncodedInput::default();
        build_input(&tok, &["person"], &words, text, &mut a).unwrap();
        build_input(
            &tok,
            &["person", "email address", "credit card"],
            &words,
            text,
            &mut b,
        )
        .unwrap();

        assert!(b.seq_len() > a.seq_len());
        assert_eq!(a.num_words, b.num_words);
        // Same number of markers regardless of prompt length — this is the
        // property that makes the word indices comparable across schemas.
        let count = |e: &EncodedInput| e.words_mask.iter().filter(|&&x| x != 0).count();
        assert_eq!(count(&a), count(&b));
    }

    #[test]
    fn multibyte_text_is_marked_word_for_word() {
        if !present() {
            return;
        }
        let tok = tokenizer();
        for text in [
            "Свяжитесь с Михаил Ястребцов",
            "Jörg Müller wohnt in München",
            "王小明在台北工作",
        ] {
            let words = words_of(text);
            let mut enc = EncodedInput::default();
            build_input(&tok, &["person"], &words, text, &mut enc).unwrap();
            let nonzero = enc.words_mask.iter().filter(|&&x| x != 0).count();
            assert_eq!(nonzero, words.len(), "word/marker mismatch for {text:?}");
        }
    }

    #[test]
    fn empty_word_list_still_produces_a_valid_prompt() {
        if !present() {
            return;
        }
        let tok = tokenizer();
        let mut enc = EncodedInput::default();
        build_input(&tok, &["person"], &[], "", &mut enc).unwrap();
        assert_eq!(enc.num_words, 0);
        assert!(enc.words_mask.iter().all(|&x| x == 0));
        assert!(!enc.input_ids.is_empty(), "prompt tokens are still there");
    }
}
