//! Text preparation and word splitting, ported from the GLiNER2 reference.
//!
//! Two things here look wrong and are not:
//!
//! 1. **A trailing `.` is appended to the text before encoding.** The reference
//!    `_collate_batch` does this unconditionally, and empty text becomes `"."`.
//!    It changes the encoder input and therefore every probability, so it must
//!    be replicated. All model-side offsets are into the *prepared* string.
//!
//! 2. **The splitter is not a whitespace splitter** despite the upstream class
//!    being named `WhitespaceTokenSplitter`. It is URL-, email- and
//!    `@handle`-aware, keeps `foo-bar` and `foo_bar` as single words, and falls
//!    back to one non-space character (so punctuation becomes its own word).
//!    Alternation order is load-bearing.
//!
//! Getting either wrong shifts every word index, which shifts every span.

use std::sync::LazyLock;

use regex::Regex;

/// Port of the reference splitter's pattern.
///
/// Rust's `regex` uses leftmost-first alternation like Python's `re`, so the
/// URL / email / handle / word / single-char preference order is preserved.
static WORD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?ix)
          (?: https?://[^\s]+ | www\.[^\s]+ )
        | [a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}
        | @[a-z0-9_]+
        | \w+(?:[-_]\w+)*
        | \S",
    )
    .expect("gliner2 word regex")
});

/// One word, carrying UTF-8 byte offsets into the prepared text.
///
/// `lower` is what gets tokenized (the reference lowercases text words, though
/// *not* schema labels); the offsets come from the un-lowercased match, so
/// case-folding that changes length cannot corrupt them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    pub lower: Box<str>,
    pub start: usize,
    pub end: usize,
}

/// Text as the model will actually see it, plus its word segmentation.
#[derive(Debug, Clone)]
pub struct PreparedText {
    /// Possibly `raw + "."`, or `"."` when `raw` was empty.
    pub text: String,
    /// True when a `.` was appended that is not in the caller's original text.
    pub appended_dot: bool,
    pub words: Vec<Word>,
}

impl PreparedText {
    /// Length of the caller's original text, i.e. the largest offset a
    /// returned span may legitimately reach.
    pub fn original_len(&self) -> usize {
        if self.appended_dot {
            self.text.len() - 1
        } else {
            self.text.len()
        }
    }
}

/// Apply the reference's text fixup.
pub fn prepare_text(raw: &str) -> (String, bool) {
    if raw.is_empty() {
        return (".".to_owned(), true);
    }
    if raw.ends_with(['.', '!', '?']) {
        return (raw.to_owned(), false);
    }
    let mut s = String::with_capacity(raw.len() + 1);
    s.push_str(raw);
    s.push('.');
    (s, true)
}

/// Split prepared text into words.
pub fn split_words(text: &str, out: &mut Vec<Word>) {
    out.clear();
    for m in WORD_RE.find_iter(text) {
        out.push(Word {
            lower: m.as_str().to_lowercase().into_boxed_str(),
            start: m.start(),
            end: m.end(),
        });
    }
}

/// Prepare and split in one step.
pub fn prepare(raw: &str) -> PreparedText {
    let (text, appended_dot) = prepare_text(raw);
    let mut words = Vec::new();
    split_words(&text, &mut words);
    PreparedText {
        text,
        appended_dot,
        words,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        let mut v = Vec::new();
        split_words(text, &mut v);
        v.iter().map(|w| w.lower.to_string()).collect()
    }

    /// Offsets must always slice back to the matched text.
    fn assert_offsets_consistent(text: &str) {
        let mut v = Vec::new();
        split_words(text, &mut v);
        for w in &v {
            assert!(
                text.is_char_boundary(w.start),
                "start {} not a boundary in {text:?}",
                w.start
            );
            assert!(
                text.is_char_boundary(w.end),
                "end {} not a boundary in {text:?}",
                w.end
            );
            assert_eq!(
                text[w.start..w.end].to_lowercase(),
                *w.lower,
                "offsets do not slice back to the word in {text:?}"
            );
        }
        // Words are non-overlapping and in order.
        for pair in v.windows(2) {
            assert!(
                pair[0].end <= pair[1].start,
                "overlapping words in {text:?}"
            );
        }
    }

    #[test]
    fn trailing_dot_is_appended_exactly_as_the_reference_does() {
        assert_eq!(prepare_text("hello"), ("hello.".to_owned(), true));
        assert_eq!(prepare_text("hello."), ("hello.".to_owned(), false));
        assert_eq!(prepare_text("hello!"), ("hello!".to_owned(), false));
        assert_eq!(prepare_text("hello?"), ("hello?".to_owned(), false));
        assert_eq!(prepare_text(""), (".".to_owned(), true));
        // Trailing whitespace does not count as terminal punctuation.
        assert_eq!(prepare_text("hello "), ("hello .".to_owned(), true));
    }

    #[test]
    fn prepared_text_reports_the_original_length() {
        let p = prepare("Contact me");
        assert!(p.appended_dot);
        assert_eq!(p.original_len(), "Contact me".len());
        let q = prepare("Contact me.");
        assert!(!q.appended_dot);
        assert_eq!(q.original_len(), "Contact me.".len());
    }

    #[test]
    fn urls_and_emails_stay_whole() {
        // The whole point of the custom pattern: these must not fragment.
        assert_eq!(
            words("see https://example.com/a?b=1 now"),
            vec!["see", "https://example.com/a?b=1", "now"]
        );
        assert_eq!(
            words("mail john.smith+tag@example.co.uk today"),
            vec!["mail", "john.smith+tag@example.co.uk", "today"]
        );
        assert_eq!(
            words("visit www.example.com ok"),
            vec!["visit", "www.example.com", "ok"]
        );
        assert_eq!(words("ping @octocat now"), vec!["ping", "@octocat", "now"]);
    }

    #[test]
    fn hyphenated_and_underscored_tokens_stay_whole() {
        assert_eq!(
            words("well-known api_key foo-bar-baz"),
            vec!["well-known", "api_key", "foo-bar-baz"]
        );
    }

    #[test]
    fn punctuation_becomes_its_own_word() {
        assert_eq!(words("a, b."), vec!["a", ",", "b", "."]);
        assert_eq!(
            words("{\"k\": 1}"),
            vec!["{", "\"", "k", "\"", ":", "1", "}"]
        );
    }

    #[test]
    fn alternation_order_prefers_the_url_over_its_pieces() {
        // If `\w+` were tried first, this would split into `https`, `example`...
        let w = words("https://user@host.example.com/path");
        assert_eq!(w.len(), 1, "URL fragmented into {w:?}");
    }

    #[test]
    fn offsets_are_byte_exact_for_non_ascii_text() {
        for text in [
            "Свяжитесь с Михаил Ястребцов по адресу m@example.com.",
            "Jörg Müller wohnt in München.",
            "José García vive en Málaga.",
            "François Dupont habite à Paris.",
            "王小明在台北工作。",
            "emoji 👋 then text.",
            "combining e\u{0301}tude.",
            "nbsp\u{00a0}separated words.",
        ] {
            assert_offsets_consistent(text);
        }
    }

    #[test]
    fn cyrillic_words_are_split_as_words_not_bytes() {
        let w = words("Михаил Ястребцов");
        assert_eq!(w, vec!["михаил", "ястребцов"]);

        let mut v = Vec::new();
        split_words("Михаил Ястребцов", &mut v);
        assert_eq!(v[0].start, 0);
        assert_eq!(v[0].end, 12, "6 Cyrillic chars at 2 bytes each");
        assert_eq!(v[1].start, 13);
    }

    #[test]
    fn cjk_text_is_segmented_and_offsets_hold() {
        let mut v = Vec::new();
        split_words("王小明在台北", &mut v);
        // `\w` matches CJK, so this is one word; what matters is that the
        // offsets round-trip.
        assert_offsets_consistent("王小明在台北");
        assert_eq!(
            v.iter().map(|w| w.end - w.start).sum::<usize>(),
            "王小明在台北".len()
        );
    }

    #[test]
    fn empty_and_whitespace_only_text_are_handled() {
        assert!(words("").is_empty());
        assert!(words("   \n\t ").is_empty());
        // ...but `prepare` guarantees at least one word, because the reference
        // turns empty text into ".".
        assert_eq!(prepare("").words.len(), 1);
        assert_eq!(prepare("   ").words.len(), 1);
    }

    #[test]
    fn structured_input_shapes_split_sensibly() {
        assert_offsets_consistent("DATABASE_URL=postgres://admin:secret@prod-db.internal/main");
        assert_offsets_consistent(
            "{\"customer\": \"John Smith\", \"email\": \"john@example.com\"}",
        );
        assert_offsets_consistent("customer:\n  name: John Smith\n  phone: +1-415-555-1234");
        assert_offsets_consistent("SELECT * FROM customers WHERE email = 'john@example.com';");
        assert_offsets_consistent("2026-08-08 user=john@example.com ip=10.20.30.40 login failed");

        // A key=value line splits into key, `=`, value — the value being one
        // word is what lets a span land exactly on it.
        let w = words("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE");
        assert_eq!(w, vec!["aws_access_key_id", "=", "akiaiosfodnn7example"]);
    }

    #[test]
    fn splitting_is_deterministic() {
        let text = "Contact John Smith at john@example.com or +1-415-555-1234.";
        let first = words(text);
        for _ in 0..16 {
            assert_eq!(words(text), first);
        }
    }
}
