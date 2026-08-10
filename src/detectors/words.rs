//! Word segmentation shared by the span-based detectors.
//!
//! GLiNER models score *words*, not sub-tokens, so the word boundaries decide
//! every span the model can possibly emit. They are also where offset bugs
//! live, which is why the [`Word`] type carries UTF-8 byte offsets from the
//! moment of splitting and never re-derives them.
//!
//! The two GLiNER generations use **different splitters** and the difference is
//! not cosmetic — it changes `num_words`, and therefore every index the model
//! returns:
//!
//! * GLiNER1 (`words_splitter_type: whitespace`) uses `\w+(?:[-_]\w+)*|\S`.
//! * GLiNER2 extends that with URL-, email- and `@handle`-aware alternatives,
//!   so `john@example.com` is one word there and five words here.

use regex::Regex;

/// One word, carrying UTF-8 byte offsets into the text it was split from.
///
/// `lower` is what gets tokenized where a model expects lowercased input;
/// the offsets come from the un-lowercased match, so a case fold that changes
/// length cannot corrupt them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    pub lower: Box<str>,
    pub start: usize,
    pub end: usize,
}

impl Word {
    /// The original (un-lowercased) slice this word came from.
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start..self.end]
    }
}

/// Split `text` with `re`, reusing the caller's buffer.
///
/// `lowercase` is a per-model decision: the GLiNER2 reference lowercases text
/// words before tokenizing, GLiNER1 does not.
pub fn split_with(re: &Regex, text: &str, lowercase: bool, out: &mut Vec<Word>) {
    out.clear();
    for m in re.find_iter(text) {
        let s = m.as_str();
        out.push(Word {
            lower: if lowercase {
                s.to_lowercase().into_boxed_str()
            } else {
                s.into()
            },
            start: m.start(),
            end: m.end(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    static GLINER1_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\w+(?:[-_]\w+)*|\S").unwrap());

    fn split(text: &str) -> Vec<Word> {
        let mut v = Vec::new();
        split_with(&GLINER1_RE, text, false, &mut v);
        v
    }

    #[test]
    fn offsets_slice_back_to_the_word() {
        for text in [
            "Contact John Smith at john@example.com",
            "Свяжитесь с Михаил Ястребцов",
            "Jörg Müller, IBAN DE89370400440532013000",
            "王小明在台北",
        ] {
            for w in split(text) {
                assert!(text.is_char_boundary(w.start) && text.is_char_boundary(w.end));
                assert_eq!(w.text(text), &*w.lower, "offsets do not match the word");
            }
        }
    }

    #[test]
    fn words_are_ordered_and_non_overlapping() {
        let text = "a bb ccc dddd";
        let v = split(text);
        assert!(v.windows(2).all(|p| p[0].end <= p[1].start));
        assert_eq!(v.len(), 4);
    }

    /// The difference that matters: GLiNER1's splitter shatters an email,
    /// GLiNER2's keeps it whole. Getting these two confused silently shifts
    /// every word index the model returns.
    #[test]
    fn gliner1_splits_an_email_that_gliner2_keeps_whole() {
        let text = "mail john@example.com now";
        let g1: Vec<_> = split(text).iter().map(|w| w.lower.to_string()).collect();
        assert_eq!(g1, vec!["mail", "john", "@", "example", ".", "com", "now"]);

        let mut g2 = Vec::new();
        crate::detectors::gliner2::words::split_words(text, &mut g2);
        let g2: Vec<_> = g2.iter().map(|w| w.lower.to_string()).collect();
        assert_eq!(g2, vec!["mail", "john@example.com", "now"]);
    }

    #[test]
    fn lowercasing_is_opt_in_and_does_not_move_offsets() {
        let text = "Jörg MÜLLER";
        let mut lower = Vec::new();
        split_with(&GLINER1_RE, text, true, &mut lower);
        let raw = split(text);
        assert_eq!(lower[1].lower.as_ref(), "müller");
        assert_eq!(raw[1].lower.as_ref(), "MÜLLER");
        // The offsets are identical either way — that is the point.
        assert_eq!(
            lower.iter().map(|w| (w.start, w.end)).collect::<Vec<_>>(),
            raw.iter().map(|w| (w.start, w.end)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn empty_and_whitespace_only_text_yield_no_words() {
        assert!(split("").is_empty());
        assert!(split("   \n\t ").is_empty());
    }
}
