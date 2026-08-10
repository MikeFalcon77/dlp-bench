//! Deterministic adversarial transformations for robustness evaluation.
//!
//! The point is not to be exhaustive but to be *honest about offsets*: every
//! transformation rebuilds the text segment by segment and recomputes gold
//! spans as it goes, so a transformed example is as trustworthy as the original.
//! A transformation layer that mangled the annotations would silently invent
//! false negatives and make every model look worse than it is.
//!
//! All transformations are pure functions of `(example, seed)`.

use super::{Example, GoldEntity, Tag};
use crate::entity::EntityType;

/// Tiny deterministic PRNG (SplitMix64). Avoids a dependency and, more
/// importantly, avoids any chance of a rand-crate version change altering
/// results between runs.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform choice from a non-empty slice.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next_u64() % items.len() as u64) as usize]
    }

    pub fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// Uniform integer in `0..n`. Panics on `n == 0`.
    ///
    /// Modulo bias is negligible here (`n` is at most a few thousand against a
    /// 64-bit range) and the alternative — rejection sampling — would make the
    /// sequence depend on how many draws were rejected, which is a worse
    /// property for a benchmark that must be reproducible.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0) has no valid result");
        (self.next_u64() % n as u64) as usize
    }
}

/// A robustness transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// `john@example.com` -> `john [at] example [dot] com`, `john@example[.]com`, ...
    ObfuscateEmail,
    /// Vary phone separators: spaces, dashes, dots, parens.
    PhoneSeparators,
    /// Vary card separators: grouped by 4 with space / dash, or unseparated.
    CardSeparators,
    /// Upper/lower/mixed-case the entity surface (only where case is not
    /// semantically load-bearing).
    CaseShift,
    /// Collapse or expand whitespace around entities.
    WhitespaceNoise,
    /// Wrap the document in a Markdown fenced code block.
    MarkdownCodeBlock,
    /// Embed the document as a JSON string value.
    JsonQuoted,
    /// Rewrite as `key=value` lines.
    KeyValue,
    /// Insert hard line breaks between words.
    LineBreaks,
}

impl Transform {
    pub const ALL: &'static [Transform] = &[
        Transform::ObfuscateEmail,
        Transform::PhoneSeparators,
        Transform::CardSeparators,
        Transform::CaseShift,
        Transform::WhitespaceNoise,
        Transform::MarkdownCodeBlock,
        Transform::JsonQuoted,
        Transform::KeyValue,
        Transform::LineBreaks,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Transform::ObfuscateEmail => "obfuscate_email",
            Transform::PhoneSeparators => "phone_separators",
            Transform::CardSeparators => "card_separators",
            Transform::CaseShift => "case_shift",
            Transform::WhitespaceNoise => "whitespace_noise",
            Transform::MarkdownCodeBlock => "markdown_code_block",
            Transform::JsonQuoted => "json_quoted",
            Transform::KeyValue => "key_value",
            Transform::LineBreaks => "line_breaks",
        }
    }

    pub fn parse(s: &str) -> Option<Transform> {
        Transform::ALL.iter().copied().find(|t| t.as_str() == s)
    }

    /// Apply to `ex`, returning `None` when the transformation does not apply
    /// (e.g. email obfuscation on an example with no email).
    ///
    /// The returned example has a suffixed id, the `adversarial` tag, and gold
    /// spans recomputed against the new text.
    pub fn apply(self, ex: &Example, seed: u64) -> Option<Example> {
        let mut rng = Rng::new(seed ^ (self as u64).wrapping_mul(0x1000_0001B3));
        let out = match self {
            Transform::ObfuscateEmail => {
                rewrite_entities(ex, &[EntityType::Email], &mut rng, obfuscate_email)
            }
            Transform::PhoneSeparators => {
                rewrite_entities(ex, &[EntityType::Phone], &mut rng, respace_digits)
            }
            Transform::CardSeparators => {
                rewrite_entities(ex, &[EntityType::CreditCard], &mut rng, regroup_card)
            }
            Transform::CaseShift => rewrite_entities(
                ex,
                &[EntityType::Email, EntityType::Person, EntityType::Url],
                &mut rng,
                shift_case,
            ),
            Transform::WhitespaceNoise => wrap_document(ex, |t| {
                let mut s = String::with_capacity(t.len() + 16);
                for (i, line) in t.lines().enumerate() {
                    if i > 0 {
                        s.push('\n');
                    }
                    s.push_str(line);
                    s.push(' ');
                }
                s
            }),
            Transform::MarkdownCodeBlock => wrap_document(ex, |t| format!("```\n{t}\n```")),
            Transform::JsonQuoted => wrap_document(ex, |t| {
                // Only applies when escaping would not shift interior offsets;
                // see `wrap_document`'s contract.
                format!("{{\"prompt\": \"{t}\"}}")
            }),
            Transform::KeyValue => wrap_document(ex, |t| format!("input={t}")),
            Transform::LineBreaks => wrap_document(ex, |t| t.replace(". ", ".\n")),
        }?;
        Some(out)
    }
}

/// Rewrite the surface text of entities of the given types, rebuilding the
/// document and the annotations together.
///
/// This is the only sanctioned way to change text that overlaps an annotation.
fn rewrite_entities(
    ex: &Example,
    types: &[EntityType],
    rng: &mut Rng,
    f: impl Fn(&str, &mut Rng) -> Option<String>,
) -> Option<Example> {
    let mut ents: Vec<&GoldEntity> = ex.entities.iter().collect();
    ents.sort_by_key(|e| (e.start, e.end));

    let mut text = String::with_capacity(ex.text.len() + 32);
    let mut new_entities = Vec::with_capacity(ex.entities.len());
    let mut cursor = 0usize;
    let mut changed = false;

    for e in ents {
        text.push_str(&ex.text[cursor..e.start]);
        let original = &ex.text[e.start..e.end];
        let replacement = if types.contains(&e.entity_type) {
            f(original, rng)
        } else {
            None
        };
        let surface = match replacement {
            Some(s) if s != original => {
                changed = true;
                s
            }
            _ => original.to_owned(),
        };
        let start = text.len();
        text.push_str(&surface);
        new_entities.push(GoldEntity {
            entity_type: e.entity_type,
            start,
            end: text.len(),
        });
        cursor = e.end;
    }
    text.push_str(&ex.text[cursor..]);

    changed.then(|| finish(ex, text, new_entities))
}

/// Apply a document-level wrapper that does not modify any annotated region.
///
/// Returns `None` if the transformation would alter bytes inside an annotation,
/// because we cannot recompute those offsets reliably — better to skip the
/// example than to emit a mis-annotated one.
fn wrap_document(ex: &Example, f: impl Fn(&str) -> String) -> Option<Example> {
    let new_text = f(&ex.text);
    if new_text == ex.text {
        return None;
    }

    // Locate the original text verbatim inside the wrapper. If the wrapper
    // rewrote interior bytes (e.g. `LineBreaks` on ". "), fall back to
    // per-entity relocation and bail out if any entity surface is ambiguous.
    if let Some(offset) = new_text.find(&ex.text) {
        let entities = ex
            .entities
            .iter()
            .map(|e| GoldEntity {
                entity_type: e.entity_type,
                start: e.start + offset,
                end: e.end + offset,
            })
            .collect();
        return Some(finish(ex, new_text, entities));
    }

    let mut entities = Vec::with_capacity(ex.entities.len());
    let mut search_from = 0usize;
    for e in &ex.entities {
        let surface = &ex.text[e.start..e.end];
        let found = new_text[search_from..].find(surface)? + search_from;
        entities.push(GoldEntity {
            entity_type: e.entity_type,
            start: found,
            end: found + surface.len(),
        });
        search_from = found + surface.len();
    }
    Some(finish(ex, new_text, entities))
}

fn finish(ex: &Example, text: String, mut entities: Vec<GoldEntity>) -> Example {
    entities.sort_by_key(|e| (e.start, e.end));
    let mut tags = ex.tags.clone();
    if !tags.contains(&Tag::Adversarial) {
        tags.push(Tag::Adversarial);
    }
    Example {
        id: ex.id.clone(),
        source: ex.source.clone(),
        language: ex.language.clone(),
        text,
        entities,
        tags,
    }
}

// --- surface rewriters -----------------------------------------------------

fn obfuscate_email(s: &str, rng: &mut Rng) -> Option<String> {
    let (user, domain) = s.split_once('@')?;
    let style = *rng.pick(&[0u8, 1, 2, 3]);
    Some(match style {
        0 => format!(
            "{user} [at] {} [dot] {}",
            domain.rsplit_once('.')?.0,
            domain.rsplit_once('.')?.1
        ),
        1 => format!("{user}@{}", domain.replacen('.', "[.]", 1)),
        2 => format!("{user} (at) {domain}"),
        _ => format!("{user}[at]{}", domain.replace('.', "[dot]")),
    })
}

fn respace_digits(s: &str, rng: &mut Rng) -> Option<String> {
    let plus = s.starts_with('+');
    let digits: String = s.chars().filter(char::is_ascii_digit).collect();
    if digits.len() < 7 {
        return None;
    }
    let sep = *rng.pick(&[" ", "-", ".", ""]);
    // Regroup as 1-3-3-4 where the length allows; otherwise fall back to
    // uniform 3-digit groups.
    let grouped = if digits.len() == 11 {
        format!(
            "{}{sep}{}{sep}{}{sep}{}",
            &digits[0..1],
            &digits[1..4],
            &digits[4..7],
            &digits[7..11]
        )
    } else if digits.len() == 10 {
        format!(
            "{}{sep}{}{sep}{}",
            &digits[0..3],
            &digits[3..6],
            &digits[6..10]
        )
    } else {
        digits
            .as_bytes()
            .chunks(3)
            .map(|c| std::str::from_utf8(c).unwrap_or_default())
            .collect::<Vec<_>>()
            .join(sep)
    };
    Some(if plus { format!("+{grouped}") } else { grouped })
}

fn regroup_card(s: &str, rng: &mut Rng) -> Option<String> {
    let digits: String = s.chars().filter(char::is_ascii_digit).collect();
    if digits.len() < 13 {
        return None;
    }
    let sep = *rng.pick(&[" ", "-", ""]);
    Some(
        digits
            .as_bytes()
            .chunks(4)
            .map(|c| std::str::from_utf8(c).unwrap_or_default())
            .collect::<Vec<_>>()
            .join(sep),
    )
}

fn shift_case(s: &str, rng: &mut Rng) -> Option<String> {
    if !s.chars().any(char::is_alphabetic) {
        return None;
    }
    Some(if rng.bool() {
        s.to_uppercase()
    } else {
        s.to_lowercase()
    })
}

/// Expand a dataset with adversarial variants.
///
/// Ids are suffixed with the transformation name so a result row can always be
/// traced back to the base example and the transformation that produced it.
pub fn expand(examples: &[Example], transforms: &[Transform], seed: u64) -> Vec<Example> {
    let mut out = Vec::new();
    for ex in examples {
        for &t in transforms {
            if let Some(mut v) = t.apply(ex, seed) {
                v.id = format!("{}#{}", ex.id, t.as_str());
                out.push(v);
            }
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(text: &str, entities: Vec<(EntityType, usize, usize)>) -> Example {
        Example {
            id: "t".into(),
            source: None,
            language: Some("en".into()),
            text: text.into(),
            entities: entities
                .into_iter()
                .map(|(entity_type, start, end)| GoldEntity {
                    entity_type,
                    start,
                    end,
                })
                .collect(),
            tags: vec![Tag::Prose],
        }
    }

    /// The invariant that makes this module trustworthy: after any
    /// transformation, every gold span still slices to sensible text.
    fn assert_spans_valid(out: &Example) {
        for e in &out.entities {
            crate::span::validate(&out.text, e.start, e.end, false)
                .unwrap_or_else(|err| panic!("invalid span in {:?}: {err}", out.text));
            assert!(
                !out.text[e.start..e.end].trim().is_empty(),
                "span sliced to whitespace in {:?}",
                out.text
            );
        }
    }

    #[test]
    fn email_obfuscation_rewrites_text_and_moves_the_span_with_it() {
        let e = ex(
            "Mail john@example.com now",
            vec![(EntityType::Email, 5, 21)],
        );
        assert_eq!(&e.text[5..21], "john@example.com");

        let out = Transform::ObfuscateEmail.apply(&e, 42).unwrap();
        assert_ne!(out.text, e.text);
        assert_spans_valid(&out);
        let surface = &out.text[out.entities[0].start..out.entities[0].end];
        assert!(surface.starts_with("john"), "got {surface:?}");
        assert!(
            !surface.contains("@") || surface.contains("[."),
            "got {surface:?}"
        );
        assert!(out.text.starts_with("Mail ") && out.text.ends_with(" now"));
        assert!(out.tags.contains(&Tag::Adversarial));
    }

    #[test]
    fn all_transforms_preserve_span_validity_on_a_multi_entity_example() {
        let e = ex(
            "Contact John Smith at john@example.com or +1-415-555-1234. Card 4111 1111 1111 1111.",
            vec![
                (EntityType::Person, 8, 18),
                (EntityType::Email, 22, 38),
                (EntityType::Phone, 42, 57),
                (EntityType::CreditCard, 64, 83),
            ],
        );
        assert_eq!(&e.text[8..18], "John Smith");
        assert_eq!(&e.text[42..57], "+1-415-555-1234");
        assert_eq!(&e.text[64..83], "4111 1111 1111 1111");

        let mut applied = 0;
        for &t in Transform::ALL {
            if let Some(out) = t.apply(&e, 7) {
                assert_spans_valid(&out);
                assert_eq!(out.entities.len(), e.entities.len(), "{t:?} lost an entity");
                applied += 1;
            }
        }
        assert!(applied >= 7, "only {applied} transforms applied");
    }

    #[test]
    fn transforms_are_deterministic_for_a_given_seed() {
        let e = ex("Mail john@example.com", vec![(EntityType::Email, 5, 21)]);
        for &t in Transform::ALL {
            let a = t.apply(&e, 42);
            let b = t.apply(&e, 42);
            assert_eq!(a, b, "{t:?} is not deterministic");
        }
        // ...and the seed actually matters for the randomised ones.
        let seeds: std::collections::BTreeSet<_> = (0..16)
            .filter_map(|s| Transform::ObfuscateEmail.apply(&e, s).map(|v| v.text))
            .collect();
        assert!(seeds.len() > 1, "seed has no effect on email obfuscation");
    }

    /// Re-separating a number must never add, drop or reorder a digit —
    /// otherwise the "adversarial" example is just a different number and the
    /// gold annotation is a lie.
    fn assert_digits_preserved(t: Transform, e: &Example, expect: &str) {
        let mut variants = 0;
        for seed in 0..16 {
            // `None` is correct when the chosen separator reproduces the
            // original: no change means no adversarial variant.
            let Some(out) = t.apply(e, seed) else {
                continue;
            };
            variants += 1;
            assert_spans_valid(&out);
            let s = &out.text[out.entities[0].start..out.entities[0].end];
            let digits: String = s.chars().filter(char::is_ascii_digit).collect();
            assert_eq!(digits, expect, "{t:?} altered the digits: {s:?}");
        }
        assert!(variants > 0, "{t:?} produced no variants at all");
    }

    #[test]
    fn card_regrouping_preserves_the_digits() {
        let e = ex(
            "Card 4111-1111-1111-1111 ok",
            vec![(EntityType::CreditCard, 5, 24)],
        );
        assert_eq!(&e.text[5..24], "4111-1111-1111-1111");
        assert_digits_preserved(Transform::CardSeparators, &e, "4111111111111111");
    }

    #[test]
    fn phone_respacing_preserves_the_digits() {
        let e = ex("Call +1-415-555-1234 now", vec![(EntityType::Phone, 5, 20)]);
        assert_eq!(&e.text[5..20], "+1-415-555-1234");
        assert_digits_preserved(Transform::PhoneSeparators, &e, "14155551234");
    }

    #[test]
    fn wrappers_shift_offsets_by_the_prefix_length() {
        let e = ex("email john@example.com", vec![(EntityType::Email, 6, 22)]);
        let out = Transform::MarkdownCodeBlock.apply(&e, 1).unwrap();
        assert!(out.text.starts_with("```\n"));
        assert_eq!(out.entities[0].start, 6 + 4);
        assert_eq!(
            &out.text[out.entities[0].start..out.entities[0].end],
            "john@example.com"
        );
    }

    #[test]
    fn transforms_that_do_not_apply_return_none_rather_than_a_bogus_example() {
        let no_email = ex("Fix issue #4411.", vec![]);
        assert_eq!(Transform::ObfuscateEmail.apply(&no_email, 1), None);
        assert_eq!(Transform::CardSeparators.apply(&no_email, 1), None);
        // A phone too short to regroup is skipped, not mangled.
        let short = ex("call 12345", vec![(EntityType::Phone, 5, 10)]);
        assert_eq!(Transform::PhoneSeparators.apply(&short, 1), None);
    }

    #[test]
    fn unicode_entities_survive_transformation() {
        // Byte offsets, not char offsets: "Свяжитесь с " is 22 bytes, and
        // " по адресу " between the name and the email is another 19.
        let text = "Свяжитесь с Михаил Ястребцов по адресу m@example.com";
        let name_at = text.find("Михаил").unwrap();
        let mail_at = text.find("m@example.com").unwrap();
        let e = ex(
            text,
            vec![
                (
                    EntityType::Person,
                    name_at,
                    name_at + "Михаил Ястребцов".len(),
                ),
                (EntityType::Email, mail_at, mail_at + "m@example.com".len()),
            ],
        );
        assert_eq!((name_at, mail_at), (22, 72));
        assert_eq!(&e.text[22..53], "Михаил Ястребцов");
        assert_eq!(&e.text[72..85], "m@example.com");
        for &t in Transform::ALL {
            if let Some(out) = t.apply(&e, 3) {
                assert_spans_valid(&out);
            }
        }
    }

    #[test]
    fn expand_names_variants_after_their_transform() {
        let e = ex("Mail john@example.com", vec![(EntityType::Email, 5, 21)]);
        let out = expand(
            &[e],
            &[Transform::ObfuscateEmail, Transform::MarkdownCodeBlock],
            42,
        );
        let ids: Vec<_> = out.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, vec!["t#markdown_code_block", "t#obfuscate_email"]);
    }

    #[test]
    fn rng_below_is_bounded_and_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..500 {
            let x = a.below(7);
            assert!(x < 7);
            assert_eq!(x, b.below(7));
        }
        assert_eq!(Rng::new(1).below(1), 0);
        // Different seeds must diverge.
        let s1: Vec<_> = (0..20).map(|_| Rng::new(1).below(100)).collect();
        let s2: Vec<_> = (0..20).map(|_| Rng::new(2).below(100)).collect();
        assert_ne!(s1, s2);
    }

    #[test]
    fn transform_names_round_trip() {
        for &t in Transform::ALL {
            assert_eq!(Transform::parse(t.as_str()), Some(t));
        }
        assert_eq!(Transform::parse("nope"), None);
    }
}
