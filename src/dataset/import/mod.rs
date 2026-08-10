//! Importers for third-party evaluation corpora.
//!
//! An importer is the only place allowed to know a foreign schema or a foreign
//! offset convention. Everything downstream sees the crate's own JSONL with
//! UTF-8 byte offsets, so a corpus with character offsets, a different label
//! vocabulary or overlapping gold spans is normalised exactly once, here, with
//! the conversion verified rather than assumed.

pub mod presidio;
