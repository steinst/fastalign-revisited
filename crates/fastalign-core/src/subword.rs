//! Subword alignment support: word boundaries of a segmented corpus and the projection of piece
//! alignments to word alignments.
//!
//! The aligner itself never knows about subwords: a segmented corpus is an ordinary corpus whose
//! tokens are pieces. This module recovers *which word each piece belongs to* from the pieces'
//! text alone, and turns the Pharaoh links between pieces into links between words:
//!
//! * [`Boundary::Start`]`(m)` — the SentencePiece convention: a piece that begins with the
//!   marker `m` (`▁` by default) begins a word; the others continue the current word.
//! * [`Boundary::Continue`]`(m)` — the subword-nmt convention: a piece that ends with `m`
//!   (`@@`) continues into the next piece; the others end a word.
//!
//! In both, the first piece of a line begins word 0. A link `(p, q)` between pieces becomes
//! `(word(p), word(q))`; duplicates collapse, and a word none of whose pieces is linked stays
//! unlinked. Nothing here touches training, decoding or the word-level output.

use rayon::prelude::*;

use crate::corpus::{Corpus, Vocab};
use crate::symmetrize::Link;

/// How a segmented corpus marks word boundaries (see the module documentation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// A piece starting with these bytes begins a word.
    Start(Vec<u8>),
    /// A piece ending with these bytes continues into the next piece.
    Continue(Vec<u8>),
}

/// SentencePiece's word-start marker, U+2581.
pub const DEFAULT_MARKER: &str = "\u{2581}";
/// subword-nmt's continuation marker.
pub const DEFAULT_CONTINUATION: &str = "@@";

impl Boundary {
    /// The default boundary: SentencePiece's `▁` as a word-start prefix.
    pub fn sentencepiece() -> Self {
        Boundary::Start(DEFAULT_MARKER.as_bytes().to_vec())
    }

    /// The marker bytes.
    pub fn marker(&self) -> &[u8] {
        match self {
            Boundary::Start(m) | Boundary::Continue(m) => m,
        }
    }

    /// `"start"` or `"continue"`.
    pub fn kind(&self) -> &'static str {
        match self {
            Boundary::Start(_) => "start",
            Boundary::Continue(_) => "continue",
        }
    }

    /// Build from a kind (`"start"` / `"continue"`) and a non-empty marker.
    pub fn from_parts(kind: &str, marker: &[u8]) -> Result<Self, String> {
        if marker.is_empty() {
            return Err("the subword marker must not be empty".into());
        }
        if marker.iter().any(|b| b.is_ascii_whitespace()) {
            return Err("the subword marker must not contain whitespace".into());
        }
        match kind {
            "start" => Ok(Boundary::Start(marker.to_vec())),
            "continue" => Ok(Boundary::Continue(marker.to_vec())),
            _ => Err(format!("unknown boundary kind '{kind}'")),
        }
    }

    /// Per vocabulary id: does the piece have the marker (prefix for `Start`, suffix for
    /// `Continue`)?
    pub fn classify(&self, vocab: &Vocab) -> Vec<bool> {
        (0..vocab.len() as u32)
            .map(|id| {
                let w = vocab.word(id);
                match self {
                    Boundary::Start(m) => w.starts_with(m),
                    Boundary::Continue(m) => w.ends_with(m),
                }
            })
            .collect()
    }

    /// Word index of every piece of one line, appended to `out` (cleared first); returns the
    /// number of words. `marked` is [`Boundary::classify`]'s result for the line's vocabulary.
    pub fn word_map(&self, tokens: &[u32], marked: &[bool], out: &mut Vec<u32>) -> usize {
        out.clear();
        let mut word = 0u32;
        let mut prev_continues = false;
        for (k, &t) in tokens.iter().enumerate() {
            let m = marked[t as usize];
            let new_word = match self {
                Boundary::Start(_) => m && k > 0,
                Boundary::Continue(_) => k > 0 && !prev_continues,
            };
            if new_word {
                word += 1;
            }
            prev_continues = m;
            out.push(word);
        }
        if tokens.is_empty() {
            0
        } else {
            word as usize + 1
        }
    }
}

/// Which order the projected links of a line come in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkOrder {
    /// By second-side word, then first-side word: fast_align's order for a forward direction.
    BySecond,
    /// By first-side word, then second-side word: the order of an inverse direction.
    ByFirst,
}

impl LinkOrder {
    /// The order a direction's word-level output has.
    pub fn of(inverse: bool) -> Self {
        if inverse {
            LinkOrder::ByFirst
        } else {
            LinkOrder::BySecond
        }
    }
}

/// Project the piece links of one line: `(p, q)` → `(first_map[p], second_map[q])`, duplicates
/// dropped, sorted by `order`. A link outside the line is an error.
pub fn project_links(
    links: &[Link],
    first_map: &[u32],
    second_map: &[u32],
    order: LinkOrder,
    out: &mut Vec<Link>,
) -> Result<(), String> {
    out.clear();
    for &(p, q) in links {
        let (Some(&a), Some(&b)) = (first_map.get(p as usize), second_map.get(q as usize)) else {
            return Err(format!(
                "link {p}-{q} is outside the line ({} and {} pieces)",
                first_map.len(),
                second_map.len()
            ));
        };
        out.push((a, b));
    }
    match order {
        LinkOrder::BySecond => out.sort_unstable_by_key(|&(a, b)| (b, a)),
        LinkOrder::ByFirst => out.sort_unstable(),
    }
    out.dedup();
    Ok(())
}

/// Parse one Pharaoh line into its links and the (verbatim) ` ||| score` suffix, if any.
/// Tokens that are not `a-b` are ignored.
fn parse_pharaoh_line(line: &[u8]) -> (Vec<Link>, Option<&[u8]>) {
    let cut = line.windows(5).position(|w| w == b" ||| ");
    let (body, suffix) = match cut {
        Some(c) => (&line[..c], Some(&line[c + 5..])),
        None => (line, None),
    };
    let num = |s: &[u8]| -> Option<u32> {
        if s.is_empty() || s.len() > 10 || !s.iter().all(u8::is_ascii_digit) {
            return None;
        }
        std::str::from_utf8(s).ok()?.parse().ok()
    };
    let links = body
        .split(|&b| b == b' ')
        .filter_map(|t| {
            let dash = t.iter().position(|&b| b == b'-')?;
            Some((num(&t[..dash])?, num(&t[dash + 1..])?))
        })
        .collect();
    (links, suffix)
}

/// Project the Pharaoh alignments of a segmented `corpus` (one text line per corpus line, as the
/// trainer writes them, ` ||| score` suffixes kept) from pieces to words.
///
/// `inverse` selects the order of the output links ([`LinkOrder::of`]). The text must have
/// exactly one line per corpus line.
pub fn project_text(
    corpus: &Corpus,
    boundary: &Boundary,
    text: &[u8],
    inverse: bool,
) -> Result<Vec<u8>, String> {
    let mut lines: Vec<&[u8]> = text.split(|&b| b == b'\n').collect();
    if lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    if lines.len() != corpus.len() {
        return Err(format!(
            "the alignments have {} lines but the corpus has {}",
            lines.len(),
            corpus.len()
        ));
    }
    let marked = boundary.classify(&corpus.vocab);
    let order = LinkOrder::of(inverse);
    const CHUNK: usize = 4096;
    let starts: Vec<usize> = (0..lines.len()).step_by(CHUNK).collect();
    let parts: Result<Vec<Vec<u8>>, String> = starts
        .par_iter()
        .map(|&s| {
            let mut buf = Vec::new();
            let (mut fm, mut sm) = (Vec::new(), Vec::new());
            let mut projected = Vec::new();
            for (k, line) in lines
                .iter()
                .enumerate()
                .take((s + CHUNK).min(lines.len()))
                .skip(s)
            {
                let (links, suffix) = parse_pharaoh_line(line);
                boundary.word_map(corpus.first.line(k), &marked, &mut fm);
                boundary.word_map(corpus.second.line(k), &marked, &mut sm);
                project_links(&links, &fm, &sm, order, &mut projected)
                    .map_err(|e| format!("line {}: {e}", k + 1))?;
                crate::symmetrize::format_links(&projected, &mut buf);
                if let Some(sf) = suffix {
                    buf.extend_from_slice(b" ||| ");
                    buf.extend_from_slice(sf);
                }
                buf.push(b'\n');
            }
            Ok(buf)
        })
        .collect();
    Ok(parts?.concat())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::LoadOptions;

    fn corpus(text: &str) -> Corpus {
        Corpus::from_joined_bytes(text.as_bytes(), &LoadOptions::default())
    }

    fn map(b: &Boundary, c: &Corpus, k: usize) -> Vec<u32> {
        let marked = b.classify(&c.vocab);
        let mut out = Vec::new();
        b.word_map(c.first.line(k), &marked, &mut out);
        out
    }

    #[test]
    fn start_marker_word_map() {
        let c = corpus("▁un ▁gra nd ▁chat ||| ▁a\n");
        assert_eq!(map(&Boundary::sentencepiece(), &c, 0), [0, 1, 1, 2]);
        // a first piece without the marker still starts word 0; the marker alone is a piece
        let c = corpus("x y ▁ z ▁w ||| ▁a\n");
        assert_eq!(map(&Boundary::sentencepiece(), &c, 0), [0, 0, 1, 1, 2]);
    }

    #[test]
    fn continuation_word_map() {
        let b = Boundary::Continue(b"@@".to_vec());
        let c = corpus("un gra@@ nd@@ e chat ||| a\n");
        assert_eq!(map(&b, &c, 0), [0, 1, 1, 1, 2]);
        // a dangling continuation at the end of a line ends the word anyway
        let c = corpus("a@@ ||| a\n");
        assert_eq!(map(&b, &c, 0), [0]);
    }

    #[test]
    fn projection_merges_orders_and_drops() {
        let mut out = Vec::new();
        let (f, s) = ([0, 1, 1, 2], [0, 0, 1]);
        project_links(
            &[(0, 0), (1, 1), (2, 0), (3, 2), (1, 0)],
            &f,
            &s,
            LinkOrder::BySecond,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [(0, 0), (1, 0), (2, 1)]);
        project_links(&[(3, 2), (0, 1)], &f, &s, LinkOrder::ByFirst, &mut out).unwrap();
        assert_eq!(out, [(0, 0), (2, 1)]);
        assert!(project_links(&[(4, 0)], &f, &s, LinkOrder::ByFirst, &mut out).is_err());
        project_links(&[], &f, &s, LinkOrder::ByFirst, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn project_text_keeps_scores_and_checks_lines() {
        let c = corpus("▁a b ▁c ||| ▁x ▁y z\n\n▁q ||| ▁r\n");
        let b = Boundary::sentencepiece();
        // lines: 3 pieces / 3 pieces; empty; 1 / 1
        let text = b"0-0 1-0 2-2 ||| -1.5\n\n0-0\n";
        let got = project_text(&c, &b, text, false).unwrap();
        assert_eq!(got, b"0-0 1-1 ||| -1.5\n\n0-0\n");
        assert!(project_text(&c, &b, b"0-0\n", false).is_err());
        assert!(project_text(&c, &b, b"9-0\n\n0-0\n", false).is_err());
        // no final newline is fine; the output always ends with one
        assert_eq!(
            project_text(&c, &b, b"0-0\n\n0-0", true).unwrap(),
            b"0-0\n\n0-0\n"
        );
    }

    #[test]
    fn identity_segmentation_changes_nothing() {
        // every word one piece: projection is the identity apart from ordering/duplicates
        let c = corpus("▁a ▁b ▁c ||| ▁x ▁y\n");
        let got = project_text(&c, &Boundary::sentencepiece(), b"0-0 2-0 1-1\n", false).unwrap();
        assert_eq!(got, b"0-0 2-0 1-1\n");
    }

    #[test]
    fn boundary_parts_validate() {
        assert!(Boundary::from_parts("start", b"").is_err());
        assert!(Boundary::from_parts("start", b"a b").is_err());
        assert!(Boundary::from_parts("middle", b"x").is_err());
        assert_eq!(
            Boundary::from_parts("continue", b"@@").unwrap(),
            Boundary::Continue(b"@@".to_vec())
        );
    }
}
