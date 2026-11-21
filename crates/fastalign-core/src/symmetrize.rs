//! Symmetrisation of two directional alignments, identical to fast_align's `atools`
//! (`docs/SPEC-fast-align.md` §17): `intersect`, `union`, `grow-diag`, `grow-diag-final`,
//! `grow-diag-final-and`, and `invert`.
//!
//! Alignments are lists of `(i, j)` links, `i` the index in the first corpus side and `j` in the
//! second, sorted by `(i, j)`. The grow step is atools' own (candidate-driven, in `(i, j)`
//! order), which differs from Moses' `symal`.
//!
//! Deviation from atools: a last input line without a trailing newline is processed once, not
//! twice (atools' stream loop duplicates it; that is a bug, not behaviour worth keeping).

use std::borrow::Cow;
use std::fmt;
use std::io::Write;
use std::str::FromStr;

use rayon::prelude::*;

/// One alignment link `(first-side index, second-side index)`.
pub type Link = (u32, u32);

/// A symmetrisation method (atools' `-c COMMAND`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Intersect,
    Union,
    GrowDiag,
    GrowDiagFinal,
    GrowDiagFinalAnd,
    /// Swap the two sides of one alignment (needs no second file).
    Invert,
}

impl Method {
    pub const ALL: [Method; 6] = [
        Method::Intersect,
        Method::Union,
        Method::GrowDiag,
        Method::GrowDiagFinal,
        Method::GrowDiagFinalAnd,
        Method::Invert,
    ];

    /// The command name atools uses.
    pub fn name(self) -> &'static str {
        match self {
            Method::Intersect => "intersect",
            Method::Union => "union",
            Method::GrowDiag => "grow-diag",
            Method::GrowDiagFinal => "grow-diag-final",
            Method::GrowDiagFinalAnd => "grow-diag-final-and",
            Method::Invert => "invert",
        }
    }
}

impl FromStr for Method {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        Method::ALL
            .into_iter()
            .find(|m| m.name() == s)
            .ok_or_else(|| {
                let names: Vec<_> = Method::ALL.iter().map(|m| m.name()).collect();
                format!(
                    "unknown symmetrisation method '{s}' (one of: {})",
                    names.join(", ")
                )
            })
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Errors of reading or symmetrising alignment text.
#[derive(Debug, PartialEq, Eq)]
pub enum SymError {
    /// A line (1-based) is not a list of `i-j` links.
    BadAlignment(usize),
    /// The two files have different numbers of lines.
    Mismatched,
    /// The method needs a second alignment.
    NeedsTwo(Method),
}

impl fmt::Display for SymError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymError::BadAlignment(n) => {
                write!(f, "line {n}: bad alignment (expected `i-j` links)")
            }
            SymError::Mismatched => f.write_str("Mismatched number of lines!"),
            SymError::NeedsTwo(m) => write!(f, "{m} needs two alignments"),
        }
    }
}

impl std::error::Error for SymError {}

/// Parse one line of an alignment file. This is the single definition of the accepted layouts:
///
/// 1. links only: `0-0 1-2` (possibly empty);
/// 2. scored, as `train --scores` writes: `0-0 1-2 ||| -4.8351` (an empty link set is
///    ` ||| -4.8351`); the suffix must be a number (`nan` and `inf` included, any sign);
/// 3. atools-prefixed: `source words ||| target words ||| 0-0 1-2`, only the part after the last
///    ` ||| ` counts (a scored line of this layout works too).
///
/// A link text that is not a list of `i-j` links is `None`. As in atools, a carriage return ends
/// the line's content (so CRLF files read like LF files); the delimiter is searched in the whole
/// line. Returns the links sorted by `(i, j)`, duplicates removed.
pub fn parse_line(line: &[u8]) -> Option<Vec<Link>> {
    let (before, last) = split_last(line);
    let last = cut_cr(last);
    if let Some(links) = parse_links(last) {
        return Some(links);
    }
    // Not links: only exactly one score suffix may follow the links.
    if before.is_none() || !is_score(last) {
        return None;
    }
    let (_, links) = split_last(before?);
    parse_links(cut_cr(links))
}

/// Split at the last ` ||| `: the text before it (if there is one) and the text after it.
fn split_last(line: &[u8]) -> (Option<&[u8]>, &[u8]) {
    match line.windows(5).rposition(|w| w == b" ||| ") {
        Some(k) => (Some(&line[..k]), &line[k + 5..]),
        None => (None, line),
    }
}

/// The text up to the first carriage return.
fn cut_cr(s: &[u8]) -> &[u8] {
    s.iter().position(|&b| b == b'\r').map_or(s, |k| &s[..k])
}

/// A score as `train --scores` prints it: a decimal number, `inf` or `nan`, with an optional sign.
fn is_score(seg: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(seg) else {
        return false;
    };
    let t = text.trim_matches(' ');
    let unsigned = t.strip_prefix(['-', '+']).unwrap_or(t);
    !unsigned.is_empty()
        && !unsigned.starts_with(['-', '+'])
        && (t.parse::<f64>().is_ok() || unsigned.eq_ignore_ascii_case("nan"))
}

fn parse_links(text: &[u8]) -> Option<Vec<Link>> {
    let mut links = Vec::new();
    for tok in text.split(|&b| b == b' ').filter(|t| !t.is_empty()) {
        let dash = tok.iter().position(|&b| b == b'-')?;
        let (x, y) = (&tok[..dash], &tok[dash + 1..]);
        links.push((parse_u32(x)?, parse_u32(y)?));
    }
    links.sort_unstable();
    links.dedup();
    Some(links)
}

fn parse_u32(s: &[u8]) -> Option<u32> {
    if s.is_empty() || !s.iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(s).ok()?.parse().ok()
}

/// Write links as `i-j` separated by spaces (no newline).
pub fn format_links(links: &[Link], out: &mut Vec<u8>) {
    let mut first = true;
    for &(i, j) in links {
        if !first {
            out.push(b' ');
        }
        first = false;
        let _ = write!(out, "{i}-{j}");
    }
}

/// `links` sorted by `(i, j)` without duplicates (borrowed when it already is).
fn normalised(links: &[Link]) -> Cow<'_, [Link]> {
    if links.windows(2).all(|w| w[0] < w[1]) {
        Cow::Borrowed(links)
    } else {
        let mut v = links.to_vec();
        v.sort_unstable();
        v.dedup();
        Cow::Owned(v)
    }
}

/// In the first alignment.
const IN_A: u8 = 1;
/// In the second alignment.
const IN_B: u8 = 2;

/// The sorted union of two sorted, duplicate-free alignments, with the [`IN_A`]/[`IN_B`]
/// membership of each link.
fn merge(a: &[Link], b: &[Link]) -> (Vec<Link>, Vec<u8>) {
    let mut all = Vec::with_capacity(a.len() + b.len());
    let mut flags = Vec::with_capacity(a.len() + b.len());
    let (mut x, mut y) = (0, 0);
    while x < a.len() || y < b.len() {
        let (l, f) = match (a.get(x), b.get(y)) {
            (Some(&p), Some(&q)) if p == q => {
                x += 1;
                y += 1;
                (p, IN_A | IN_B)
            }
            (Some(&p), Some(&q)) if p < q => {
                x += 1;
                (p, IN_A)
            }
            (Some(&p), None) => {
                x += 1;
                (p, IN_A)
            }
            (_, Some(&q)) => {
                y += 1;
                (q, IN_B)
            }
            (None, None) => unreachable!(),
        };
        all.push(l);
        flags.push(f);
    }
    (all, flags)
}

/// Grids up to this many cells (one byte each) are always dense; see [`dense_budget`].
const DENSE_MIN_CELLS: usize = 1 << 16;
/// Never a dense grid above this many cells.
const DENSE_MAX_CELLS: usize = 1 << 24;

/// Cells a dense membership grid may have for a union of `n` links: a grid costs one byte
/// per cell to clear, the sparse lookup a binary search over the union per neighbour row.
fn dense_budget(n: usize) -> usize {
    n.saturating_mul(64).clamp(DENSE_MIN_CELLS, DENSE_MAX_CELLS)
}

/// The links accepted so far, a subset of the union `all`. Membership of any cell is either a
/// dense grid over the union's bounding box or a binary search in `all` (sparse alignments
/// with large indices: memory and time scale with the number of links, not the box).
struct Accepted<'a> {
    all: &'a [Link],
    acc: Vec<bool>,
    /// Rank of each union member's first-side / second-side index among the distinct ones.
    rank_i: Vec<u32>,
    rank_j: Vec<u32>,
    /// Whether a distinct first-side / second-side index has an accepted link.
    aligned_i: Vec<bool>,
    aligned_j: Vec<bool>,
    /// Dense grid over `[i0, i0 + w) × [j0, j0 + h)`, if the box is within the budget.
    grid: Option<Vec<bool>>,
    i0: i64,
    j0: i64,
    w: i64,
    h: i64,
}

impl<'a> Accepted<'a> {
    fn new(all: &'a [Link], budget: usize) -> Self {
        let (mut i0, mut i1, mut j0, mut j1) = (u32::MAX, 0, u32::MAX, 0);
        for &(i, j) in all {
            i0 = i0.min(i);
            i1 = i1.max(i);
            j0 = j0.min(j);
            j1 = j1.max(j);
        }
        let (w, h) = if all.is_empty() {
            (0, 0)
        } else {
            (u64::from(i1 - i0) + 1, u64::from(j1 - j0) + 1)
        };
        let grid = w
            .checked_mul(h)
            .filter(|&c| c <= budget as u64)
            .map(|c| vec![false; c as usize]);
        // ranks of i follow the sorted union; ranks of j need the sorted distinct j's
        let mut rank_i = Vec::with_capacity(all.len());
        let mut n_i = 0;
        for (p, &(i, _)) in all.iter().enumerate() {
            if p > 0 && all[p - 1].0 != i {
                n_i += 1;
            }
            rank_i.push(n_i);
        }
        let mut js: Vec<u32> = all.iter().map(|l| l.1).collect();
        js.sort_unstable();
        js.dedup();
        let rank_j = all
            .iter()
            .map(|l| js.binary_search(&l.1).expect("in the list") as u32)
            .collect();
        Accepted {
            all,
            acc: vec![false; all.len()],
            rank_i,
            rank_j,
            aligned_i: vec![false; n_i as usize + usize::from(!all.is_empty())],
            aligned_j: vec![false; js.len()],
            grid,
            i0: i64::from(i0),
            j0: i64::from(j0),
            w: w as i64,
            h: h as i64,
        }
    }

    /// Accept the union member at position `p`.
    fn set(&mut self, p: usize) {
        self.acc[p] = true;
        self.aligned_i[self.rank_i[p] as usize] = true;
        self.aligned_j[self.rank_j[p] as usize] = true;
        if let Some(g) = &mut self.grid {
            let (i, j) = self.all[p];
            let k = (i64::from(i) - self.i0) * self.h + (i64::from(j) - self.j0);
            g[k as usize] = true;
        }
    }

    /// Whether the row and the column of the union member at position `p` have an accepted
    /// link.
    fn aligned(&self, p: usize) -> (bool, bool) {
        (
            self.aligned_i[self.rank_i[p] as usize],
            self.aligned_j[self.rank_j[p] as usize],
        )
    }

    /// Whether any of the 8 neighbours of the union member at position `p` is accepted.
    fn near(&self, p: usize) -> bool {
        let (i, j) = self.all[p];
        let (i, j) = (i64::from(i), i64::from(j));
        if let Some(g) = &self.grid {
            let at = |di: i64, dj: i64| {
                let (x, y) = (i + di - self.i0, j + dj - self.j0);
                x >= 0 && y >= 0 && x < self.w && y < self.h && g[(x * self.h + y) as usize]
            };
            return at(-1, 0)
                || at(0, -1)
                || at(1, 0)
                || at(0, 1)
                || at(-1, -1)
                || at(-1, 1)
                || at(1, -1)
                || at(1, 1);
        }
        // Same row: the union's neighbours of `p` in sorted order.
        let same = |q: usize| self.all[q].0 as i64 == i && (self.all[q].1 as i64 - j).abs() == 1;
        if (p > 0 && same(p - 1) && self.acc[p - 1])
            || (p + 1 < self.all.len() && same(p + 1) && self.acc[p + 1])
        {
            return true;
        }
        // Rows i - 1 and i + 1: the members in columns j - 1 ..= j + 1 are consecutive.
        for r in [i - 1, i + 1] {
            if r < 0 || r > i64::from(u32::MAX) {
                continue;
            }
            let lo = (r as u32, (j - 1).max(0) as u32);
            let mut q = self.all.partition_point(|&l| l < lo);
            while q < self.all.len() {
                let (x, y) = self.all[q];
                if i64::from(x) != r || i64::from(y) > j + 1 {
                    break;
                }
                if self.acc[q] {
                    return true;
                }
                q += 1;
            }
        }
        false
    }
}

/// Symmetrise two alignments of one line. `a` is the forward and `b` the reverse alignment, both
/// in first-side/second-side orientation; the result is sorted by `(i, j)`. Inputs that are not
/// sorted or have duplicates are normalised first (as [`parse_line`] returns them).
///
/// The grow methods follow atools exactly: the intersection, then passes over the remaining
/// union links in `(i, j)` order, each accepting a link whose row or column is still unaligned
/// and that has an accepted neighbour (accepted links count at once, within the pass), until a
/// pass accepts nothing; then (`-final`) the links of `a` and then of `b` in `(i, j)` order.
/// Time and memory scale with the number of links: a dense grid is used only for a bounding
/// box within a budget proportional to the links, otherwise lookups search the sorted union.
pub fn symmetrize(a: &[Link], b: &[Link], method: Method) -> Vec<Link> {
    symmetrize_with(a, b, method, dense_budget(a.len() + b.len()))
}

/// [`symmetrize`] with a given dense-grid budget in cells (0 = always sparse); the result
/// never depends on it.
#[doc(hidden)]
pub fn symmetrize_with(a: &[Link], b: &[Link], method: Method, budget: usize) -> Vec<Link> {
    let a = normalised(a);
    if method == Method::Invert {
        return invert(&a);
    }
    let b = normalised(b);
    let (all, flags) = merge(&a, &b);
    match method {
        Method::Union => return all,
        Method::Intersect => {
            return all
                .iter()
                .zip(&flags)
                .filter(|(_, &f)| f == IN_A | IN_B)
                .map(|(&l, _)| l)
                .collect();
        }
        _ => {}
    }
    let mut r = Accepted::new(&all, budget);
    for (p, &f) in flags.iter().enumerate() {
        if f == IN_A | IN_B {
            r.set(p);
        }
    }
    // Candidates: the union without the intersection, in (i, j) order.
    let mut pending: Vec<usize> = (0..all.len())
        .filter(|&p| flags[p] != IN_A | IN_B)
        .collect();
    loop {
        let mut added = false;
        let mut keep = Vec::with_capacity(pending.len());
        for &p in &pending {
            let (ai, aj) = r.aligned(p);
            if (!ai || !aj) && r.near(p) {
                r.set(p);
                added = true;
            } else {
                keep.push(p);
            }
        }
        pending = keep;
        if !added {
            break;
        }
    }
    if method != Method::GrowDiag {
        let and = method == Method::GrowDiagFinalAnd;
        for side in [IN_A, IN_B] {
            for (p, &f) in flags.iter().enumerate() {
                if f & side == 0 || r.acc[p] {
                    continue;
                }
                let (ai, aj) = r.aligned(p);
                if if and { !ai && !aj } else { !ai || !aj } {
                    r.set(p);
                }
            }
        }
    }
    all.iter()
        .zip(&r.acc)
        .filter(|(_, &x)| x)
        .map(|(&l, _)| l)
        .collect()
}

/// Swap the two sides of an alignment (atools `invert`); the result is sorted and, like every
/// alignment read from text, has no duplicate links.
pub fn invert(a: &[Link]) -> Vec<Link> {
    let mut v: Vec<Link> = a.iter().map(|&(i, j)| (j, i)).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Split text into lines the way atools reads a file: `\n`-terminated, the last line optionally
/// not (no empty last line is invented).
fn lines(text: &[u8]) -> Vec<&[u8]> {
    let mut v: Vec<&[u8]> = text.split(|&b| b == b'\n').collect();
    if v.last().is_some_and(|l| l.is_empty()) {
        v.pop();
    }
    v
}

/// Symmetrise two Pharaoh-style texts line by line (`atools -c METHOD -i FWD -j INV`); with
/// [`Method::Invert`] only `fwd` is used. Output is one line per input line.
pub fn symmetrize_text(
    fwd: &[u8],
    inv: Option<&[u8]>,
    method: Method,
    out: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let fl = lines(fwd);
    let il = match (method, inv) {
        (Method::Invert, _) => None,
        (_, Some(t)) => Some(lines(t)),
        (m, None) => return Err(SymError::NeedsTwo(m).into()),
    };
    if il.as_ref().is_some_and(|l| l.len() != fl.len()) {
        return Err(SymError::Mismatched.into());
    }
    const CHUNK: usize = 8192;
    let window = 8 * rayon::current_num_threads();
    let idx: Vec<usize> = (0..fl.len()).step_by(CHUNK).collect();
    for win in idx.chunks(window) {
        let texts: Vec<Result<Vec<u8>, SymError>> = win
            .par_iter()
            .map(|&s| {
                let e = (s + CHUNK).min(fl.len());
                let mut buf = Vec::new();
                for k in s..e {
                    let a = parse_line(fl[k]).ok_or(SymError::BadAlignment(k + 1))?;
                    let r = match &il {
                        None => invert(&a),
                        Some(il) => {
                            let b = parse_line(il[k]).ok_or(SymError::BadAlignment(k + 1))?;
                            symmetrize(&a, &b, method)
                        }
                    };
                    format_links(&r, &mut buf);
                    buf.push(b'\n');
                }
                Ok(buf)
            })
            .collect();
        for t in texts {
            out.write_all(&t?)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Vec<Link> {
        parse_line(s.as_bytes()).unwrap()
    }

    fn sym(a: &str, b: &str, m: Method) -> String {
        let mut o = Vec::new();
        format_links(&symmetrize(&p(a), &p(b), m), &mut o);
        String::from_utf8(o).unwrap()
    }

    #[test]
    fn parse_variants() {
        assert_eq!(p(""), vec![]);
        assert_eq!(p("1-2 0-0"), vec![(0, 0), (1, 2)]);
        assert_eq!(p("a b ||| x ||| 0-1"), vec![(0, 1)]);
        assert!(parse_line(b"0-").is_none());
        assert!(parse_line(b"x-1").is_none());
        assert_eq!(parse_line(b"0-1 ||| -3.5"), Some(vec![(0, 1)]));
    }

    #[test]
    fn layouts() {
        let l = |s: &str| parse_line(s.as_bytes());
        // scored, including empty link sets and unusual numbers
        assert_eq!(l("0-0 1-1 ||| -4.8351"), Some(vec![(0, 0), (1, 1)]));
        assert_eq!(l(" ||| -2"), Some(vec![]));
        for score in [
            "nan", "-nan", "NaN", "inf", "-inf", "+1.5", "1e-05", "-0", "3",
        ] {
            assert_eq!(
                l(&format!("0-1 ||| {score}")),
                Some(vec![(0, 1)]),
                "{score}"
            );
            assert_eq!(l(&format!(" ||| {score}")), Some(vec![]), "{score}");
        }
        // prefixed, with and without a score
        assert_eq!(l("a b ||| x y ||| 1-0 0-1"), Some(vec![(0, 1), (1, 0)]));
        assert_eq!(l("a ||| x ||| 0-0 ||| -3.5"), Some(vec![(0, 0)]));
        assert_eq!(l("a ||| x ||| "), Some(vec![]));
        // not accepted: a score that is no number, two scores, a score with no links text,
        // a score alone
        assert!(l("0-1 ||| abc").is_none());
        assert!(l("0-1 ||| 1 ||| 2").is_none());
        assert!(l("0-1 ||| --3").is_none());
        assert!(l("-3.5").is_none());
        assert!(l("0-1 ||| ").is_some());
        // CR ends the content, as in atools
        assert_eq!(l("0-0 1-1\r"), Some(vec![(0, 0), (1, 1)]));
        assert_eq!(l("\r"), Some(vec![]));
        assert_eq!(l("0-0 ||| -3.5\r"), Some(vec![(0, 0)]));
    }

    #[test]
    fn line_endings() {
        let run = |a: &[u8], b: &[u8], m| {
            let mut o = Vec::new();
            symmetrize_text(a, Some(b), m, &mut o).unwrap();
            o
        };
        for m in Method::ALL {
            let lf = run(b"0-0 1-1\n\n1-0", b"0-0\n\n1-0\n", m);
            let crlf = run(b"0-0 1-1\r\n\r\n1-0", b"0-0\r\n\r\n1-0\r\n", m);
            assert_eq!(lf, crlf, "{m}");
        }
    }

    #[test]
    fn invert_removes_duplicates() {
        assert_eq!(invert(&[(0, 1), (0, 1)]), vec![(1, 0)]);
    }

    #[test]
    fn simple_methods() {
        assert_eq!(sym("0-0 1-1", "1-1 2-2", Method::Intersect), "1-1");
        assert_eq!(sym("0-0 1-1", "1-1 2-2", Method::Union), "0-0 1-1 2-2");
        assert_eq!(invert(&p("0-1 2-0")), vec![(0, 2), (1, 0)]);
    }

    #[test]
    fn grow_and_final() {
        // Intersection {1-1}. 0-0 grows (diagonal neighbour); then 0-2 is adjacent to 1-1 and
        // j=2 is still free; 2-2 likewise.
        let (a, b) = ("0-0 1-1 0-2", "1-1 2-2");
        assert_eq!(sym(a, b, Method::GrowDiag), "0-0 0-2 1-1 2-2");
        // 3-3 is not adjacent to anything; the final step adds it only if a side is free.
        assert_eq!(sym("0-0 1-1 3-3", "1-1", Method::GrowDiag), "0-0 1-1");
        assert_eq!(
            sym("0-0 1-1 3-3", "1-1", Method::GrowDiagFinalAnd),
            "0-0 1-1 3-3"
        );
        // 1-3: i=1 aligned, j=3 free -> "final" adds it, "final-and" does not.
        assert_eq!(
            sym("0-0 1-1 1-3", "1-1", Method::GrowDiagFinal),
            "0-0 1-1 1-3"
        );
        assert_eq!(
            sym("0-0 1-1 1-3", "1-1", Method::GrowDiagFinalAnd),
            "0-0 1-1"
        );
    }

    #[test]
    fn empty_inputs() {
        assert_eq!(sym("", "", Method::GrowDiagFinalAnd), "");
        assert_eq!(sym("0-0", "", Method::GrowDiagFinalAnd), "0-0");
    }

    #[test]
    fn text_roundtrip_and_errors() {
        let mut o = Vec::new();
        symmetrize_text(b"0-0 1-1\n\n", Some(b"1-1\n2-2\n"), Method::Union, &mut o).unwrap();
        assert_eq!(o, b"0-0 1-1\n2-2\n");
        assert!(symmetrize_text(b"0-0\n", Some(b"0-0\n0-0\n"), Method::Union, &mut o).is_err());
        assert!(symmetrize_text(b"zz\n", Some(b"0-0\n"), Method::Union, &mut o).is_err());
        let mut o = Vec::new();
        symmetrize_text(b"0-1 3-2", None, Method::Invert, &mut o).unwrap();
        assert_eq!(o, b"1-0 2-3\n");
    }

    /// The previous implementation (Steps 7–25, equal to atools on every golden run): a dense
    /// grid over `[0, max i] × [0, max j]`, kept as the reference for the sparse path.
    fn grid_reference(a: &[Link], b: &[Link], method: Method) -> Vec<Link> {
        let inter: Vec<Link> = a
            .iter()
            .copied()
            .filter(|l| b.binary_search(l).is_ok())
            .collect();
        let mut all: Vec<Link> = a.iter().chain(b).copied().collect();
        all.sort_unstable();
        all.dedup();
        match method {
            Method::Intersect => return inter,
            Method::Union => return all,
            Method::Invert => return invert(a),
            _ => {}
        }
        let w = all.iter().map(|l| l.0 as usize + 1).max().unwrap_or(1);
        let h = all.iter().map(|l| l.1 as usize + 1).max().unwrap_or(1);
        let mut g = vec![false; w * h];
        let get = |g: &[bool], i: i64, j: i64| {
            i >= 0
                && j >= 0
                && (i as usize) < w
                && (j as usize) < h
                && g[i as usize * h + j as usize]
        };
        let (mut ai, mut aj) = (vec![false; w], vec![false; h]);
        let mut set = |g: &mut [bool], i: u32, j: u32| {
            g[i as usize * h + j as usize] = true;
            ai[i as usize] = true;
            aj[j as usize] = true;
            (ai.clone(), aj.clone())
        };
        let (mut fi, mut fj) = (vec![false; w], vec![false; h]);
        for &(i, j) in &inter {
            (fi, fj) = set(&mut g, i, j);
        }
        let mut pending: Vec<Link> = all
            .iter()
            .copied()
            .filter(|&(i, j)| !get(&g, i64::from(i), i64::from(j)))
            .collect();
        loop {
            let mut added = false;
            let mut keep = Vec::new();
            for &(i, j) in &pending {
                let (ii, jj) = (i64::from(i), i64::from(j));
                let near = (-1..=1)
                    .flat_map(|di| (-1..=1).map(move |dj| (di, dj)))
                    .filter(|&d| d != (0, 0))
                    .any(|(di, dj)| get(&g, ii + di, jj + dj));
                if (!fi[i as usize] || !fj[j as usize]) && near {
                    (fi, fj) = set(&mut g, i, j);
                    added = true;
                } else {
                    keep.push((i, j));
                }
            }
            pending = keep;
            if !added {
                break;
            }
        }
        if method != Method::GrowDiag {
            for x in [a, b] {
                for &(i, j) in x {
                    if get(&g, i64::from(i), i64::from(j)) {
                        continue;
                    }
                    let (pi, pj) = (fi[i as usize], fj[j as usize]);
                    let ok = if method == Method::GrowDiagFinalAnd {
                        !pi && !pj
                    } else {
                        !pi || !pj
                    };
                    if ok {
                        (fi, fj) = set(&mut g, i, j);
                    }
                }
            }
        }
        let mut out = Vec::new();
        for i in 0..w {
            for j in 0..h {
                if g[i * h + j] {
                    out.push((i as u32, j as u32));
                }
            }
        }
        out
    }

    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: u32) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % u64::from(n)) as u32
        }
        fn links(&mut self, n: usize, w: u32, h: u32) -> Vec<Link> {
            (0..n).map(|_| (self.below(w), self.below(h))).collect()
        }
    }

    #[test]
    fn sparse_and_dense_equal_the_grid_reference() {
        let mut rng = Rng(0x2026_1120);
        for case in 0..4000 {
            let (w, h) = (1 + rng.below(12), 1 + rng.below(12));
            let na = rng.below(w * h / 2 + 2) as usize;
            let nb = rng.below(w * h / 2 + 2) as usize;
            let raw_a = rng.links(na, w, h);
            let raw_b = rng.links(nb, w, h);
            let (mut a, mut b) = (raw_a.clone(), raw_b.clone());
            for v in [&mut a, &mut b] {
                v.sort_unstable();
                v.dedup();
            }
            for m in Method::ALL {
                let want = grid_reference(&a, &b, m);
                for budget in [0, 1, 7, usize::MAX] {
                    // unsorted, duplicated input is normalised first
                    let got = symmetrize_with(&raw_a, &raw_b, m, budget);
                    assert_eq!(got, want, "case {case} {m} budget {budget}: {a:?} / {b:?}");
                }
                assert_eq!(symmetrize(&raw_a, &raw_b, m), want);
            }
        }
    }

    #[test]
    fn one_common_link_far_away() {
        // review performance item 4: the old grid needed 400M cells and refused this line
        for (i, j) in [
            (20000, 20000),
            (u32::MAX, u32::MAX),
            (0, u32::MAX),
            (u32::MAX, 0),
        ] {
            for m in Method::ALL {
                let want = if m == Method::Invert {
                    vec![(j, i)]
                } else {
                    vec![(i, j)]
                };
                assert_eq!(symmetrize(&[(i, j)], &[(i, j)], m), want, "{m}");
            }
        }
    }

    #[test]
    fn large_sparse_alignments() {
        // Far-apart diagonal blocks: (0, 0) and from 4e9 on. a has the diagonal, b the diagonal
        // plus the cell right of each diagonal cell. Intersection = the diagonal; every extra b
        // cell (k, k+1) has its accepted neighbour (k, k) but column k+1 is already aligned
        // (by (k+1, k+1)) and row k too, so grow adds nothing; except the last block cell, whose
        // column k+1 has no diagonal cell: grow adds it.
        let base = 4_000_000_000u32;
        let mut a = vec![(0, 0)];
        let mut b = vec![(0, 0), (0, 1)];
        for k in 0..100 {
            a.push((base + k, base + k));
            b.push((base + k, base + k));
            b.push((base + k, base + k + 1));
        }
        let mut want = a.clone();
        want.insert(1, (0, 1));
        want.push((base + 99, base + 100));
        for m in [Method::GrowDiag, Method::GrowDiagFinal] {
            assert_eq!(symmetrize(&a, &b, m), want, "{m}");
        }
        // final-and adds only links whose row and column are both unaligned: none here
        assert_eq!(symmetrize(&a, &b, Method::GrowDiagFinalAnd), want);
        assert_eq!(symmetrize(&a, &b, Method::Intersect), a);
        // A chain grows in one pass because accepted links are visible within the pass:
        // intersection (base, base); candidates (base+k, base+k) for k = 1..1000 in order.
        let chain: Vec<Link> = (0..1000).map(|k| (base + k, base + k)).collect();
        assert_eq!(
            symmetrize(&chain, &chain[..1], Method::GrowDiag),
            chain,
            "chain"
        );
        // Reverse chain: candidates before the seed in (i, j) order grow one per pass.
        let mut rev: Vec<Link> = (0..300).map(|k| (base + k, base + 299 - k)).collect();
        rev.sort_unstable();
        let seed = [(base + 299, base)];
        assert_eq!(
            symmetrize(&rev, &seed, Method::GrowDiag),
            rev,
            "anti-diagonal"
        );
    }
}
