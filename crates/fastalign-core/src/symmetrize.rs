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
    /// A line's links span a grid too large to handle.
    TooLarge(usize),
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
            SymError::TooLarge(n) => write!(f, "line {n}: alignment indices are too large"),
            SymError::NeedsTwo(m) => write!(f, "{m} needs two alignments"),
        }
    }
}

impl std::error::Error for SymError {}

/// Parse one line as atools does: if it contains ` ||| ` only the part after the last one
/// counts; links are `x-y` separated by spaces. Returns the links sorted by `(x, y)`,
/// duplicates removed.
pub fn parse_line(line: &[u8]) -> Option<Vec<Link>> {
    let mut start = 0;
    let mut k = 0;
    while k + 5 <= line.len() {
        if &line[k..k + 5] == b" ||| " {
            start = k + 5;
            k += 5;
        } else {
            k += 1;
        }
    }
    let mut links = Vec::new();
    for tok in line[start..]
        .split(|&b| b == b' ')
        .filter(|t| !t.is_empty())
    {
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

/// Dense membership grid over `[0, w) × [0, h)`.
struct Grid {
    w: usize,
    h: usize,
    cells: Vec<bool>,
}

impl Grid {
    fn new(w: usize, h: usize) -> Self {
        Grid {
            w,
            h,
            cells: vec![false; w * h],
        }
    }

    fn get(&self, i: i64, j: i64) -> bool {
        i >= 0
            && j >= 0
            && (i as usize) < self.w
            && (j as usize) < self.h
            && self.cells[i as usize * self.h + j as usize]
    }

    fn set(&mut self, i: u32, j: u32) {
        self.cells[i as usize * self.h + j as usize] = true;
    }
}

/// Largest grid (cells) the grow methods will allocate per line.
const MAX_GRID: usize = 1 << 28;

fn union_sorted(a: &[Link], b: &[Link]) -> Vec<Link> {
    let mut v = Vec::with_capacity(a.len() + b.len());
    v.extend_from_slice(a);
    v.extend_from_slice(b);
    v.sort_unstable();
    v.dedup();
    v
}

/// Symmetrise two alignments of one line. `a` is the forward and `b` the reverse alignment, both
/// in first-side/second-side orientation; the result is sorted by `(i, j)`. Returns `None` if the
/// grid would be larger than [`MAX_GRID`] cells.
pub fn symmetrize(a: &[Link], b: &[Link], method: Method) -> Option<Vec<Link>> {
    match method {
        Method::Invert => return Some(invert(a)),
        Method::Union => return Some(union_sorted(a, b)),
        Method::Intersect => {
            return Some(
                a.iter()
                    .copied()
                    .filter(|l| b.binary_search(l).is_ok())
                    .collect(),
            );
        }
        _ => {}
    }
    let inter: Vec<Link> = a
        .iter()
        .copied()
        .filter(|l| b.binary_search(l).is_ok())
        .collect();
    let all = union_sorted(a, b);
    let w = all.iter().map(|l| l.0 as usize + 1).max().unwrap_or(1);
    let h = all.iter().map(|l| l.1 as usize + 1).max().unwrap_or(1);
    if w.checked_mul(h).is_none_or(|n| n > MAX_GRID) {
        return None;
    }
    let mut r = Grid::new(w, h);
    let mut aligned_i = vec![false; w];
    let mut aligned_j = vec![false; h];
    for &(i, j) in &inter {
        r.set(i, j);
        aligned_i[i as usize] = true;
        aligned_j[j as usize] = true;
    }
    // Candidates: the union without the intersection, ordered by (i, j).
    let mut pending: Vec<Link> = all
        .iter()
        .copied()
        .filter(|l| !r.get(i64::from(l.0), i64::from(l.1)))
        .collect();
    loop {
        let mut added = false;
        let mut keep = Vec::with_capacity(pending.len());
        for &(i, j) in &pending {
            let free = !aligned_i[i as usize] || !aligned_j[j as usize];
            let (ii, jj) = (i64::from(i), i64::from(j));
            let near = free
                && [
                    (-1, 0),
                    (0, -1),
                    (1, 0),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]
                .iter()
                .any(|&(di, dj)| r.get(ii + di, jj + dj));
            if near {
                r.set(i, j);
                aligned_i[i as usize] = true;
                aligned_j[j as usize] = true;
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
        let and = method == Method::GrowDiagFinalAnd;
        for x in [a, b] {
            for &(i, j) in x {
                if r.get(i64::from(i), i64::from(j)) {
                    continue;
                }
                let (ai, aj) = (aligned_i[i as usize], aligned_j[j as usize]);
                let ok = if and { !ai && !aj } else { !ai || !aj };
                if ok {
                    r.set(i, j);
                    aligned_i[i as usize] = true;
                    aligned_j[j as usize] = true;
                }
            }
        }
    }
    let mut out = Vec::new();
    for i in 0..w {
        for j in 0..h {
            if r.cells[i * h + j] {
                out.push((i as u32, j as u32));
            }
        }
    }
    Some(out)
}

/// Swap the two sides of an alignment (atools `invert`).
pub fn invert(a: &[Link]) -> Vec<Link> {
    let mut v: Vec<Link> = a.iter().map(|&(i, j)| (j, i)).collect();
    v.sort_unstable();
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
                            symmetrize(&a, &b, method).ok_or(SymError::TooLarge(k + 1))?
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
        format_links(&symmetrize(&p(a), &p(b), m).unwrap(), &mut o);
        String::from_utf8(o).unwrap()
    }

    #[test]
    fn parse_variants() {
        assert_eq!(p(""), vec![]);
        assert_eq!(p("1-2 0-0"), vec![(0, 0), (1, 2)]);
        assert_eq!(p("a b ||| x ||| 0-1"), vec![(0, 1)]);
        assert!(parse_line(b"0-").is_none());
        assert!(parse_line(b"x-1").is_none());
        assert!(parse_line(b"0-1 ||| -3.5").is_none());
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
}
