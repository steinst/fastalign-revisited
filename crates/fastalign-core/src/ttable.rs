//! Translation-table structure: CSR rows with the emulated libstdc++ iteration order, built by
//! fast_align's initial pass (`docs/SPEC-fast-align.md` §5, §6, §14).
//!
//! Row `e` (a source word id, row 1 = NULL) holds the target ids `f` that co-occur with `e`,
//! sorted ascending, so that a cell can be found by binary search. Next to it, `order` holds the
//! row's hash iteration order as a permutation of the row's cells. That order decides the
//! summation order of the M-step row totals and the line order of the `-p` table.
//!
//! The values (probabilities and counts) are not stored here; the trainers keep them in arrays
//! parallel to [`TTable::cols`].

use std::ops::Range;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::cancel::Cancel;
use crate::corpus::{part_ranges, Corpus, Side, MAX_PART_ENTRIES, NULL_ID};
use crate::stlhash::{hash_len_pair, hash_u32, StlOrder};

/// Options of the initial pass that change the table.
#[derive(Clone, Copy, Debug)]
pub struct TableOptions {
    /// fast_align's `-r`: condition on the second side and predict the first.
    pub reverse: bool,
    /// `false` for fast_align's `-N`.
    pub use_null: bool,
}
// fast_align's `-b` does not change the table: the insert buffer only delays inserts, and the
// only row that also receives direct inserts (NULL) gets every target word directly before any
// buffered `<eps>` insert of the same line could add it (spec §6).

impl Default for TableOptions {
    fn default() -> Self {
        TableOptions {
            reverse: false,
            use_null: true,
        }
    }
}

/// Rows up to this many cells store their hash order as `u16` offsets (PERF-TODO 7).
const SHORT_ROW: usize = 1 << 16;

/// CSR translation-table structure for one direction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TTable {
    row_ptr: Vec<usize>,
    cols: Vec<u32>,
    /// Hash order of every row of at most [`SHORT_ROW`] cells, parallel to `cols` (0 in the
    /// cells of longer rows).
    order: Vec<u16>,
    /// `(row, start in long_order)` of the longer rows, ascending by row.
    long_rows: Vec<(u32, usize)>,
    long_order: Vec<u32>,
    buckets: Vec<u64>,
}

/// Hash iteration order of one row: offsets within the row, stored narrow where they fit.
#[derive(Clone, Copy, Debug)]
pub enum RowOrder<'a> {
    Short(&'a [u16]),
    Long(&'a [u32]),
}

impl<'a> RowOrder<'a> {
    /// The offsets in order.
    pub fn iter(self) -> impl Iterator<Item = usize> + 'a {
        let (s, l): (&[u16], &[u32]) = match self {
            RowOrder::Short(s) => (s, &[]),
            RowOrder::Long(l) => (&[], l),
        };
        s.iter()
            .map(|&k| usize::from(k))
            .chain(l.iter().map(|&k| k as usize))
    }

    pub fn len(self) -> usize {
        match self {
            RowOrder::Short(s) => s.len(),
            RowOrder::Long(l) => l.len(),
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Split a flat per-row order (offsets within each row, parallel to the cells) into the
/// narrow array and the long rows' array.
fn pack_order<'s>(
    row_ptr: &[usize],
    row_order: impl Fn(usize) -> &'s [u32] + Sync,
    order: &mut [u16],
) -> (Vec<(u32, usize)>, Vec<u32>) {
    let mut long_rows = Vec::new();
    let mut long_order = Vec::new();
    for e in 0..row_ptr.len() - 1 {
        if row_ptr[e + 1] - row_ptr[e] > SHORT_ROW {
            long_rows.push((e as u32, long_order.len()));
            long_order.extend(row_order(e));
        }
    }
    let mut rest = &mut order[..];
    let mut parts = Vec::with_capacity(row_ptr.len() - 1);
    for w in row_ptr.windows(2) {
        let (h, t) = rest.split_at_mut(w[1] - w[0]);
        parts.push(h);
        rest = t;
    }
    parts
        .into_par_iter()
        .enumerate()
        .with_max_len(256)
        .for_each(|(e, dst)| {
            if dst.len() <= SHORT_ROW {
                for (d, &k) in dst.iter_mut().zip(row_order(e)) {
                    *d = k as u16;
                }
            }
        });
    (long_rows, long_order)
}

impl TTable {
    /// Number of rows (fast_align's `counts.size()`, see [`InitialPass`]).
    pub fn n_rows(&self) -> usize {
        self.row_ptr.len() - 1
    }

    /// Number of cells (co-occurring `(e, f)` pairs).
    pub fn n_cells(&self) -> usize {
        self.cols.len()
    }

    /// Cell range of row `e` (empty for rows beyond the table).
    #[inline]
    pub fn row(&self, e: u32) -> Range<usize> {
        let e = e as usize;
        if e + 1 < self.row_ptr.len() {
            self.row_ptr[e]..self.row_ptr[e + 1]
        } else {
            0..0
        }
    }

    /// Target ids of all cells; within a row ascending.
    pub fn cols(&self) -> &[u32] {
        &self.cols
    }

    /// Row offsets into [`TTable::cols`] (length `n_rows + 1`).
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Hash iteration order of row `e`: offsets *within the row* (add `row(e).start`).
    #[inline]
    pub fn row_order(&self, e: u32) -> RowOrder<'_> {
        let r = self.row(e);
        if r.len() <= SHORT_ROW {
            return RowOrder::Short(&self.order[r]);
        }
        let k = self
            .long_rows
            .binary_search_by_key(&e, |x| x.0)
            .expect("every long row has a long order");
        let s = self.long_rows[k].1;
        RowOrder::Long(&self.long_order[s..s + r.len()])
    }

    /// Cells of row `e` in hash iteration order, as global cell indices.
    pub fn iter_hash_order(&self, e: u32) -> impl Iterator<Item = usize> + '_ {
        let start = self.row(e).start;
        self.row_order(e).iter().map(move |k| start + k)
    }

    /// Bucket count of row `e`'s `unordered_map` (for validation against fast_align).
    pub fn bucket_count(&self, e: u32) -> u64 {
        self.buckets.get(e as usize).copied().unwrap_or(1)
    }

    /// Global cell index of `(e, f)`.
    #[inline]
    pub fn find(&self, e: u32, f: u32) -> Option<usize> {
        let r = self.row(e);
        let row = &self.cols[r.clone()];
        row.binary_search(&f).ok().map(|k| r.start + k)
    }

    /// Flat hash-order permutation (offsets within each row), parallel to [`TTable::cols`].
    pub fn order(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.n_rows() as u32).flat_map(|e| self.row_order(e).iter().map(|k| k as u32))
    }

    /// Bucket counts of all rows' `unordered_map`s.
    pub fn buckets(&self) -> &[u64] {
        &self.buckets
    }

    /// Rebuild a table from saved parts, checking every invariant the lookups rely on: row
    /// offsets non-decreasing and ending at the cell count, target ids strictly ascending in a
    /// row and below `vocab_len`, `order` a permutation within each row, one bucket count per
    /// row.
    pub fn from_parts(
        row_ptr: Vec<usize>,
        cols: Vec<u32>,
        order: Vec<u32>,
        buckets: Vec<u64>,
        vocab_len: usize,
    ) -> Result<Self, String> {
        if row_ptr.first() != Some(&0) || row_ptr.last() != Some(&cols.len()) {
            return Err("bad row offsets".into());
        }
        if order.len() != cols.len() || buckets.len() + 1 != row_ptr.len() {
            return Err("inconsistent table sizes".into());
        }
        if row_ptr.windows(2).any(|w| w[0] > w[1]) {
            return Err("row offsets decrease".into());
        }
        let mut seen = Vec::new();
        for w in row_ptr.windows(2) {
            let row = &cols[w[0]..w[1]];
            if row.windows(2).any(|p| p[0] >= p[1]) {
                return Err("table row is not strictly ascending".into());
            }
            if row.last().is_some_and(|&f| f as usize >= vocab_len) {
                return Err("table refers to a word outside the vocabulary".into());
            }
            seen.clear();
            seen.resize(row.len(), false);
            for &k in &order[w[0]..w[1]] {
                match seen.get_mut(k as usize) {
                    Some(s) if !*s => *s = true,
                    _ => return Err("table order is not a permutation".into()),
                }
            }
        }
        let mut narrow = vec![0u16; cols.len()];
        let (long_rows, long_order) = pack_order(
            &row_ptr,
            |e| &order[row_ptr[e]..row_ptr[e + 1]],
            &mut narrow,
        );
        drop(order);
        Ok(TTable {
            row_ptr,
            cols,
            order: narrow,
            long_rows,
            long_order,
            buckets,
        })
    }

    /// Heap bytes of the structure (without value arrays).
    pub fn memory_bytes(&self) -> usize {
        self.row_ptr.capacity() * std::mem::size_of::<usize>()
            + self.cols.capacity() * 4
            + self.order.capacity() * 2
            + self.long_rows.capacity() * 16
            + self.long_order.capacity() * 4
            + self.buckets.capacity() * 8
    }
}

/// One entry of fast_align's `size_counts`: target length `m`, source length `n` (as `i16`,
/// truncated like fast_align), number of lines with these lengths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeCount {
    pub m: i16,
    pub n: i16,
    pub count: u32,
}

/// Everything fast_align's initial pass produces for one direction.
#[derive(Clone, Debug)]
pub struct InitialPass {
    pub table: TTable,
    /// In libstdc++ iteration order (the summation order of the tension gradient).
    pub size_counts: Vec<SizeCount>,
    /// Sum over lines of `m / n` (f64, line order).
    pub tot_len_ratio: f64,
    /// Number of target tokens, as f64.
    pub n_target_tokens: f64,
    /// `tot_len_ratio / lines`.
    pub mean_srclen_multiplier: f64,
    pub n_lines: usize,
    pub options: TableOptions,
}

/// Lines (or target tokens of the NULL row) between two cancellation checks of the setup.
pub(crate) const CANCEL_STRIDE: usize = 1 << 16;

/// Run fast_align's initial pass over `corpus` for one direction.
pub fn initial_pass(corpus: &Corpus, opts: TableOptions) -> InitialPass {
    initial_pass_with(corpus, opts, &Cancel::default())
}

/// [`initial_pass`], checking `cancel` per row, per stretch of lines and per stretch of the
/// NULL row (see [`crate::cancel`]).
pub fn initial_pass_with(corpus: &Corpus, opts: TableOptions, cancel: &Cancel) -> InitialPass {
    cancel.check();
    let (src, trg) = corpus.sides(opts.reverse);
    let (table, stats) = rayon::join(
        || build_table(corpus.vocab.len(), src, trg, opts, cancel),
        || length_statistics(src, trg, cancel),
    );
    let (size_counts, tot_len_ratio, n_target_tokens) = stats;
    let n_lines = corpus.len();
    InitialPass {
        table,
        size_counts,
        tot_len_ratio,
        n_target_tokens,
        mean_srclen_multiplier: tot_len_ratio / n_lines as f64,
        n_lines,
        options: opts,
    }
}

/// `size_counts` in iteration order, `tot_len_ratio`, `n_target_tokens` (spec §6, §14.6).
fn length_statistics(src: &Side, trg: &Side, cancel: &Cancel) -> (Vec<SizeCount>, f64, f64) {
    let n_lines = src.offsets().len() - 1;
    let mut tot = 0.0f64;
    let mut ntt = 0.0f64;
    let mut index: FxHashMap<(i16, i16), u32> = FxHashMap::default();
    let mut keys: Vec<(i16, i16)> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut order = StlOrder::new();
    for k in 0..n_lines {
        if k % CANCEL_STRIDE == 0 {
            cancel.check();
        }
        let m = trg.line_len(k);
        let n = src.line_len(k);
        tot += m as f64 / n as f64;
        ntt += m as f64;
        let key = (m as i16, n as i16);
        let slot = *index.entry(key).or_insert_with(|| {
            keys.push(key);
            counts.push(0);
            order.insert_new(hash_len_pair(key.0, key.1))
        });
        counts[slot as usize] = counts[slot as usize].wrapping_add(1);
    }
    let size_counts = order
        .order()
        .map(|i| SizeCount {
            m: keys[i as usize].0,
            n: keys[i as usize].1,
            count: counts[i as usize],
        })
        .collect();
    (size_counts, tot, ntt)
}

/// Source-word postings: for each part of the corpus, a local CSR from source id to the lines
/// (in order, one entry per occurrence) where it occurs.
struct Postings {
    parts: Vec<(Vec<u32>, Vec<u32>)>,
}

impl Postings {
    fn build(src: &Side, n_vocab: usize, cancel: &Cancel) -> Self {
        Self::build_with(src, n_vocab, MAX_PART_ENTRIES, cancel)
    }

    /// `max_entries` bounds every part's occurrences (its `u32` offsets); tests lower it.
    fn build_with(src: &Side, n_vocab: usize, max_entries: usize, cancel: &Cancel) -> Self {
        let n_lines = src.offsets().len() - 1;
        let n_parts = rayon::current_num_threads().clamp(1, n_lines.max(1));
        let parts = part_ranges(src, n_parts, max_entries)
            .into_par_iter()
            .map(|lines| {
                let mut start = vec![0u32; n_vocab + 1];
                for k in lines.clone() {
                    if k % CANCEL_STRIDE == 0 {
                        cancel.check();
                    }
                    for &e in src.line(k) {
                        start[e as usize + 1] += 1;
                    }
                }
                for e in 0..n_vocab {
                    start[e + 1] += start[e];
                }
                let mut pos = start.clone();
                let mut out = vec![0u32; start[n_vocab] as usize];
                for k in lines {
                    for &e in src.line(k) {
                        out[pos[e as usize] as usize] = k as u32;
                        pos[e as usize] += 1;
                    }
                }
                (start, out)
            })
            .collect();
        Postings { parts }
    }

    /// Lines where `e` occurs, in corpus order (repeated if `e` occurs several times).
    fn lines(&self, e: u32) -> impl Iterator<Item = u32> + '_ {
        let e = e as usize;
        self.parts.iter().flat_map(move |(start, out)| {
            out[start[e] as usize..start[e + 1] as usize]
                .iter()
                .copied()
        })
    }
}

struct Scratch {
    seen: Vec<u32>,
    seq: Vec<u32>,
    stl: StlOrder,
    idx: Vec<u32>,
    rank: Vec<u32>,
}

struct RowOut {
    cols: Vec<u32>,
    order: Vec<u32>,
    buckets: u64,
}

fn build_table(
    n_vocab: usize,
    src: &Side,
    trg: &Side,
    opts: TableOptions,
    cancel: &Cancel,
) -> TTable {
    let postings = Postings::build(src, n_vocab, cancel);
    let max_src = src.tokens().par_iter().copied().max();
    // fast_align's counts.size(): the insert buffer covers every source id (SetMaxE), and a
    // direct NULL insert makes row 1 exist. (An input without any source token would make
    // fast_align resize the table to 0 rows; we keep the NULL row instead.)
    // The NULL row also exists (empty) when there is no target token but NULL is in use, so
    // that lookups of it never index past the table.
    let n_rows = max_src.map_or(0, |m| m as usize + 1).max(if opts.use_null {
        NULL_ID as usize + 1
    } else {
        0
    });

    let rows: Vec<RowOut> = (0..n_rows as u32)
        .into_par_iter()
        // The busiest rows have the smallest ids (ids are first-occurrence order), so they all
        // fall into the first few of rayon's adaptive pieces and the loop ran on ~2 threads.
        // Small pieces spread them (Step 9b: 1.67 -> 0.30 s at 20 threads on en-fr large).
        .with_max_len(16)
        // Once a row has unwound (cancellation), skip the remaining rows instead of starting
        // and unwinding every one of the thousands of small tasks in turn.
        .panic_fuse()
        .map_init(
            || Scratch {
                seen: vec![0; n_vocab],
                seq: Vec::new(),
                stl: StlOrder::new(),
                idx: Vec::new(),
                rank: Vec::new(),
            },
            |s, e| {
                cancel.check();
                s.seq.clear();
                let stamp = e + 1;
                if e == NULL_ID {
                    null_row_sequence(trg, &postings, opts, s, stamp, cancel);
                } else {
                    ordinary_row_sequence(trg, &postings, e, s, stamp);
                }
                finish_row(s)
            },
        )
        .collect();

    let mut row_ptr = Vec::with_capacity(n_rows + 1);
    row_ptr.push(0);
    for r in &rows {
        row_ptr.push(row_ptr.last().unwrap() + r.cols.len());
    }
    let total = *row_ptr.last().unwrap();
    let mut cols = vec![0u32; total];
    let mut order = vec![0u16; total];
    let buckets: Vec<u64> = rows.iter().map(|r| r.buckets).collect();
    // Parallel copy into disjoint row slices.
    let mut dst = Vec::with_capacity(rows.len());
    let mut crest = &mut cols[..];
    for r in &rows {
        let (ch, ct) = crest.split_at_mut(r.cols.len());
        dst.push(ch);
        crest = ct;
    }
    dst.into_par_iter()
        .zip(rows.par_iter())
        .with_min_len(256)
        .for_each(|(c, r)| c.copy_from_slice(&r.cols));
    let (long_rows, long_order) = pack_order(&row_ptr, |e| &rows[e].order, &mut order);
    TTable {
        row_ptr,
        cols,
        order,
        long_rows,
        long_order,
        buckets,
    }
}

#[inline]
fn push_new(s: &mut Scratch, f: u32, stamp: u32) {
    let seen = &mut s.seen[f as usize];
    if *seen != stamp {
        *seen = stamp;
        s.seq.push(f);
    }
}

/// Insertion sequence of the NULL row (spec §6). With the NULL word, every target token is
/// inserted directly in corpus order; buffered inserts of `<eps>` as a source word only repeat
/// target words of lines already inserted directly. Without it, row 1 is an ordinary row (only
/// non-empty if `<eps>` occurs as a source word).
fn null_row_sequence(
    trg: &Side,
    postings: &Postings,
    opts: TableOptions,
    s: &mut Scratch,
    stamp: u32,
    cancel: &Cancel,
) {
    if opts.use_null {
        for part in trg.tokens().chunks(CANCEL_STRIDE) {
            cancel.check();
            for &f in part {
                push_new(s, f, stamp);
            }
        }
    } else {
        ordinary_row_sequence(trg, postings, NULL_ID, s, stamp);
    }
}

fn ordinary_row_sequence(trg: &Side, postings: &Postings, e: u32, s: &mut Scratch, stamp: u32) {
    let mut last = u32::MAX;
    for line in postings.lines(e) {
        if line == last {
            continue;
        }
        last = line;
        for &f in trg.line(line as usize) {
            push_new(s, f, stamp);
        }
    }
}

/// Emulate the row's unordered_map, then express its order over the sorted keys.
fn finish_row(s: &mut Scratch) -> RowOut {
    let n = s.seq.len();
    s.stl.clear();
    for &f in &s.seq {
        s.stl.insert_new(hash_u32(f));
    }
    let buckets = s.stl.bucket_count();
    s.idx.clear();
    s.idx.extend(0..n as u32);
    let seq = &s.seq;
    s.idx.sort_unstable_by_key(|&i| seq[i as usize]);
    s.rank.clear();
    s.rank.resize(n, 0);
    for (pos, &i) in s.idx.iter().enumerate() {
        s.rank[i as usize] = pos as u32;
    }
    let cols = s.idx.iter().map(|&i| seq[i as usize]).collect();
    let order = s.stl.order().map(|node| s.rank[node as usize]).collect();
    RowOut {
        cols,
        order,
        buckets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::LoadOptions;
    use crate::stlhash::u32_order;

    fn corpus(text: &[u8]) -> Corpus {
        Corpus::from_joined_bytes(text, &LoadOptions::default()).unwrap()
    }

    /// Rows longer than 65,536 cells keep their order in `u32`; every row's order and the
    /// flat order survive the narrow storage unchanged.
    #[test]
    fn narrow_and_long_row_order() {
        let long = SHORT_ROW + 5;
        let lens = [3usize, 0, long, SHORT_ROW, 1];
        let mut row_ptr = vec![0usize];
        let mut cols = Vec::new();
        let mut order = Vec::new();
        for (r, &len) in lens.iter().enumerate() {
            row_ptr.push(row_ptr.last().unwrap() + len);
            cols.extend(0..len as u32);
            // A permutation that is not the identity: reversed, rotated by the row number.
            order.extend((0..len).map(|k| ((len - 1 - k + r) % len.max(1)) as u32));
        }
        let t = TTable::from_parts(
            row_ptr.clone(),
            cols,
            order.clone(),
            vec![1; lens.len()],
            long + 1,
        )
        .unwrap();
        for (r, w) in row_ptr.windows(2).enumerate() {
            let got: Vec<u32> = t.row_order(r as u32).iter().map(|k| k as u32).collect();
            assert_eq!(got, order[w[0]..w[1]], "row {r}");
            assert_eq!(
                matches!(t.row_order(r as u32), RowOrder::Long(_)),
                w[1] - w[0] > SHORT_ROW
            );
        }
        assert_eq!(t.order().collect::<Vec<_>>(), order);
    }

    #[test]
    fn rows_and_order() {
        // ids: <eps>=1 |||=2 a=3 b=4 x=5 y=6 z=7
        let c = corpus(b"a b ||| x y\nb ||| z x\n");
        let ip = initial_pass(&c, TableOptions::default());
        let t = &ip.table;
        assert_eq!(t.n_rows(), 5);
        assert_eq!(&t.cols()[t.row(3)], [5, 6]);
        assert_eq!(&t.cols()[t.row(4)], [5, 6, 7]);
        assert_eq!(&t.cols()[t.row(1)], [5, 6, 7]);
        assert!(t.row(0).is_empty() && t.row(2).is_empty());
        // NULL row insertion sequence x y z -> libstdc++ order
        let want: Vec<u32> = u32_order(&[5, 6, 7])
            .iter()
            .map(|&i| [5, 6, 7][i as usize])
            .collect();
        let got: Vec<u32> = t.iter_hash_order(1).map(|k| t.cols()[k]).collect();
        assert_eq!(got, want);
        // row b: sequence x y z (line 1) then z x (already seen)
        let got: Vec<u32> = t.iter_hash_order(4).map(|k| t.cols()[k]).collect();
        assert_eq!(got, want);
        assert_eq!(t.find(4, 7), Some(t.row(4).start + 2));
        assert_eq!(t.find(3, 7), None);
        assert_eq!(ip.n_target_tokens, 4.0);
        assert_eq!(ip.tot_len_ratio, 1.0 + 2.0);
        assert_eq!(ip.size_counts.len(), 2);
    }

    #[test]
    fn no_null_and_reverse() {
        let c = corpus(b"a b ||| x\n");
        let fwd = initial_pass(
            &c,
            TableOptions {
                use_null: false,
                ..Default::default()
            },
        );
        assert!(fwd.table.row(1).is_empty());
        let rev = initial_pass(
            &c,
            TableOptions {
                reverse: true,
                ..Default::default()
            },
        );
        // reverse: source = x (id 5); NULL row and row x both have targets a b
        assert_eq!(&rev.table.cols()[rev.table.row(5)], [3, 4]);
        assert_eq!(&rev.table.cols()[rev.table.row(1)], [3, 4]);
        assert_eq!(
            rev.size_counts,
            [SizeCount {
                m: 2,
                n: 1,
                count: 1
            }]
        );
    }

    #[test]
    fn eps_as_source_word() {
        // <eps> as a source word is the NULL row: with -N it is an ordinary row of its lines.
        let c = corpus(b"a ||| r\n<eps> ||| p q\n");
        let with_null = initial_pass(&c, TableOptions::default());
        let t = &with_null.table;
        assert_eq!(&t.cols()[t.row(1)], [4, 5, 6]); // r p q (ids: a=3 r=4 p=5 q=6)
        let no_null = initial_pass(
            &c,
            TableOptions {
                use_null: false,
                ..Default::default()
            },
        );
        let t = &no_null.table;
        assert_eq!(&t.cols()[t.row(1)], [5, 6]); // p q
    }

    /// A corpus with repeated words, several occurrences per line and lines of varied length.
    fn varied_corpus() -> Corpus {
        let mut t = String::new();
        let mut x = 12345u64;
        for _ in 0..400 {
            for side in 0..2 {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let n = (x >> 60) as usize;
                for k in 0..n {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    t += &format!(
                        "{}{} ",
                        if side == 0 { "s" } else { "t" },
                        (x >> 59) * (k as u64 % 3)
                    );
                }
                if side == 0 {
                    t += "||| ";
                }
            }
            t += "\n";
        }
        Corpus::from_joined_bytes(t.as_bytes(), &LoadOptions::default()).unwrap()
    }

    /// Review #12: parts are cut so that their `u32` offsets cannot overflow; the postings
    /// (and so the table) are the same whatever the cut.
    #[test]
    fn postings_do_not_depend_on_the_part_limit() {
        let c = varied_corpus();
        let n_vocab = c.vocab.len();
        let longest = (0..c.len()).map(|k| c.first.line_len(k)).max().unwrap();
        let cancel = Cancel::default();
        // one thread: one part unless the limit cuts it
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let base = pool.install(|| Postings::build(&c.first, n_vocab, &cancel));
        assert_eq!(base.parts.len(), 1);
        for limit in [longest, longest + 3, 50, 997] {
            let p = pool.install(|| Postings::build_with(&c.first, n_vocab, limit, &cancel));
            assert!(p.parts.len() > base.parts.len());
            for (_, out) in &p.parts {
                assert!(out.len() <= limit);
            }
            for e in 0..n_vocab as u32 {
                assert!(base.lines(e).eq(p.lines(e)), "row {e}, limit {limit}");
            }
        }
    }
}
