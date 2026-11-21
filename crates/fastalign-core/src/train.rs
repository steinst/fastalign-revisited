//! Deterministic parallel trainer: bit-identical to [`crate::oracle`] (and so to
//! `OMP_NUM_THREADS=1 fast_align`) at every thread count.
//!
//! # Design
//!
//! fast_align's E-step adds every posterior to its cell in corpus order (line, target position
//! `j`, NULL then source position `i`, spec §8), and sums `likelihood`, `c0` and `emp_feat` as
//! running sums within each `-b` buffer (§9). f64 addition is not associative, so both orders
//! must be kept exactly. The two orders are different, and so are the two phases here:
//!
//! **Phase A, line-parallel.** The corpus is cut into *groups of whole fast_align buffers*.
//! Each task runs the E-step over its lines exactly as fast_align does, sums the three
//! statistics per buffer in fast_align's order, and stores every target token's column sum
//! `sum_j` (one f64 per target token). It adds nothing to any count. In the final iteration
//! this phase is the whole iteration: Viterbi decoding, output lines in corpus order.
//!
//! **Phase B, row-parallel (with the M-step fused in).** Each source row `e` (one task, or
//! several for hot rows, split by target-id range) walks the lines that contain `e`, in corpus
//! order, and recomputes each of its posteriors `p = (prob(e, f) ⊗ a) ⊘ sum_j` from the same
//! operands as phase A. So `p` has the same bits, and the cell receives exactly fast_align's
//! sequence of additions. Counts live in a row-local buffer. When the row is finished, it is
//! normalised in its hash order (§11) and written over the row's probabilities. This is safe
//! because phase B of row `e` reads only row `e`'s probabilities. The global `counts` table of
//! fast_align (and of the oracle) does not exist here.
//!
//! Both phases do the same arithmetic as the oracle, only distributed differently, so the
//! results do not depend on the thread count, the group size, row splitting or the prior cache.
//! Phase A's parallelism is the number of fast_align buffers (corpus lines / `-b`), because
//! `emp_feat`'s running sum over a buffer can only be computed in one sequence.
//!
//! **Prior cache.** With `-d`, the prior `a(j, i)` depends only on `(j, i, m, n, λ)`. Per
//! iteration it is computed once per distinct length pair: the normaliser `ComputeZ(j+1, m, n)
//! ⊘ (1 − p0)` for every pair, and the full `a` table for the most frequent pairs up to
//! [`ParallelOptions::prior_cache_entries`]. Other pairs evaluate `exp` per cell, exactly as
//! fast_align does.

use std::fmt::Write as _;
use std::io::{self, Write};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::cancel::Cancel;
use crate::corpus::{part_ranges, Corpus, Side, Vocab, MAX_PART_ENTRIES, NULL_ID};
use crate::math::{self, G};
use crate::oracle::{IterationStats, TrainOptions, UNINITIALISED_PROB};
use crate::ttable::{
    initial_pass_with, InitialPass, RowOrder, TTable, TableOptions, CANCEL_STRIDE,
};

/// `ln 2`, as GCC folds `log(2)` (spec §10.1).
const LN_2: f64 = std::f64::consts::LN_2;

/// Tuning knobs of the parallel trainer. None of them changes any output bit.
#[derive(Clone, Debug)]
pub struct ParallelOptions {
    /// f64 entries of the per-length-pair prior table (`-d` only), per direction. 0 disables
    /// the table (the per-pair normalisers are always cached).
    pub prior_cache_entries: usize,
    /// Approximate number of lines per phase-A task (always whole fast_align buffers).
    pub group_lines: usize,
    /// Phase-B work above which a row is split into several tasks by target-id range; `None`
    /// chooses from the thread count.
    pub split_cost: Option<u64>,
    /// `u32` entries (4 B each) of the dense lookup index, per direction: the busiest rows get
    /// a vocabulary-sized target id → cell array, so their lookups need no search. 0 disables
    /// it.
    pub dense_index_entries: usize,
    /// Rows shorter than this are never given a dense index (their search is cheap).
    pub dense_min_row_len: usize,
    /// Phase B walks the rows with a dense index over the corpus in blocks of this many lines,
    /// all workers on the same block, so the block's target tokens, column sums and prior
    /// tables are read from cache by every such row (PERF-TODO 9). 0 disables blocking; it is
    /// also off with one thread.
    pub block_lines: usize,
    /// Rows without a dense index at least this long get a sampled key index (every 16th
    /// target id, ≈ 0.25 B per cell) that narrows their search to one 16-key block
    /// (PERF-TODO 2c). `usize::MAX` disables it.
    pub sample_min_row_len: usize,
    /// Most `f64` cells of one worker's per-line probability matrix (one cell per distinct
    /// source × distinct target word of a line). A line needing more is served column by
    /// column from a cache of this many cells, filled by direct table lookups (review
    /// performance item 8); the values, and so every output bit, are the same.
    pub line_cells: usize,
    /// Bytes of the opt-in cell-offset cache (review performance item 6), per direction; 0
    /// (the default) disables it. The first E-step after the first M-step stores, for the
    /// lines that fit the budget (in corpus order), the table cell of every distinct word pair
    /// it looks up; later E-steps and decoding read the probabilities through those offsets
    /// instead of searching the table. Lines served column by column (see `line_cells`) are
    /// not cached. The cell offsets of a line never change, so neither does any value.
    pub offset_cache_bytes: usize,
}

impl Default for ParallelOptions {
    fn default() -> Self {
        ParallelOptions {
            prior_cache_entries: 4 << 20,
            group_lines: 4096,
            split_cost: None,
            dense_index_entries: 16 << 20,
            dense_min_row_len: 1024,
            block_lines: 65536,
            sample_min_row_len: 64,
            line_cells: 1 << 20,
            offset_cache_bytes: 0,
        }
    }
}

/// fast_align's progress output (spec §15.4).
#[derive(Default)]
struct Progress {
    flag: bool,
}

impl Progress {
    fn tick(&mut self, lc: usize, log: &mut Vec<u8>) {
        if lc % 1000 == 0 {
            log.push(b'.');
            self.flag = true;
        }
        if lc % 50000 == 0 {
            let _ = writeln!(log, " [{lc}]");
            self.flag = false;
        }
    }

    fn finish(&mut self, log: &mut Vec<u8>) {
        if self.flag {
            log.push(b'\n');
        }
    }
}

/// Wall time spent in each phase of one direction's training (measurement only; Step 9a).
/// Summed over iterations. With two directions trained concurrently the phases of the two
/// overlap in time, so these are per-direction wall times, not shares of the run.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseTimes {
    /// `initial_pass`: table structure, hash order, `size_counts`.
    pub initial_pass: Duration,
    /// Length pairs, per-row line lists, target frequencies.
    pub setup: Duration,
    /// Per-iteration prior cache (`-d`).
    pub prior: Duration,
    /// Phase A of the training iterations (E-step column sums and statistics).
    pub e_step: Duration,
    /// Phase B (count accumulation in corpus order fused with the M-step normalisation).
    pub m_step: Duration,
    /// Tension optimisation (`-d -o`).
    pub tension: Duration,
    /// Final iteration: Viterbi decoding and output formatting.
    pub decode: Duration,
    /// Number of training (non-final) iterations measured.
    pub train_iterations: usize,
}

/// Per-buffer partial sums (spec §9).
#[derive(Clone, Copy, Default)]
struct Partial {
    likelihood: f64,
    c0: f64,
    emp_feat: f64,
}

// ------------------------------------------------------------------------------------------
// Static structure built once per direction

/// Distinct `(m, n)` length pairs, the pair of every line, and which pairs get a full prior
/// table.
struct LengthPairs {
    /// Pair index of every line.
    line_pair: Vec<u32>,
    /// `(m, n)` = (target length, source length) per pair.
    pairs: Vec<(u32, u32)>,
    /// Offsets of each pair's `m` normalisers in the per-iteration `az` array.
    az_off: Vec<usize>,
    /// Offset of each pair's `m × n` prior table (by source position, then `j`), if cached.
    a_off: Vec<Option<usize>>,
    a_len: usize,
}

impl LengthPairs {
    fn build(src: &Side, trg: &Side, budget: usize, cancel: &Cancel) -> Self {
        let n_lines = src.offsets().len() - 1;
        let mut index: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut pairs = Vec::new();
        let mut counts: Vec<u64> = Vec::new();
        let mut line_pair = Vec::with_capacity(n_lines);
        for k in 0..n_lines {
            if k % CANCEL_STRIDE == 0 {
                cancel.check();
            }
            let key = (trg.line_len(k) as u32, src.line_len(k) as u32);
            let p = *index.entry(key).or_insert_with(|| {
                pairs.push(key);
                counts.push(0);
                (pairs.len() - 1) as u32
            });
            counts[p as usize] += 1;
            line_pair.push(p);
        }
        let mut az_off = Vec::with_capacity(pairs.len() + 1);
        let mut off = 0;
        for &(m, _) in &pairs {
            az_off.push(off);
            off += m as usize;
        }
        az_off.push(off);
        // Most frequent pairs first (a pair's table saves `count × m × n` exps per phase).
        let mut by_count: Vec<usize> = (0..pairs.len()).collect();
        by_count.sort_by_key(|&p| (std::cmp::Reverse(counts[p]), p));
        let mut a_off = vec![None; pairs.len()];
        let mut a_len = 0;
        for p in by_count {
            let (m, n) = pairs[p];
            let size = m as usize * n as usize;
            if size > 0 && a_len + size <= budget {
                a_off[p] = Some(a_len);
                a_len += size;
            }
        }
        LengthPairs {
            line_pair,
            pairs,
            az_off,
            a_off,
            a_len,
        }
    }
}

/// For every table row, the lines that contain the row's source word (each line once, in
/// corpus order), stored as one local CSR per corpus part; plus the phase-B work per row.
/// The NULL row (with the NULL word) is implicit: every line.
///
/// Each entry also keeps the word's 1-based source position in that line when it occurs there
/// exactly once (and the position fits a `u16`), else 0; phase B then needs no scan of the
/// source line for most entries (PERF-TODO 3a).
struct RowLines {
    parts: Vec<RowPart>,
    /// Phase-B additions per row (Σ over its lines of `m × occurrences`, plus `m` per line for
    /// the NULL row).
    cost: Vec<u64>,
    /// Whether `<eps>` (the NULL word's id) occurs as a source token anywhere.
    eps_in_src: bool,
}

/// One corpus part of [`RowLines`]: CSR offsets per row, line ids, single positions.
struct RowPart {
    start: Vec<u32>,
    lines: Vec<u32>,
    first: Vec<u16>,
}

impl RowLines {
    fn build(src: &Side, trg: &Side, n_rows: usize, use_null: bool, cancel: &Cancel) -> Self {
        Self::build_with(src, trg, n_rows, use_null, MAX_PART_ENTRIES, cancel)
    }

    /// `max_entries` bounds every part's entries (its `u32` offsets; at most one per source
    /// token); tests lower it.
    fn build_with(
        src: &Side,
        trg: &Side,
        n_rows: usize,
        use_null: bool,
        max_entries: usize,
        cancel: &Cancel,
    ) -> Self {
        let n_lines = src.offsets().len() - 1;
        let n_parts = rayon::current_num_threads().clamp(1, n_lines.max(1));
        let null = NULL_ID as usize;
        // Per part: local CSR (start, lines), cost per row, whether `<eps>` is a source word.
        type PartOut = (RowPart, Vec<u64>, bool);
        let built: Vec<PartOut> = part_ranges(src, n_parts, max_entries)
            .into_par_iter()
            .map(|lines| {
                let mut start = vec![0u32; n_rows + 1];
                let mut cost = vec![0u64; n_rows];
                let mut stamp = vec![u32::MAX; n_rows];
                let mut eps = false;
                for k in lines.clone() {
                    if k % CANCEL_STRIDE == 0 {
                        cancel.check();
                    }
                    let m = trg.line_len(k) as u64;
                    if use_null && n_rows > null {
                        cost[null] += m;
                    }
                    for &e in src.line(k) {
                        let e = e as usize;
                        eps |= e == null;
                        cost[e] += m;
                        if stamp[e] != k as u32 {
                            stamp[e] = k as u32;
                            if !(use_null && e == null) {
                                start[e + 1] += 1;
                            }
                        }
                    }
                }
                for e in 0..n_rows {
                    start[e + 1] += start[e];
                }
                let mut pos = start.clone();
                let mut out = vec![0u32; start[n_rows] as usize];
                let mut first = vec![0u16; start[n_rows] as usize];
                stamp.fill(u32::MAX);
                for k in lines {
                    if k % CANCEL_STRIDE == 0 {
                        cancel.check();
                    }
                    for (i, &e) in src.line(k).iter().enumerate() {
                        let e = e as usize;
                        if use_null && e == null {
                            continue;
                        }
                        if stamp[e] != k as u32 {
                            stamp[e] = k as u32;
                            out[pos[e] as usize] = k as u32;
                            first[pos[e] as usize] = u16::try_from(i + 1).unwrap_or(0);
                            pos[e] += 1;
                        } else {
                            // A second occurrence: phase B scans this line.
                            first[pos[e] as usize - 1] = 0;
                        }
                    }
                }
                (
                    RowPart {
                        start,
                        lines: out,
                        first,
                    },
                    cost,
                    eps,
                )
            })
            .collect();
        let mut cost = vec![0u64; n_rows];
        let mut eps_in_src = false;
        let mut parts = Vec::with_capacity(built.len());
        for (part, c, eps) in built {
            for (t, x) in cost.iter_mut().zip(&c) {
                *t += x;
            }
            eps_in_src |= eps;
            parts.push(part);
        }
        RowLines {
            parts,
            cost,
            eps_in_src,
        }
    }

    /// Lines containing source word `e` with its single position or 0 (not for the implicit
    /// NULL row).
    fn lines(&self, e: u32) -> impl Iterator<Item = (u32, u16)> + '_ {
        let e = e as usize;
        self.parts.iter().flat_map(move |p| {
            let r = p.start[e] as usize..p.start[e + 1] as usize;
            p.lines[r.clone()]
                .iter()
                .copied()
                .zip(p.first[r].iter().copied())
        })
    }

    /// Like [`RowLines::lines`], but resumable: continues from `cur` (start with `(0, 0)`) and
    /// stops before the first line `>= end`.
    fn lines_before<'a>(
        &'a self,
        e: u32,
        cur: &'a mut (usize, usize),
        end: u32,
    ) -> impl Iterator<Item = (u32, u16)> + 'a {
        let e = e as usize;
        std::iter::from_fn(move || loop {
            let p = self.parts.get(cur.0)?;
            cur.1 = cur.1.max(p.start[e] as usize);
            if cur.1 < p.start[e + 1] as usize {
                let line = p.lines[cur.1];
                if line >= end {
                    return None;
                }
                let first = p.first[cur.1];
                cur.1 += 1;
                return Some((line, first));
            }
            cur.0 += 1;
            cur.1 = 0;
        })
    }

    fn memory_bytes(&self) -> usize {
        self.parts
            .iter()
            .map(|p| p.start.capacity() * 4 + p.lines.capacity() * 4 + p.first.capacity() * 2)
            .sum::<usize>()
            + self.cost.capacity() * 8
    }
}

/// Dense lookup index of the busiest rows: for a selected row, a vocabulary-sized array from
/// target id to the offset within the row (`u32::MAX` where the row has no such cell). Chosen
/// once per direction by phase-B cost (≈ the row's cell visits per iteration), within
/// [`ParallelOptions::dense_index_entries`]. It returns exactly the offset the binary search
/// would (PERF-TODO 2).
struct DenseRows {
    /// Slot of every row, `u32::MAX` if none.
    slot: Vec<u32>,
    index: Vec<u32>,
    n_vocab: usize,
}

impl DenseRows {
    fn build(table: &TTable, cost: &[u64], n_vocab: usize, popts: &ParallelOptions) -> Self {
        let n_rows = table.n_rows();
        let max_slots = popts.dense_index_entries / n_vocab.max(1);
        let mut by: Vec<u32> = (0..n_rows as u32)
            .filter(|&e| {
                let len = table.row(e).len();
                len > 0 && len >= popts.dense_min_row_len
            })
            .collect();
        by.sort_by_key(|&e| (std::cmp::Reverse(cost[e as usize]), e));
        by.truncate(max_slots);
        let mut slot = vec![u32::MAX; n_rows];
        for (k, &e) in by.iter().enumerate() {
            slot[e as usize] = k as u32;
        }
        let mut index = vec![u32::MAX; by.len() * n_vocab];
        let cols = table.cols();
        index
            .par_chunks_mut(n_vocab.max(1))
            .zip(by.par_iter())
            .for_each(|(dst, &e)| {
                for (k, &f) in cols[table.row(e)].iter().enumerate() {
                    dst[f as usize] = k as u32;
                }
            });
        DenseRows {
            slot,
            index,
            n_vocab,
        }
    }

    /// The dense array of row `e`, if it has one.
    #[inline]
    fn row(&self, e: u32) -> Option<&[u32]> {
        let s = *self.slot.get(e as usize)?;
        if s == u32::MAX {
            return None;
        }
        let s = s as usize * self.n_vocab;
        Some(&self.index[s..s + self.n_vocab])
    }

    fn memory_bytes(&self) -> usize {
        self.slot.capacity() * 4 + self.index.capacity() * 4
    }
}

/// Keys sampled from a row at this stride ([`SampledRows`]).
const SAMPLE: usize = 16;

/// Sampled key index of the rows searched by binary search (PERF-TODO 2c): for each such row
/// every [`SAMPLE`]-th target id, starting with the first. A search then probes the small,
/// well-cached sample array and reads one block of `SAMPLE` keys of the row, instead of
/// probing the whole row (whose deep probes are cache misses). It returns exactly the offset
/// the binary search would: keys are unique and sorted, and every searched key is present.
struct SampledRows {
    /// Range of every row's samples in `keys` (empty: plain binary search).
    start: Vec<usize>,
    keys: Vec<u32>,
}

impl SampledRows {
    fn build(table: &TTable, dense: &DenseRows, min_len: usize) -> Self {
        let n_rows = table.n_rows();
        let mut start = Vec::with_capacity(n_rows + 1);
        start.push(0usize);
        let mut total = 0usize;
        for e in 0..n_rows as u32 {
            let len = table.row(e).len();
            if len >= min_len && dense.row(e).is_none() {
                total += len.div_ceil(SAMPLE);
            }
            start.push(total);
        }
        let cols = table.cols();
        let mut keys = Vec::with_capacity(total);
        for e in 0..n_rows as u32 {
            if start[e as usize + 1] > start[e as usize] {
                keys.extend(cols[table.row(e)].iter().step_by(SAMPLE));
            }
        }
        SampledRows { start, keys }
    }

    /// Samples of row `e` (empty: the row is searched by plain binary search).
    #[inline]
    fn samples(&self, e: u32) -> &[u32] {
        &self.keys[self.start[e as usize]..self.start[e as usize + 1]]
    }

    /// Start of the block of `f` in its row, from the row's non-empty `samples`: the last
    /// block whose first key is <= f (`samples[0]` is the row's smallest key).
    #[inline]
    fn block(samples: &[u32], f: u32) -> usize {
        (samples.partition_point(|&x| x <= f) - 1) * SAMPLE
    }

    /// Offset of `f` in `row`, given the start `lo` of its block (`f` must be present).
    #[inline]
    fn in_block(row: &[u32], lo: usize, f: u32) -> usize {
        let blk = &row[lo..(lo + SAMPLE).min(row.len())];
        // Keys before the block are smaller than f, keys after it larger.
        let k = lo
            + match <&[u32; SAMPLE]>::try_from(blk) {
                Ok(b) => b.iter().map(|&x| usize::from(x < f)).sum::<usize>(),
                Err(_) => blk.iter().filter(|&&x| x < f).count(),
            };
        debug_assert_eq!(row[k], f);
        k
    }

    fn memory_bytes(&self) -> usize {
        self.start.capacity() * 8 + self.keys.capacity() * 4
    }
}

/// Prior of one line: `a(j, i)` for 0-based target position `j` and 1-based source position
/// `i`, exactly `UnnormalizedProb(j+1, i, m, n, λ) ⊘ (ComputeZ(j+1, m, n, λ) ⊘ (1 − p0))`.
#[derive(Clone, Copy)]
struct LinePrior<'a> {
    az: &'a [f64],
    table: Option<&'a [f64]>,
    m: u32,
    n: u32,
    lambda: f64,
}

impl LinePrior<'_> {
    #[inline]
    fn get(&self, j: usize, i: usize) -> f64 {
        match self.table {
            // Stored by source position (column-major): phase B reads one source position
            // over all `j` of a line, contiguously.
            Some(t) => t[(i - 1) * self.m as usize + j],
            None => {
                math::unnormalized_prob(j as u32 + 1, i as u32, self.m, self.n, self.lambda)
                    / self.az[j]
            }
        }
    }
}

/// Translation probabilities of one line, with each distinct `(e, f)` pair of the line looked
/// up in the table once (a third of a line's cells repeat a pair, PERF-TODO 1). The values are
/// copies of the table's `f64`s, so every product formed from them is unchanged.
///
/// The matrix of a line holds at most `cap` cells. A longer line (many distinct words on both
/// sides) holds a tile of whole target columns instead: when a target position's column is not
/// in the tile, the tile is replaced by the columns of the next distinct targets the line needs
/// from that position on. The column values, and the order in which a column is summed, are
/// the same either way.
struct LineLookup {
    /// Distinct-source index of every source position.
    src_d: Vec<u32>,
    /// Distinct-target index of every target position.
    trg_d: Vec<u32>,
    src_words: Vec<u32>,
    trg_words: Vec<u32>,
    /// Full mode: `mat[dt * n_ds + ds]` = `prob(src_words[ds], trg_words[dt])`. Column mode:
    /// `mat[slot * n_ds + ds]` for the target held in `slot`.
    mat: Vec<f64>,
    /// `prob(NULL, trg_words[dt])`.
    null: Vec<f64>,
    n_ds: usize,
    /// Most cells of `mat`.
    cap: usize,
    /// Column mode for this line.
    columns: bool,
    /// Column mode: the slot holding each distinct target, or [`NO_SLOT`].
    slot_of: Vec<u32>,
    /// Column mode: the distinct targets held by the slots, in slot order.
    owner: Vec<u32>,
    /// Column mode: number of column slots.
    slots: usize,
    /// Column mode: the target words of the tile being loaded.
    keys: Vec<u32>,
    /// Vocabulary-sized stamps and slots for finding the distinct words.
    mark: Vec<u32>,
    slot: Vec<u32>,
    stamp: u32,
    /// Row offsets found by [`search_all`].
    hit: Vec<u32>,
}

const NO_SLOT: u32 = u32::MAX;

/// Hint the CPU to load the cache line of `x` (no effect on any value).
#[inline(always)]
fn prefetch<T>(x: &T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch is only a hint; it never faults and reads nothing into the program.
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch::<_MM_HINT_T0>((x as *const T).cast());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = x;
}

/// Offsets of all `keys` from a row's dense index `d`, in two passes whose cache misses
/// overlap: the index entries are prefetched, then each found cell's probability.
#[inline]
fn dense_all(d: &[u32], prob: &[f64], keys: &[u32], hit: &mut Vec<u32>) {
    hit.clear();
    for &f in keys {
        prefetch(&d[f as usize]);
    }
    hit.extend(keys.iter().map(|&f| {
        let k = d[f as usize];
        prefetch(&prob[k as usize]);
        k
    }));
}

/// Offsets of all `keys` in the sorted row `e` (every key is present), with each found cell's
/// probability in `prob` (the row's probabilities) prefetched. For a sampled row the searches
/// run in passes over all keys, so that their cache misses overlap: every key's block start
/// (from the cached samples) with a prefetch of the block, then the offsets within the blocks.
#[inline]
fn search_all(
    sampled: &SampledRows,
    e: u32,
    row: &[u32],
    prob: &[f64],
    keys: &[u32],
    hit: &mut Vec<u32>,
) {
    hit.clear();
    let samples = sampled.samples(e);
    if samples.is_empty() {
        hit.extend(keys.iter().map(|f| {
            let k = row
                .binary_search(f)
                .expect("every (source, target) pair of the corpus is in the table");
            prefetch(&prob[k]);
            k as u32
        }));
        return;
    }
    hit.extend(keys.iter().map(|&f| {
        let lo = SampledRows::block(samples, f);
        prefetch(&row[lo]);
        prefetch(&row[(lo + SAMPLE).min(row.len()) - 1]);
        lo as u32
    }));
    for (h, &f) in hit.iter_mut().zip(keys) {
        let k = SampledRows::in_block(row, *h as usize, f);
        prefetch(&prob[k]);
        *h = k as u32;
    }
}

/// The opt-in cell-offset cache ([`ParallelOptions::offset_cache_bytes`]): for lines
/// `0..covered()`, the table cell of every distinct pair `LineLookup::prepare` looks up, in its
/// matrix order followed by the NULL row's cells.
struct OffsetCache {
    /// Line `l`'s entries are `off[start[l]..start[l + 1]]` (none for a column-mode line).
    start: Vec<u64>,
    off: Vec<u32>,
    /// Set once an E-step has stored every covered line's offsets.
    filled: bool,
    /// Distinct pairs looked up per pass over the corpus, and how many of them are cached.
    pairs_total: u64,
    pairs_cached: u64,
}

impl OffsetCache {
    /// The cache for the longest prefix of lines whose entries (4 B each, plus 8 B per line)
    /// fit `budget` bytes; `None` if nothing fits or cell offsets do not fit `u32`.
    #[allow(clippy::too_many_arguments)]
    fn build(
        src: &Side,
        trg: &Side,
        n_vocab: usize,
        use_null: bool,
        line_cells: usize,
        n_cells: usize,
        budget: usize,
        cancel: &Cancel,
    ) -> Option<Self> {
        if budget == 0 || n_cells > u32::MAX as usize {
            return None;
        }
        let n = src.offsets().len() - 1;
        let cap = line_cells.max(1);
        let marks: Mutex<Vec<(Vec<u32>, u32)>> = Mutex::new(Vec::new());
        // (entries if cached, distinct pairs) per line.
        let per_line: Vec<(u64, u64)> = (0..n)
            .into_par_iter()
            .with_min_len(4096)
            .map_init(
                || marks.lock().unwrap().pop().unwrap_or((vec![0; n_vocab], 0)),
                |(mark, stamp), l| {
                    if l % CANCEL_STRIDE == 0 {
                        cancel.check();
                    }
                    let mut distinct = |words: &[u32]| {
                        if *stamp == u32::MAX {
                            mark.fill(0);
                            *stamp = 0;
                        }
                        *stamp += 1;
                        let mut k = 0usize;
                        for &w in words {
                            if mark[w as usize] != *stamp {
                                mark[w as usize] = *stamp;
                                k += 1;
                            }
                        }
                        k
                    };
                    let n_ds = distinct(src.line(l));
                    let n_dt = distinct(trg.line(l));
                    let null = if use_null { n_dt } else { 0 };
                    let pairs = (n_ds * n_dt + null) as u64;
                    let cached = if n_ds.saturating_mul(n_dt) > cap {
                        0
                    } else {
                        pairs
                    };
                    (cached, pairs)
                },
            )
            .collect();
        let mut start = vec![0u64];
        let mut bytes = 8usize;
        let mut pairs_cached = 0;
        for &(c, _) in &per_line {
            let need = 8 + 4 * c as usize;
            if bytes + need > budget {
                break;
            }
            bytes += need;
            pairs_cached += c;
            start.push(start.last().unwrap() + c);
        }
        if start.len() == 1 {
            return None;
        }
        Some(OffsetCache {
            off: vec![0; *start.last().unwrap() as usize],
            start,
            filled: false,
            pairs_total: per_line.iter().map(|x| x.1).sum(),
            pairs_cached,
        })
    }

    fn covered(&self) -> usize {
        self.start.len() - 1
    }

    fn range(&self, line: usize) -> Range<usize> {
        self.start[line] as usize..self.start[line + 1] as usize
    }

    fn line(&self, line: usize) -> Cached<'_> {
        if line < self.covered() {
            Cached::Use(&self.off[self.range(line)])
        } else {
            Cached::No
        }
    }

    fn memory_bytes(&self) -> usize {
        self.start.capacity() * 8 + self.off.capacity() * 4
    }
}

/// A line's part of the [`OffsetCache`] in [`LineLookup::prepare`].
enum Cached<'a> {
    /// Not cached: look every pair up.
    No,
    /// Look every pair up and store its cell offset here (`mat` order, then the NULL row's).
    Fill(&'a mut [u32]),
    /// Read the probabilities at these cell offsets.
    Use(&'a [u32]),
}

impl LineLookup {
    fn new(n_vocab: usize, cap: usize) -> Self {
        LineLookup {
            src_d: Vec::new(),
            trg_d: Vec::new(),
            src_words: Vec::new(),
            trg_words: Vec::new(),
            mat: Vec::new(),
            null: Vec::new(),
            n_ds: 0,
            cap: cap.max(1),
            columns: false,
            slot_of: Vec::new(),
            owner: Vec::new(),
            slots: 0,
            keys: Vec::new(),
            mark: vec![0; n_vocab],
            slot: vec![0; n_vocab],
            stamp: 0,
            hit: Vec::new(),
        }
    }

    /// Distinct words of `words` into `out`, the distinct index of each position into `d`.
    fn distinct(&mut self, words: &[u32], stamp: u32, src: bool) {
        let (d, out) = if src {
            (&mut self.src_d, &mut self.src_words)
        } else {
            (&mut self.trg_d, &mut self.trg_words)
        };
        d.clear();
        out.clear();
        for &w in words {
            let w = w as usize;
            if self.mark[w] != stamp {
                self.mark[w] = stamp;
                self.slot[w] = out.len() as u32;
                out.push(w as u32);
            }
            d.push(self.slot[w]);
        }
    }

    fn prepare(&mut self, ctx: &Ctx, src: &[u32], trg: &[u32], cache: Cached) {
        if !ctx.init {
            // Every probability is the initial constant: one distinct pair.
            self.src_d.clear();
            self.src_d.resize(src.len(), 0);
            self.trg_d.clear();
            self.trg_d.resize(trg.len(), 0);
            self.n_ds = 1;
            self.columns = false;
            self.mat.clear();
            self.mat.push(UNINITIALISED_PROB);
            self.null.clear();
            self.null.push(UNINITIALISED_PROB);
            return;
        }
        if self.stamp > u32::MAX - 4 {
            self.mark.fill(0);
            self.stamp = 0;
        }
        self.stamp += 2;
        let s = self.stamp;
        self.distinct(src, s, true);
        self.distinct(trg, s + 1, false);
        let (n_ds, n_dt) = (self.src_words.len(), self.trg_words.len());
        self.n_ds = n_ds;
        let cols = ctx.table.cols();
        let mut hit = std::mem::take(&mut self.hit);
        self.mat.clear();
        self.columns = n_ds.saturating_mul(n_dt) > self.cap;
        if self.columns {
            // At least one column, whatever the budget: a column is O(line length), as
            // `probs` is.
            self.slots = (self.cap / n_ds).clamp(1, n_dt);
            self.mat.resize(self.slots * n_ds, 0.0);
            self.slot_of.clear();
            self.slot_of.resize(n_dt, NO_SLOT);
            self.owner.clear();
        } else if let Cached::Use(off) = cache {
            // Every offset was stored by a `Fill` of this line: same words, same cells.
            let n_null = if ctx.use_null { n_dt } else { 0 };
            assert_eq!(off.len(), n_ds * n_dt + n_null, "offset cache out of step");
            let (m_off, n_off) = off.split_at(n_ds * n_dt);
            const AHEAD: usize = 8;
            for &o in &m_off[..AHEAD.min(m_off.len())] {
                prefetch(&ctx.prob[o as usize]);
            }
            self.mat.extend(m_off.iter().enumerate().map(|(k, &o)| {
                if let Some(&p) = m_off.get(k + AHEAD) {
                    prefetch(&ctx.prob[p as usize]);
                }
                ctx.prob[o as usize]
            }));
            self.null.clear();
            self.null
                .extend(n_off.iter().map(|&o| ctx.prob[o as usize]));
            self.hit = hit;
            return;
        } else {
            let mut fill = match cache {
                Cached::Fill(off) => Some(off),
                _ => None,
            };
            self.mat.resize(n_ds * n_dt, 0.0);
            for (ds, &e) in self.src_words.iter().enumerate() {
                let r = ctx.table.row(e);
                let prob = &ctx.prob[r.clone()];
                if let Some(d) = ctx.dense.row(e) {
                    dense_all(d, prob, &self.trg_words, &mut hit);
                } else {
                    search_all(
                        ctx.sampled,
                        e,
                        &cols[r.clone()],
                        prob,
                        &self.trg_words,
                        &mut hit,
                    );
                }
                for (dt, &k) in hit.iter().enumerate() {
                    self.mat[dt * n_ds + ds] = prob[k as usize];
                }
                if let Some(off) = fill.as_deref_mut() {
                    for (dt, &k) in hit.iter().enumerate() {
                        off[dt * n_ds + ds] = (r.start + k as usize) as u32;
                    }
                }
            }
            if let Some(off) = fill {
                if ctx.use_null {
                    let r = ctx.table.row(NULL_ID);
                    let prob = &ctx.prob[r.clone()];
                    self.null.clear();
                    if let Some(d) = ctx.dense.row(NULL_ID) {
                        dense_all(d, prob, &self.trg_words, &mut hit);
                    } else {
                        search_all(
                            ctx.sampled,
                            NULL_ID,
                            &cols[r.clone()],
                            prob,
                            &self.trg_words,
                            &mut hit,
                        );
                    }
                    self.null.extend(hit.iter().map(|&k| prob[k as usize]));
                    for (o, &k) in off[n_ds * n_dt..].iter_mut().zip(&hit) {
                        *o = (r.start + k as usize) as u32;
                    }
                } else {
                    self.null.clear();
                }
                self.hit = hit;
                return;
            }
        }
        self.null.clear();
        if ctx.use_null {
            let r = ctx.table.row(NULL_ID);
            let prob = &ctx.prob[r.clone()];
            if let Some(d) = ctx.dense.row(NULL_ID) {
                dense_all(d, prob, &self.trg_words, &mut hit);
            } else {
                search_all(
                    ctx.sampled,
                    NULL_ID,
                    &cols[r],
                    prob,
                    &self.trg_words,
                    &mut hit,
                );
            }
            self.null.extend(hit.iter().map(|&k| prob[k as usize]));
        }
        self.hit = hit;
    }

    #[inline]
    fn n_src(&self) -> usize {
        self.src_d.len()
    }

    /// Offset of target position `j`'s column in `mat` (in column mode, loaded first if it is
    /// not cached).
    #[inline]
    fn column(&mut self, ctx: &Ctx, j: usize) -> usize {
        let dt = self.trg_d[j] as usize;
        if !self.columns {
            return dt * self.n_ds;
        }
        let s = self.slot_of[dt];
        if s != NO_SLOT {
            return s as usize * self.n_ds;
        }
        self.load_column(ctx, j)
    }

    /// Column mode: empty the cache and load a tile of columns, the distinct targets of
    /// positions `j, j+1, …` in order of first use until every slot is taken. The tile is
    /// filled row by row, as the full matrix is (each source row's lookups for all of the
    /// tile's targets at once, so the row stays in cache). Returns the offset of `j`'s column.
    #[cold]
    fn load_column(&mut self, ctx: &Ctx, j: usize) -> usize {
        for &dt in &self.owner {
            if dt != NO_SLOT {
                self.slot_of[dt as usize] = NO_SLOT;
            }
        }
        self.owner.clear();
        let slots = self.slots;
        let mut keys = std::mem::take(&mut self.keys);
        keys.clear();
        for &dt in &self.trg_d[j..] {
            if self.owner.len() == slots {
                break;
            }
            if self.slot_of[dt as usize] == NO_SLOT {
                self.slot_of[dt as usize] = self.owner.len() as u32;
                self.owner.push(dt);
                keys.push(self.trg_words[dt as usize]);
            }
        }
        let n_ds = self.n_ds;
        let cols = ctx.table.cols();
        let mut hit = std::mem::take(&mut self.hit);
        for (ds, &e) in self.src_words.iter().enumerate() {
            let r = ctx.table.row(e);
            let prob = &ctx.prob[r.clone()];
            if let Some(d) = ctx.dense.row(e) {
                dense_all(d, prob, &keys, &mut hit);
            } else {
                search_all(ctx.sampled, e, &cols[r], prob, &keys, &mut hit);
            }
            for (s, &k) in hit.iter().enumerate() {
                self.mat[s * n_ds + ds] = prob[k as usize];
            }
        }
        self.hit = hit;
        self.keys = keys;
        self.slot_of[self.trg_d[j] as usize] as usize * n_ds
    }

    /// `prob(src[i], trg[j])` for `col = column(j)` and 0-based source position `i`.
    #[inline]
    fn get(&self, col: usize, i: usize) -> f64 {
        self.mat[col + self.src_d[i] as usize]
    }

    /// `prob(NULL, trg[j])`.
    #[inline]
    fn null(&self, j: usize) -> f64 {
        self.null[self.trg_d[j] as usize]
    }
}

/// Per-task scratch of phase A and decoding (the column buffer and a [`LineLookup`], whose
/// stamp arrays are vocabulary-sized), handed from finished tasks to new ones within one phase
/// instead of being allocated per task (review performance item 7). A phase holds at most one
/// per concurrently running task, as before, and frees them when it ends. No value depends on
/// which scratch a task gets: `LineLookup::prepare` resets everything a line reads.
#[derive(Default)]
struct ScratchPool(Mutex<Vec<TaskScratch>>);

/// One task's column buffer and line lookup.
type TaskScratch = Box<(Vec<f64>, LineLookup)>;

impl ScratchPool {
    fn get(&self, n_vocab: usize, line_cells: usize) -> TaskScratch {
        let reused = self.0.lock().unwrap().pop();
        reused.unwrap_or_else(|| Box::new((Vec::new(), LineLookup::new(n_vocab, line_cells))))
    }

    fn put(&self, s: TaskScratch) {
        self.0.lock().unwrap().push(s);
    }
}

// ------------------------------------------------------------------------------------------
// The trainer

/// Read-only state shared by the tasks of one iteration.
struct Ctx<'a> {
    src: &'a Side,
    trg: &'a Side,
    table: &'a TTable,
    prob: &'a [f64],
    init: bool,
    use_null: bool,
    diag: bool,
    p0: f64,
    lambda: f64,
    reverse: bool,
    pairs: &'a LengthPairs,
    az: &'a [f64],
    a: &'a [f64],
    dense: &'a DenseRows,
    sampled: &'a SampledRows,
}

impl Ctx<'_> {
    #[inline]
    fn prior(&self, line: usize) -> Option<LinePrior<'_>> {
        if !self.diag {
            return None;
        }
        let p = self.pairs.line_pair[line] as usize;
        let (m, n) = self.pairs.pairs[p];
        let az = &self.az[self.pairs.az_off[p]..self.pairs.az_off[p + 1]];
        let table = self.pairs.a_off[p].map(|o| &self.a[o..o + m as usize * n as usize]);
        Some(LinePrior {
            az,
            table,
            m,
            n,
            lambda: self.lambda,
        })
    }

    /// Cell values `probs[0..=n]` of target position `j` and their sum (spec §8). The
    /// translation probabilities come from `lk`, prepared for this line.
    #[inline]
    fn column(
        &self,
        lk: &mut LineLookup,
        j: usize,
        uniform: f64,
        lp: Option<LinePrior<'_>>,
        probs: &mut [f64],
    ) -> f64 {
        let mut sum = 0.0f64;
        let col = lk.column(self, j);
        if self.use_null {
            let a = if self.diag { self.p0 } else { uniform };
            probs[0] = lk.null(j) * a;
            sum += probs[0];
        }
        let n = lk.n_src();
        match lp {
            Some(lp) => {
                for i in 0..n {
                    probs[i + 1] = lk.get(col, i) * lp.get(j, i + 1);
                    sum += probs[i + 1];
                }
            }
            None => {
                for i in 0..n {
                    probs[i + 1] = lk.get(col, i) * uniform;
                    sum += probs[i + 1];
                }
            }
        }
        sum
    }

    fn uniform(&self, n: usize) -> f64 {
        1.0 / (n + usize::from(self.use_null)) as f64
    }

    /// Phase A of one line: column sums into `sums`, statistics into `part`.
    fn e_step_line(
        &self,
        line: usize,
        sums: &mut [f64],
        part: &mut Partial,
        probs: &mut Vec<f64>,
        lk: &mut LineLookup,
        cache: Cached,
    ) {
        let src = self.src.line(line);
        let trg = self.trg.line(line);
        lk.prepare(self, src, trg, cache);
        let (n, m) = (src.len(), trg.len());
        let (n32, m32) = (n as u32, m as u32);
        probs.clear();
        probs.resize(n + 1, 0.0);
        let uniform = self.uniform(n);
        let lp = self.prior(line);
        let mut ll = 0.0f64;
        for (j, out) in sums[..trg.len()].iter_mut().enumerate() {
            let sum = self.column(lk, j, uniform, lp, probs);
            *out = sum;
            if self.use_null {
                part.c0 += probs[0] / sum;
            }
            for (i, &pi) in probs.iter().enumerate().skip(1) {
                let p = pi / sum;
                part.emp_feat += math::feature(j as u32, i as u32, m32, n32) * p;
            }
            ll += sum.ln();
        }
        part.likelihood += ll;
    }

    /// Viterbi decoding of one line (final iteration): appends the output line, returns `ll`.
    fn decode_line(
        &self,
        line: usize,
        scores: Option<f64>,
        out: &mut String,
        probs: &mut Vec<f64>,
        lk: &mut LineLookup,
        cache: Cached,
    ) -> f64 {
        let src = self.src.line(line);
        let trg = self.trg.line(line);
        lk.prepare(self, src, trg, cache);
        let n = src.len();
        probs.clear();
        probs.resize(n + 1, 0.0);
        let uniform = self.uniform(n);
        let lp = self.prior(line);
        let mut ll = 0.0f64;
        let mut first_link = true;
        for j in 0..trg.len() {
            let sum = self.column(lk, j, uniform, lp, probs);
            let (mut best, mut best_p) = if self.use_null {
                (0usize, probs[0])
            } else {
                (usize::MAX, -1.0)
            };
            for (i, &p) in probs.iter().enumerate().skip(1) {
                if p > best_p {
                    best = i;
                    best_p = p;
                }
            }
            if best != 0 && best != usize::MAX {
                if !first_link {
                    out.push(' ');
                }
                first_link = false;
                let s = best - 1;
                let _ = if self.reverse {
                    write!(out, "{j}-{s}")
                } else {
                    write!(out, "{s}-{j}")
                };
            }
            ll += sum.ln();
        }
        if let Some(mean) = scores {
            let lp = math::log_poisson(trg.len() as u32, 0.05 + n as f64 * mean) + ll;
            out.push_str(" ||| ");
            math::write_g(out, lp);
        }
        out.push('\n');
        ll
    }
}

#[allow(clippy::too_many_arguments)]
fn make_ctx<'a>(
    corpus: &'a Corpus,
    opts: &TrainOptions,
    table: &'a TTable,
    prob: &'a [f64],
    init: bool,
    lambda: f64,
    pairs: &'a LengthPairs,
    az: &'a [f64],
    a: &'a [f64],
    dense: &'a DenseRows,
    sampled: &'a SampledRows,
) -> Ctx<'a> {
    let (src, trg) = corpus.sides(opts.reverse);
    Ctx {
        src,
        trg,
        table,
        prob,
        init,
        use_null: opts.use_null,
        diag: opts.favor_diagonal,
        p0: opts.p0,
        lambda,
        reverse: opts.reverse,
        pairs,
        az,
        a,
        dense,
        sampled,
    }
}

/// One phase-A task: a range of lines made of whole fast_align buffers.
#[derive(Clone, Debug)]
struct Group {
    lines: Range<usize>,
    buffer_lines: usize,
}

/// A phase-B task.
enum Job<'a> {
    /// A whole row: accumulate, then normalise into `prob` (the row's probabilities).
    Whole { e: u32, prob: &'a mut [f64] },
    /// Cells `k0..k0+cnt.len()` of a split row; normalised after all parts are done.
    Part {
        e: u32,
        k0: usize,
        prob: &'a [f64],
        cnt: &'a mut [f64],
    },
}

/// A phase-B job walked in line blocks ([`ParallelOptions::block_lines`]): cells
/// `k0..k0+cnt.len()` of row `e`, and where its line list has got to.
struct BlockJob<'a> {
    e: u32,
    k0: usize,
    prob: BlockProb<'a>,
    cnt: &'a mut [f64],
    cur: (usize, usize),
}

/// The row's probabilities: a whole row is normalised into them after its last block; a split
/// row's parts only read them.
enum BlockProb<'a> {
    Whole(&'a mut [f64]),
    Part(&'a [f64]),
}

/// A blocked phase-B job of a [`Schedule`].
#[derive(Clone, Copy)]
enum Spec {
    /// Whole row `e`, adding into the `cnt`-th block count buffer.
    Whole { e: u32, cnt: u32 },
    /// Part number `p` (see [`Schedule::parts`]).
    Part { p: u32 },
}

/// In [`Schedule::jobs`], marks a part number; other codes are whole rows.
const PART: u32 = 1 << 31;

/// Phase B's work plan: which rows are split and where, which rows are walked in line blocks,
/// the job order and the chunks workers claim. It depends only on the table structure, the
/// per-row costs, the target frequencies, the options and the thread count, none of which
/// change between iterations, so it is built once per thread count (review performance item
/// 7). Like every other scheduling choice, it changes no value.
struct Schedule {
    threads: usize,
    /// Split rows in row order: row, offset of its counts in the split buffer, part
    /// boundaries (row-local cell indices, from 0 to the row length).
    split: Vec<(u32, usize, Vec<usize>)>,
    split_len: usize,
    /// Line block size of the blocked jobs (0 disables blocking).
    block_lines: usize,
    /// Blocked whole rows in row order (their count buffers, numbered in this order).
    block_rows: Vec<u32>,
    block_len: usize,
    /// Parts of the split rows, numbered in row then cell order (their count buffers are
    /// numbered the same way): split row index, first row-local cell.
    parts: Vec<(u32, u32)>,
    /// Blocked jobs, largest first.
    bjobs: Vec<Spec>,
    /// The other jobs, largest first: a whole row `e`, or `PART | p`. Four bytes per job, so
    /// keeping the plan between iterations costs next to nothing.
    jobs: Vec<u32>,
    /// Boundaries in `jobs` of the chunks a worker claims at once.
    chunks: Vec<usize>,
}

impl Schedule {
    fn build(
        table: &TTable,
        cost: &[u64],
        target_freq: &[u32],
        dense: &DenseRows,
        popts: &ParallelOptions,
        threads: usize,
    ) -> Self {
        let total: u64 = cost.iter().sum();
        let split_cost = popts.split_cost.unwrap_or(if threads > 1 {
            (total / (2 * threads as u64)).max(1 << 14)
        } else {
            u64::MAX
        });
        let n_rows = table.n_rows();
        let cols = table.cols();
        let mut split: Vec<(u32, usize, Vec<usize>)> = Vec::new();
        let mut split_len = 0;
        for e in 0..n_rows as u32 {
            let r = table.row(e);
            let c = cost[e as usize];
            if r.len() < 2 || c <= split_cost {
                continue;
            }
            let parts = (c.div_ceil(split_cost) as usize).min(r.len()).min(256);
            if parts < 2 {
                continue;
            }
            let w: Vec<u64> = cols[r.clone()]
                .iter()
                .map(|&f| u64::from(target_freq[f as usize]).max(1))
                .collect();
            let tot: u64 = w.iter().sum();
            let mut cuts = vec![0usize];
            let mut acc = 0u64;
            for (k, &x) in w.iter().enumerate() {
                acc += x;
                let q = cuts.len() as u64;
                if acc * parts as u64 >= q * tot && k + 1 < r.len() && cuts.len() < parts {
                    cuts.push(k + 1);
                }
            }
            cuts.push(r.len());
            cuts.dedup();
            if cuts.len() > 2 {
                split.push((e, split_len, cuts));
                split_len += r.len();
            }
        }
        // Rows walked in line blocks (PERF-TODO 9): those with a dense index. One thread has
        // no shared cache to contend for; blocking only costs it (Step 9b: +4 % M-step at 1
        // thread, −6.5 % at 20).
        let block_lines = if threads > 1 { popts.block_lines } else { 0 };
        let blocked = |e: u32| block_lines > 0 && dense.row(e).is_some();
        assert!(
            n_rows <= PART as usize,
            "too many rows for the phase-B plan"
        );
        let mut bjobs: Vec<(u64, Spec)> = Vec::new();
        let mut jobs: Vec<(u64, u32)> = Vec::new();
        let mut parts: Vec<(u32, u32)> = Vec::new();
        let mut block_rows = Vec::new();
        let mut block_len = 0;
        let mut si = split.iter().enumerate().peekable();
        for e in 0..n_rows as u32 {
            let len = table.row(e).len();
            if len == 0 {
                continue;
            }
            let c = cost[e as usize];
            if let Some((s, (_, _, cuts))) = si.next_if(|(_, x)| x.0 == e) {
                let k = (cuts.len() - 1) as u64;
                for w in cuts.windows(2) {
                    let p = parts.len() as u32;
                    parts.push((s as u32, w[0] as u32));
                    if blocked(e) {
                        bjobs.push((c / k, Spec::Part { p }));
                    } else {
                        jobs.push((c / k, PART | p));
                    }
                }
            } else if blocked(e) {
                bjobs.push((
                    c,
                    Spec::Whole {
                        e,
                        cnt: block_rows.len() as u32,
                    },
                ));
                block_rows.push(e);
                block_len += len;
            } else {
                jobs.push((c, e));
            }
        }
        bjobs.sort_by_key(|(c, _)| std::cmp::Reverse(*c));
        jobs.sort_by_key(|(c, _)| std::cmp::Reverse(*c));
        // Chunks of about 1/256 of a worker's share; a job's weight counts a fixed overhead
        // per job and per cell besides its additions, so chunks of tiny rows stay short.
        let weight = |c: u64, code: u32| {
            let len = if code & PART == 0 {
                table.row(code).len() as u64
            } else {
                0
            };
            c + len + 64
        };
        let total_w: u64 = jobs.iter().map(|&(c, code)| weight(c, code)).sum();
        let target = (total_w / (threads as u64 * 256)).max(1);
        let mut chunks = vec![0];
        let mut acc = 0;
        for (k, &(c, code)) in jobs.iter().enumerate() {
            acc += weight(c, code);
            if acc >= target {
                chunks.push(k + 1);
                acc = 0;
            }
        }
        if *chunks.last().unwrap() != jobs.len() {
            chunks.push(jobs.len());
        }
        Schedule {
            threads,
            split,
            split_len,
            block_lines,
            block_rows,
            block_len,
            parts,
            bjobs: bjobs.into_iter().map(|(_, s)| s).collect(),
            jobs: jobs.into_iter().map(|(_, code)| code).collect(),
            chunks,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.split
            .iter()
            .map(|x| x.2.capacity() * 8 + 40)
            .sum::<usize>()
            + self.block_rows.capacity() * 4
            + self.parts.capacity() * 8
            + self.bjobs.capacity() * std::mem::size_of::<Spec>()
            + self.jobs.capacity() * 4
            + self.chunks.capacity() * 8
    }
}

/// Move the `i`-th slice out of `v` (leaving an empty one).
fn take<'a>(v: &mut [&'a mut [f64]], i: usize) -> &'a mut [f64] {
    std::mem::take(&mut v[i])
}

/// Phase-B job trace record (`FA_TRACE_B`, measurement only): job index, row, split part,
/// worker, start and end in seconds since the phase began.
type JobTrace = (usize, u32, bool, usize, f64, f64);

/// Per-worker scratch of phase B.
struct RowScratch {
    /// Dense target id → row-local cell index (only entries of the current row are valid).
    /// (A fused `[prob, count]` array indexed by target id was tried in Step 9b: 8 % slower at
    /// 20 threads.)
    idx: Vec<u32>,
    cnt: Vec<f64>,
    pos: Vec<u32>,
}

/// The deterministic parallel trainer for one direction.
pub struct Trainer<'c> {
    corpus: &'c Corpus,
    opts: TrainOptions,
    popts: ParallelOptions,
    pass: InitialPass,
    /// Probabilities, parallel to `pass.table.cols()` (fast_align's `ttable`).
    prob: Vec<f64>,
    probs_initialized: bool,
    tension: f64,
    pairs: LengthPairs,
    rows: RowLines,
    dense: DenseRows,
    sampled: SampledRows,
    /// Frequency of every id on the target side (weights for splitting hot rows).
    target_freq: Vec<u32>,
    /// Column sums of the current iteration, one per target token.
    sums: Vec<f64>,
    az: Vec<f64>,
    a: Vec<f64>,
    /// Bits of the tension `az` and `a` were built for: an iteration whose tension is
    /// unchanged (no `-o`, or the iteration before the first optimisation) reuses them.
    prior_lambda: Option<u64>,
    /// Phase B's work plan, built on first use for the current thread count.
    schedule: Option<Schedule>,
    /// The opt-in cell-offset cache.
    offsets: Option<OffsetCache>,
    /// `(size_counts index, j)` of every term of the tension gradient, in summation order.
    dlogz_terms: Vec<(u32, u32)>,
    times: PhaseTimes,
    /// Checked during setup, at every iteration and in every phase task.
    cancel: Cancel,
}

impl<'c> Trainer<'c> {
    /// Run the initial pass (§6), writing fast_align's initial-pass log to `log`. The trainer
    /// checks `cancel` during setup, at every iteration and in every phase task (see
    /// [`crate::cancel`]); `&Cancel::default()` cannot be cancelled.
    pub fn new(
        corpus: &'c Corpus,
        opts: TrainOptions,
        popts: ParallelOptions,
        cancel: &Cancel,
        log: &mut dyn Write,
    ) -> io::Result<Self> {
        cancel.check();
        let mut text = b"INITIAL PASS \n".to_vec();
        let mut progress = Progress::default();
        let mut errors = corpus.empty_side_lines().iter().peekable();
        for k in 0..corpus.len() {
            let lc = k + 1;
            progress.tick(lc, &mut text);
            if let Some((_, raw)) = errors.next_if(|(i, _)| *i == k) {
                let _ = writeln!(text, "Error in line {lc}");
                text.extend_from_slice(raw);
                text.push(b'\n');
            }
        }
        let (src, trg) = corpus.sides(opts.reverse);
        let mut times = PhaseTimes::default();
        let t0 = Instant::now();
        let pass = initial_pass_with(
            corpus,
            TableOptions {
                reverse: opts.reverse,
                use_null: opts.use_null,
            },
            cancel,
        );
        progress.finish(&mut text);
        let _ = writeln!(
            text,
            "expected target length = source length * {}",
            G(pass.mean_srclen_multiplier)
        );
        log.write_all(&text)?;
        times.initial_pass = t0.elapsed();

        let t0 = Instant::now();
        let n_rows = pass.table.n_rows();
        let ((pairs, rows), target_freq) = rayon::join(
            || {
                rayon::join(
                    || LengthPairs::build(src, trg, popts.prior_cache_entries, cancel),
                    || RowLines::build(src, trg, n_rows, opts.use_null, cancel),
                )
            },
            || {
                let mut freq = vec![0u32; corpus.vocab.len()];
                for part in trg.tokens().chunks(CANCEL_STRIDE) {
                    cancel.check();
                    for &f in part {
                        freq[f as usize] = freq[f as usize].saturating_add(1);
                    }
                }
                freq
            },
        );
        let mut dlogz_terms = Vec::new();
        if opts.favor_diagonal && opts.optimize_tension {
            for (k, sc) in pass.size_counts.iter().enumerate() {
                for j in 1..=i32::from(sc.m) {
                    dlogz_terms.push((k as u32, j as u32));
                }
            }
        }
        cancel.check();
        let dense = DenseRows::build(&pass.table, &rows.cost, corpus.vocab.len(), &popts);
        let sampled = SampledRows::build(&pass.table, &dense, popts.sample_min_row_len);
        let cells = pass.table.n_cells();
        let offsets = OffsetCache::build(
            src,
            trg,
            corpus.vocab.len(),
            opts.use_null,
            popts.line_cells,
            cells,
            popts.offset_cache_bytes,
            cancel,
        );
        times.setup = t0.elapsed();
        Ok(Trainer {
            corpus,
            tension: opts.tension,
            prob: vec![0.0; cells],
            probs_initialized: false,
            sums: vec![0.0; trg.n_tokens()],
            az: Vec::new(),
            a: Vec::new(),
            prior_lambda: None,
            schedule: None,
            offsets,
            opts,
            popts,
            pass,
            pairs,
            rows,
            dense,
            sampled,
            target_freq,
            dlogz_terms,
            times,
            cancel: cancel.clone(),
        })
    }

    /// Wall time per phase so far (measurement only).
    pub fn phase_times(&self) -> PhaseTimes {
        self.times
    }

    pub fn pass(&self) -> &InitialPass {
        &self.pass
    }

    pub fn table(&self) -> &TTable {
        &self.pass.table
    }

    /// Current probabilities, parallel to `table().cols()` (all 0 before the first M-step).
    pub fn probs(&self) -> &[f64] {
        &self.prob
    }

    pub fn tension(&self) -> f64 {
        self.tension
    }

    pub fn options(&self) -> &TrainOptions {
        &self.opts
    }

    /// Heap bytes held by the trainer (table structure, probabilities, work arrays).
    pub fn memory_bytes(&self) -> usize {
        self.pass.table.memory_bytes()
            + self.prob.capacity() * 8
            + self.sums.capacity() * 8
            + self.az.capacity() * 8
            + self.a.capacity() * 8
            + self.rows.memory_bytes()
            + self.dense.memory_bytes()
            + self.sampled.memory_bytes()
            + self.pairs.line_pair.capacity() * 4
            + self.target_freq.capacity() * 4
            + self.schedule.as_ref().map_or(0, Schedule::memory_bytes)
            + self.offsets.as_ref().map_or(0, OffsetCache::memory_bytes)
    }

    /// The cell-offset cache: bytes held, distinct pairs it covers per pass and distinct
    /// pairs looked up per pass (`None` when it is off or nothing fits).
    pub fn offset_cache_stats(&self) -> Option<(usize, u64, u64)> {
        self.offsets
            .as_ref()
            .map(|c| (c.memory_bytes(), c.pairs_cached, c.pairs_total))
    }

    /// Run all iterations (§7): statistics to `log`, alignments of the final iteration to `out`.
    pub fn train(&mut self, log: &mut dyn Write, out: &mut dyn Write) -> io::Result<()> {
        let iters = self.opts.iterations.max(0) as usize;
        for iter in 0..iters {
            self.iteration(iter, iter + 1 == iters, log, out)?;
        }
        Ok(())
    }

    fn buffer_lines(&self) -> usize {
        let b = self.opts.buffer_size.max(1);
        usize::try_from(b)
            .unwrap_or(usize::MAX)
            .min(self.corpus.len().max(1))
    }

    /// Phase-A tasks: groups of whole buffers of about `group_lines` lines.
    fn groups(&self) -> Vec<Group> {
        let n = self.corpus.len();
        let b = self.buffer_lines();
        let per_group = (self.popts.group_lines / b).max(1);
        let step = b.saturating_mul(per_group);
        let mut groups = Vec::new();
        let mut s = 0;
        while s < n {
            let e = s.saturating_add(step).min(n);
            groups.push(Group {
                lines: s..e,
                buffer_lines: b,
            });
            s = e;
        }
        groups
    }

    /// The iteration's log text before the statistics: progress dots and the empty-side
    /// messages, which fast_align prints when a buffer is processed (§3, §15.4).
    fn iteration_log(&self, text: &mut Vec<u8>) {
        let n = self.corpus.len();
        let b = self.buffer_lines();
        let mut progress = Progress::default();
        let mut errors = self.corpus.empty_side_lines().iter().peekable();
        let mut start = 0;
        while start < n {
            let end = start.saturating_add(b).min(n);
            for lc in start + 1..=end {
                progress.tick(lc, text);
            }
            while let Some((_, raw)) = errors.next_if(|(i, _)| *i < end) {
                let _ = writeln!(text, "Error in line {end}");
                text.extend_from_slice(raw);
                text.push(b'\n');
            }
            start = end;
        }
        progress.finish(text);
    }

    /// Fill the per-iteration prior cache for the current tension (`-d` only). The tables are
    /// a pure function of the tension (`p0` and the length pairs are fixed), so they are kept
    /// while the tension's bits are unchanged.
    fn build_prior(&mut self) {
        if !self.opts.favor_diagonal {
            return;
        }
        let lambda = self.tension;
        if self.prior_lambda == Some(lambda.to_bits()) {
            return;
        }
        self.prior_lambda = Some(lambda.to_bits());
        let pnn = 1.0 - self.opts.p0;
        let pairs = &self.pairs;
        self.az.clear();
        self.az.resize(*pairs.az_off.last().unwrap(), 0.0);
        self.a.clear();
        self.a.resize(pairs.a_len, 0.0);
        let mut az_rest = &mut self.az[..];
        let mut az_parts = Vec::with_capacity(pairs.pairs.len());
        for &(m, _) in &pairs.pairs {
            let (h, t) = az_rest.split_at_mut(m as usize);
            az_parts.push(h);
            az_rest = t;
        }
        az_parts
            .par_iter_mut()
            .zip(pairs.pairs.par_iter())
            .with_min_len(16)
            .for_each(|(az, &(m, n))| {
                for (j, z) in az.iter_mut().enumerate() {
                    *z = math::compute_z(j as u32 + 1, m, n, lambda) / pnn;
                }
            });
        // Full tables of the selected pairs, in offset order.
        let mut sel: Vec<usize> = (0..pairs.pairs.len())
            .filter(|&p| pairs.a_off[p].is_some())
            .collect();
        sel.sort_by_key(|&p| pairs.a_off[p]);
        let mut a_rest = &mut self.a[..];
        let mut a_parts = Vec::with_capacity(sel.len());
        for &p in &sel {
            let (m, n) = pairs.pairs[p];
            let (h, t) = a_rest.split_at_mut(m as usize * n as usize);
            a_parts.push((p, h));
            a_rest = t;
        }
        let az = &self.az;
        a_parts.into_par_iter().for_each(|(p, tab)| {
            let (m, n) = pairs.pairs[p];
            let z = &az[pairs.az_off[p]..pairs.az_off[p + 1]];
            for (i, col) in tab.chunks_exact_mut(m as usize).enumerate() {
                for (j, v) in col.iter_mut().enumerate() {
                    *v = math::unnormalized_prob(j as u32 + 1, i as u32 + 1, m, n, lambda) / z[j];
                }
            }
        });
    }

    /// One iteration (§7): E-step, statistics, then (unless `final_iter`) tension optimisation;
    /// the M-step is fused into phase B. In the final iteration the alignments are written to
    /// `out`.
    pub fn iteration(
        &mut self,
        iter: usize,
        final_iter: bool,
        log: &mut dyn Write,
        out: &mut dyn Write,
    ) -> io::Result<IterationStats> {
        self.cancel.check();
        writeln!(
            log,
            "ITERATION {}{}",
            iter + 1,
            if final_iter { " (FINAL)" } else { "" }
        )?;
        let t0 = Instant::now();
        self.build_prior();
        self.times.prior += t0.elapsed();
        let groups = self.groups();
        let tension = self.tension;
        let partials = if final_iter {
            let t0 = Instant::now();
            let partials = self.decode(&groups, out)?;
            self.times.decode += t0.elapsed();
            partials
        } else {
            let t0 = Instant::now();
            let partials = self.phase_a(&groups);
            let t1 = Instant::now();
            self.phase_b();
            self.times.e_step += t1 - t0;
            self.times.m_step += t1.elapsed();
            self.times.train_iterations += 1;
            self.probs_initialized = true;
            partials
        };
        let (mut likelihood, mut c0, mut emp_feat) = (0.0f64, 0.0f64, 0.0f64);
        for p in &partials {
            likelihood += p.likelihood;
            c0 += p.c0;
            emp_feat += p.emp_feat;
        }

        let mut text = Vec::new();
        self.iteration_log(&mut text);
        let t = self.pass.n_target_tokens;
        let base2 = likelihood / LN_2;
        let emp = emp_feat / t;
        let _ = writeln!(text, "  log_e likelihood: {}", G(likelihood));
        let _ = writeln!(text, "  log_2 likelihood: {}", G(base2));
        let _ = writeln!(text, "     cross entropy: {}", G(-base2 / t));
        let _ = writeln!(
            text,
            "        perplexity: {}",
            G(math::pow(2.0, -base2 / t))
        );
        let _ = writeln!(text, "      posterior p0: {}", G(c0 / t));
        let _ = writeln!(text, " posterior al-feat: {}", G(emp));
        let _ = writeln!(text, "       size counts: {}", self.pass.size_counts.len());
        if !final_iter && self.opts.favor_diagonal && self.opts.optimize_tension && iter > 0 {
            let t0 = Instant::now();
            self.optimize_tension(emp, &mut text);
            self.times.tension += t0.elapsed();
        }
        log.write_all(&text)?;
        Ok(IterationStats {
            likelihood,
            c0,
            emp_feat: emp,
            tension,
            tension_after: self.tension,
        })
    }

    fn ctx(&self) -> Ctx<'_> {
        make_ctx(
            self.corpus,
            &self.opts,
            &self.pass.table,
            &self.prob,
            self.probs_initialized,
            self.tension,
            &self.pairs,
            &self.az,
            &self.a,
            &self.dense,
            &self.sampled,
        )
    }

    /// Phase A: column sums and per-buffer statistics, in parallel over groups.
    fn phase_a(&mut self, groups: &[Group]) -> Vec<Partial> {
        let mut sums = std::mem::take(&mut self.sums);
        let mut cache = self.offsets.take();
        let fill = self.probs_initialized && cache.as_ref().is_some_and(|c| !c.filled);
        let cancel = &self.cancel;
        let partials = {
            let ctx = self.ctx();
            let n_vocab = self.corpus.vocab.len();
            let line_cells = self.popts.line_cells;
            let pool = ScratchPool::default();
            let offsets = ctx.trg.offsets();
            let mut rest = &mut sums[..];
            let mut slices = Vec::with_capacity(groups.len());
            for g in groups {
                let len = offsets[g.lines.end] - offsets[g.lines.start];
                let (h, t) = rest.split_at_mut(len);
                slices.push(h);
                rest = t;
            }
            // Filling: each group's part of the offset cache, carved in line order.
            let mut fills: Vec<Option<(usize, &mut [u32])>> = Vec::with_capacity(groups.len());
            let (cache_use, start): (Option<&OffsetCache>, &[u64]) = if fill {
                let c = cache.as_mut().expect("filling needs the cache");
                let cov = c.covered();
                let mut rest = &mut c.off[..];
                for g in groups {
                    let (a, b) = (g.lines.start.min(cov), g.lines.end.min(cov));
                    let (h, t) = rest.split_at_mut((c.start[b] - c.start[a]) as usize);
                    rest = t;
                    fills.push((a < b).then_some((c.start[a] as usize, h)));
                }
                (None, &c.start[..])
            } else {
                (cache.as_ref().filter(|c| c.filled), &[])
            };
            fills.resize_with(groups.len(), || None);
            let per_group: Vec<Vec<Partial>> = slices
                .into_par_iter()
                .zip(fills)
                .zip(groups.par_iter())
                .with_max_len(1)
                .panic_fuse()
                .map(|((sums, mut fill), g)| {
                    cancel.check();
                    let base = offsets[g.lines.start];
                    let mut scratch = pool.get(n_vocab, line_cells);
                    let (probs, lk) = &mut *scratch;
                    let mut parts = Vec::new();
                    let mut s = g.lines.start;
                    while s < g.lines.end {
                        let e = (s + g.buffer_lines).min(g.lines.end);
                        let mut part = Partial::default();
                        for line in s..e {
                            let r = offsets[line] - base..offsets[line + 1] - base;
                            let cached = match (&mut fill, cache_use) {
                                (Some((o, f)), _) if line + 1 < start.len() => {
                                    let a = start[line] as usize - *o;
                                    let b = start[line + 1] as usize - *o;
                                    Cached::Fill(&mut f[a..b])
                                }
                                (_, Some(c)) => c.line(line),
                                _ => Cached::No,
                            };
                            ctx.e_step_line(line, &mut sums[r], &mut part, probs, lk, cached);
                        }
                        parts.push(part);
                        s = e;
                    }
                    pool.put(scratch);
                    parts
                })
                .collect();
            per_group.into_iter().flatten().collect()
        };
        if let Some(c) = &mut cache {
            c.filled |= fill;
        }
        self.offsets = cache;
        self.sums = sums;
        partials
    }

    /// Final iteration: Viterbi decoding in parallel, output in corpus order.
    fn decode(&self, groups: &[Group], out: &mut dyn Write) -> io::Result<Vec<Partial>> {
        let cancel = &self.cancel;
        let ctx = self.ctx();
        let n_vocab = self.corpus.vocab.len();
        let line_cells = self.popts.line_cells;
        let scores = self
            .opts
            .print_scores
            .then_some(self.pass.mean_srclen_multiplier);
        let window = 4 * rayon::current_num_threads();
        let pool = ScratchPool::default();
        let cache = self.offsets.as_ref().filter(|c| c.filled);
        let mut partials = Vec::new();
        for win in groups.chunks(window) {
            let res: Vec<(String, Vec<Partial>)> = win
                .par_iter()
                .with_max_len(1)
                .panic_fuse()
                .map(|g| {
                    cancel.check();
                    let mut scratch = pool.get(n_vocab, line_cells);
                    let (probs, lk) = &mut *scratch;
                    let mut text = String::new();
                    let mut parts = Vec::new();
                    let mut s = g.lines.start;
                    while s < g.lines.end {
                        let e = (s + g.buffer_lines).min(g.lines.end);
                        let mut part = Partial::default();
                        for line in s..e {
                            let cached = cache.map_or(Cached::No, |c| c.line(line));
                            part.likelihood +=
                                ctx.decode_line(line, scores, &mut text, probs, lk, cached);
                        }
                        parts.push(part);
                        s = e;
                    }
                    pool.put(scratch);
                    (text, parts)
                })
                .collect();
            for (text, parts) in res {
                out.write_all(text.as_bytes())?;
                partials.extend(parts);
            }
        }
        Ok(partials)
    }

    /// Phase B: per-row count accumulation in corpus order, fused with the M-step.
    fn phase_b(&mut self) {
        let t_setup = Instant::now();
        let trace_on = std::env::var_os("FA_TRACE_B").is_some();
        let threads = rayon::current_num_threads();
        let sched = match self.schedule.take() {
            Some(s) if s.threads == threads => s,
            _ => Schedule::build(
                &self.pass.table,
                &self.rows.cost,
                &self.target_freq,
                &self.dense,
                &self.popts,
                threads,
            ),
        };
        let table = &self.pass.table;
        let row_ptr = table.row_ptr();
        let n_rows = table.n_rows();
        let cols = table.cols();
        let mut split_cnt = vec![0.0f64; sched.split_len];
        let mut block_cnt = vec![0.0f64; sched.block_len];

        {
            // Phase B reads probabilities only through each job's own row.
            let ctx = make_ctx(
                self.corpus,
                &self.opts,
                table,
                &[],
                self.probs_initialized,
                self.tension,
                &self.pairs,
                &self.az,
                &self.a,
                &self.dense,
                &self.sampled,
            );
            // Row probabilities: whole rows are written by their job, split rows only read.
            let mut whole: Vec<&mut [f64]> = Vec::with_capacity(n_rows);
            let mut split_prob: Vec<&[f64]> = Vec::with_capacity(sched.split.len());
            let mut rest = &mut self.prob[..];
            let mut si = sched.split.iter().peekable();
            for e in 0..n_rows {
                let (h, t) = rest.split_at_mut(row_ptr[e + 1] - row_ptr[e]);
                rest = t;
                if si.next_if(|x| x.0 as usize == e).is_some() {
                    split_prob.push(h);
                    whole.push(&mut []);
                } else {
                    whole.push(h);
                }
            }
            // Count buffers of the split rows' parts and of the blocked whole rows, in the
            // schedule's numbering.
            let mut parts: Vec<&mut [f64]> = Vec::new();
            let mut rest = &mut split_cnt[..];
            for (_, _, cuts) in &sched.split {
                for w in cuts.windows(2) {
                    let (h, t) = rest.split_at_mut(w[1] - w[0]);
                    parts.push(h);
                    rest = t;
                }
            }
            let mut bcnt: Vec<&mut [f64]> = Vec::with_capacity(sched.block_rows.len());
            let mut rest = &mut block_cnt[..];
            for &e in &sched.block_rows {
                let (h, t) = rest.split_at_mut(table.row(e).len());
                bcnt.push(h);
                rest = t;
            }
            let bjobs: Vec<BlockJob> = sched
                .bjobs
                .iter()
                .map(|s| match *s {
                    Spec::Whole { e, cnt } => BlockJob {
                        e,
                        k0: 0,
                        prob: BlockProb::Whole(take(&mut whole, e as usize)),
                        cnt: take(&mut bcnt, cnt as usize),
                        cur: (0, 0),
                    },
                    Spec::Part { p } => {
                        let (s, k0) = sched.parts[p as usize];
                        BlockJob {
                            e: sched.split[s as usize].0,
                            k0: k0 as usize,
                            prob: BlockProb::Part(split_prob[s as usize]),
                            cnt: take(&mut parts, p as usize),
                            cur: (0, 0),
                        }
                    }
                })
                .collect();
            if trace_on {
                eprintln!("phase_b setup {:.4} s", t_setup.elapsed().as_secs_f64());
            }
            let t_blocked = Instant::now();
            run_blocked(
                &ctx,
                &self.rows,
                &self.sums,
                self.probs_initialized,
                &self.opts,
                bjobs,
                sched.block_lines,
                &self.cancel,
            );
            if trace_on {
                eprintln!(
                    "phase_b blocked rows {:.3} s",
                    t_blocked.elapsed().as_secs_f64()
                );
            }
            // The other jobs, largest first, claimed a chunk at a time: a longest-processing-
            // time schedule without a lock and an atomic per (often tiny) job.
            let mut specs = sched.jobs.iter().map(|&code| {
                if code & PART == 0 {
                    Job::Whole {
                        e: code,
                        prob: take(&mut whole, code as usize),
                    }
                } else {
                    let p = (code & !PART) as usize;
                    let (s, k0) = sched.parts[p];
                    Job::Part {
                        e: sched.split[s as usize].0,
                        k0: k0 as usize,
                        prob: split_prob[s as usize],
                        cnt: take(&mut parts, p),
                    }
                }
            });
            let chunks: Vec<Mutex<Vec<Job>>> = sched
                .chunks
                .windows(2)
                .map(|w| Mutex::new(specs.by_ref().take(w[1] - w[0]).collect()))
                .collect();
            let trace: Mutex<Vec<JobTrace>> = Mutex::new(Vec::new());
            let t_phase = Instant::now();
            let next = AtomicUsize::new(0);
            let n_vocab = self.corpus.vocab.len();
            let rows = &self.rows;
            let opts = &self.opts;
            let sums = &self.sums[..];
            let prob_init = self.probs_initialized;
            let cancel = &self.cancel;
            let first_job = &sched.chunks;
            (0..threads.min(chunks.len()).max(1))
                .into_par_iter()
                .for_each(|worker| {
                    let mut s = RowScratch {
                        idx: vec![0; n_vocab],
                        cnt: Vec::new(),
                        pos: Vec::new(),
                    };
                    loop {
                        let c = next.fetch_add(1, Ordering::Relaxed);
                        let Some(slot) = chunks.get(c) else { break };
                        cancel.check();
                        let jobs = std::mem::take(&mut *slot.lock().unwrap());
                        for (n, job) in jobs.into_iter().enumerate() {
                            let t_job = trace_on.then(|| t_phase.elapsed().as_secs_f64());
                            let (je, jpart) = match &job {
                                Job::Whole { e, .. } => (*e, false),
                                Job::Part { e, .. } => (*e, true),
                            };
                            match job {
                                Job::Whole { e, prob } => {
                                    let row_cols = &cols[table.row(e)];
                                    let mut cnt = std::mem::take(&mut s.cnt);
                                    cnt.clear();
                                    cnt.resize(row_cols.len(), 0.0);
                                    accumulate(
                                        &ctx, rows, sums, e, row_cols, true, prob, prob_init,
                                        &mut cnt, &mut s,
                                    );
                                    normalise(&cnt, table.row_order(e), prob, opts);
                                    s.cnt = cnt;
                                }
                                Job::Part { e, k0, prob, cnt } => {
                                    let row_cols = &cols[table.row(e)][k0..k0 + cnt.len()];
                                    accumulate(
                                        &ctx,
                                        rows,
                                        sums,
                                        e,
                                        row_cols,
                                        false,
                                        &prob[k0..],
                                        prob_init,
                                        cnt,
                                        &mut s,
                                    );
                                }
                            }
                            if let Some(t0) = t_job {
                                let t1 = t_phase.elapsed().as_secs_f64();
                                let k = first_job[c] + n;
                                trace.lock().unwrap().push((k, je, jpart, worker, t0, t1));
                            }
                        }
                    }
                });
            if trace_on {
                let cost = &self.rows.cost;
                let job_cost: Vec<u64> = sched
                    .jobs
                    .iter()
                    .map(|&code| {
                        if code & PART == 0 {
                            cost[code as usize]
                        } else {
                            let (s, _) = sched.parts[(code & !PART) as usize];
                            let (e, _, cuts) = &sched.split[s as usize];
                            cost[*e as usize] / (cuts.len() - 1) as u64
                        }
                    })
                    .collect();
                let wall = t_phase.elapsed().as_secs_f64();
                let mut t = trace.into_inner().unwrap();
                let busy: f64 = t.iter().map(|x| x.5 - x.4).sum();
                let total_cost: u64 = job_cost.iter().sum();
                let last_start = t.iter().map(|x| x.4).fold(0.0, f64::max);
                eprintln!(
                    "phase_b wall {wall:.3} s, jobs {}, chunks {}, busy {busy:.2} worker-s, last job start {last_start:.3}",
                    t.len(),
                    chunks.len()
                );
                let mut per_worker = vec![0.0f64; threads];
                for x in &t {
                    per_worker[x.3] += x.5 - x.4;
                }
                eprintln!("  per worker busy: {per_worker:.2?}");
                t.sort_by(|a, b| (b.5 - b.4).total_cmp(&(a.5 - a.4)));
                for x in t.iter().take(8) {
                    eprintln!(
                        "  job {} row {} {} len {} cost {:.1}% worker {} {:.3}-{:.3} ({:.3} s)",
                        x.0,
                        x.1,
                        if x.2 { "part" } else { "whole" },
                        table.row(x.1).len(),
                        100.0 * job_cost[x.0] as f64 / total_cost as f64,
                        x.3,
                        x.4,
                        x.5,
                        x.5 - x.4
                    );
                }
            }
        }
        // Normalise the split rows.
        let opts = &self.opts;
        let prob = &mut self.prob;
        let mut rest = &mut prob[..];
        let mut consumed = 0;
        let mut targets = Vec::with_capacity(sched.split.len());
        for (e, off, _) in &sched.split {
            let r = table.row(*e);
            let (_, t) = rest.split_at_mut(r.start - consumed);
            let (h, t) = t.split_at_mut(r.len());
            targets.push((*e, &split_cnt[*off..*off + r.len()], h));
            rest = t;
            consumed = r.end;
        }
        targets.into_par_iter().for_each(|(e, cnt, prob)| {
            normalise(cnt, table.row_order(e), prob, opts);
        });
        self.schedule = Some(sched);
    }

    /// Eight gradient steps on λ (§10.2); terms in parallel, summed in fast_align's order.
    fn optimize_tension(&mut self, emp: f64, text: &mut Vec<u8>) {
        let t = self.pass.n_target_tokens;
        let sc = &self.pass.size_counts;
        let mut terms = vec![0.0f64; self.dlogz_terms.len()];
        for ii in 1..=8 {
            let lambda = self.tension;
            terms
                .par_iter_mut()
                .zip(self.dlogz_terms.par_iter())
                .with_min_len(1024)
                .for_each(|(x, &(k, j))| {
                    let s = &sc[k as usize];
                    let (mu, nu) = (i32::from(s.m) as u32, i32::from(s.n) as u32);
                    *x = f64::from(s.count) * math::compute_dlogz(j, mu, nu, lambda);
                });
            let mut mod_feat = 0.0f64;
            for &x in &terms {
                mod_feat += x;
            }
            mod_feat /= t;
            let _ = writeln!(
                text,
                "  {ii}  model al-feat: {} (tension={})",
                G(mod_feat),
                G(lambda)
            );
            self.tension = (lambda + (emp - mod_feat) * 20.0).clamp(0.1, 14.0);
        }
        let _ = writeln!(text, "     final tension: {}", G(self.tension));
    }

    /// Write fast_align's `-p` table (§15.3), formatted in parallel, written in row order.
    pub fn write_table(
        &self,
        vocab: &Vocab,
        beam_threshold: f64,
        w: &mut dyn Write,
    ) -> io::Result<()> {
        write_text_table(
            &self.pass.table,
            &self.prob,
            vocab,
            beam_threshold,
            &self.cancel,
            w,
        )
    }

    /// Give up the corpus borrow: the table structure and the final probabilities.
    pub fn into_parts(self) -> TrainerParts {
        TrainerParts {
            tension: self.tension,
            mean_srclen_multiplier: self.pass.mean_srclen_multiplier,
            table: self.pass.table,
            prob: self.prob,
            opts: self.opts,
        }
    }
}

/// What remains of a [`Trainer`] once training is over.
pub struct TrainerParts {
    pub table: TTable,
    pub prob: Vec<f64>,
    /// Tension after the last optimisation (the value `-T` needs to reproduce the alignments).
    pub tension: f64,
    pub mean_srclen_multiplier: f64,
    pub opts: TrainOptions,
}

/// Write fast_align's `-p` table (§15.3) for a table and probabilities parallel to its cells,
/// formatted in parallel row chunks and written in row order.
pub fn write_text_table(
    table: &TTable,
    prob: &[f64],
    vocab: &Vocab,
    beam_threshold: f64,
    cancel: &Cancel,
    w: &mut dyn Write,
) -> io::Result<()> {
    let row_ptr = table.row_ptr();
    let n_rows = table.n_rows();
    // Chunks of rows with about 64k cells each.
    let mut chunks = Vec::new();
    let mut s = 0;
    while s < n_rows {
        let mut e = s + 1;
        while e < n_rows && row_ptr[e + 1] - row_ptr[s] <= 1 << 16 {
            e += 1;
        }
        chunks.push(s..e);
        s = e;
    }
    let window = 4 * rayon::current_num_threads();
    for win in chunks.chunks(window) {
        let texts: Vec<Vec<u8>> = win
            .par_iter()
            .panic_fuse()
            .map(|rows| {
                cancel.check();
                let mut buf = Vec::new();
                let mut num = String::new();
                for e in rows.clone() {
                    let e = e as u32;
                    write_table_row(table, prob, vocab, e, beam_threshold, &mut buf, &mut num);
                }
                buf
            })
            .collect();
        for t in texts {
            w.write_all(&t)?;
        }
    }
    Ok(())
}

fn write_table_row(
    table: &TTable,
    prob: &[f64],
    vocab: &Vocab,
    e: u32,
    beam_threshold: f64,
    buf: &mut Vec<u8>,
    num: &mut String,
) {
    let cols = table.cols();
    let mut max_p = -1.0f64;
    for c in table.iter_hash_order(e) {
        if prob[c] > max_p {
            max_p = prob[c];
        }
    }
    let threshold = -max_p.ln() * beam_threshold;
    for c in table.iter_hash_order(e) {
        let lp = prob[c].ln();
        if lp >= threshold {
            buf.extend_from_slice(vocab.word(e));
            buf.push(b'\t');
            buf.extend_from_slice(vocab.word(cols[c]));
            buf.push(b'\t');
            num.clear();
            math::write_g(num, lp);
            buf.extend_from_slice(num.as_bytes());
            buf.push(b'\n');
        }
    }
}

/// Phase B of the rows with a dense index, in line blocks: per block, every row's lines of
/// that block, all workers on the same block; then the whole rows are normalised. Each
/// cell still receives its additions in corpus order (blocks in order, lines in order
/// within a block), so nothing changes but the memory access pattern.
#[allow(clippy::too_many_arguments)]
fn run_blocked(
    ctx: &Ctx,
    rows: &RowLines,
    sums: &[f64],
    prob_init: bool,
    opts: &TrainOptions,
    bjobs: Vec<BlockJob>,
    block_lines: usize,
    cancel: &Cancel,
) {
    if bjobs.is_empty() {
        return;
    }
    let threads = rayon::current_num_threads();
    let table = ctx.table;
    let cols = table.cols();
    let n_lines = ctx.src.offsets().len() - 1;
    // Already largest first (`Schedule::bjobs`).
    let bjobs: Vec<Mutex<BlockJob>> = bjobs.into_iter().map(Mutex::new).collect();
    for b0 in (0..n_lines).step_by(block_lines.max(1)) {
        cancel.check();
        let end = (b0 + block_lines).min(n_lines) as u32;
        let next = AtomicUsize::new(0);
        (0..threads.min(bjobs.len())).into_par_iter().for_each(|_| {
            let mut pos = Vec::new();
            loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(slot) = bjobs.get(k) else { break };
                let mut guard = slot.lock().unwrap();
                let BlockJob {
                    e,
                    k0,
                    prob,
                    cnt,
                    cur,
                } = &mut *guard;
                let (e, k0) = (*e, *k0);
                let prob: &[f64] = match prob {
                    BlockProb::Whole(p) => p,
                    BlockProb::Part(p) => p,
                };
                let r = table.row(e);
                let row_cols = &cols[r.start + k0..r.start + k0 + cnt.len()];
                let lim = (
                    row_cols[0],
                    row_cols[row_cols.len() - 1],
                    cnt.len() == r.len(),
                );
                let idx = (ctx.dense.row(e).expect("blocked rows are dense"), k0 as u32);
                if e == NULL_ID && ctx.use_null {
                    accumulate_lines(
                        ctx,
                        rows.eps_in_src,
                        sums,
                        e,
                        (b0 as u32..end).map(|k| (k, 0)),
                        lim,
                        &prob[k0..],
                        prob_init,
                        cnt,
                        idx,
                        &mut pos,
                    );
                } else {
                    accumulate_lines(
                        ctx,
                        true,
                        sums,
                        e,
                        rows.lines_before(e, cur, end),
                        lim,
                        &prob[k0..],
                        prob_init,
                        cnt,
                        idx,
                        &mut pos,
                    );
                }
            }
        });
    }
    bjobs.into_par_iter().for_each(|m| {
        let j = m.into_inner().unwrap();
        if let BlockProb::Whole(prob) = j.prob {
            normalise(j.cnt, table.row_order(j.e), prob, opts);
        }
    });
}

/// Phase B for the cells `row_cols` of row `e` (all of them if `full`): add each posterior
/// to its count in corpus order (line, `j`, NULL, `i`). `prob` and `cnt` are indexed like
/// `row_cols`.
#[allow(clippy::too_many_arguments)]
fn accumulate(
    ctx: &Ctx,
    rows: &RowLines,
    sums: &[f64],
    e: u32,
    row_cols: &[u32],
    full: bool,
    prob: &[f64],
    prob_init: bool,
    cnt: &mut [f64],
    s: &mut RowScratch,
) {
    for (k, &f) in row_cols.iter().enumerate() {
        s.idx[f as usize] = k as u32;
    }
    let (lo, hi) = (row_cols[0], row_cols[row_cols.len() - 1]);
    let null_row = e == NULL_ID && ctx.use_null;
    if null_row {
        accumulate_lines(
            ctx,
            rows.eps_in_src,
            sums,
            e,
            (0..ctx.src.offsets().len() as u32 - 1).map(|k| (k, 0)),
            (lo, hi, full),
            prob,
            prob_init,
            cnt,
            (&s.idx, 0),
            &mut s.pos,
        );
    } else {
        accumulate_lines(
            ctx,
            true,
            sums,
            e,
            rows.lines(e),
            (lo, hi, full),
            prob,
            prob_init,
            cnt,
            (&s.idx, 0),
            &mut s.pos,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn accumulate_lines<I: Iterator<Item = (u32, u16)>>(
    ctx: &Ctx,
    scan_src: bool,
    sums: &[f64],
    e: u32,
    lines: I,
    (lo, hi, full): (u32, u32, bool),
    prob: &[f64],
    prob_init: bool,
    cnt: &mut [f64],
    (idx, k0): (&[u32], u32),
    pos: &mut Vec<u32>,
) {
    let null_row = e == NULL_ID && ctx.use_null;
    let offsets = ctx.trg.offsets();
    for (line, first) in Lookahead::new(ctx, sums, lines) {
        let line = line as usize;
        let trg = ctx.trg.line(line);
        if trg.is_empty() {
            continue;
        }
        let n = ctx.src.line_len(line);
        pos.clear();
        if first != 0 {
            pos.push(u32::from(first));
        } else if scan_src {
            for (i, &w) in ctx.src.line(line).iter().enumerate() {
                if w == e {
                    pos.push(i as u32 + 1);
                }
            }
        }
        let sums = &sums[offsets[line]..offsets[line + 1]];
        let uniform = 1.0 / (n + usize::from(ctx.use_null)) as f64;
        let a0 = if ctx.diag { ctx.p0 } else { uniform };
        let lp = ctx.prior(line);
        for (j, (&f, &sum)) in trg.iter().zip(sums).enumerate() {
            if !full && (f < lo || f > hi) {
                continue;
            }
            let k = (idx[f as usize] - k0) as usize;
            let pv = if prob_init {
                prob[k]
            } else {
                UNINITIALISED_PROB
            };
            let c = &mut cnt[k];
            if null_row {
                *c += pv * a0 / sum;
            }
            match lp {
                Some(lp) => {
                    for &i in pos.iter() {
                        *c += pv * lp.get(j, i as usize) / sum;
                    }
                }
                None => {
                    for _ in pos.iter() {
                        *c += pv * uniform / sum;
                    }
                }
            }
        }
    }
}

/// Lines ahead of the current one whose data phase B prefetches (two stages of this many).
const AHEAD: usize = 8;

/// Phase B's walk over a row's lines with prefetching: `2 × AHEAD` lines ahead the line's
/// offsets and length pair are prefetched, `AHEAD` lines ahead its target tokens and column
/// sums (their addresses need the offsets). Yields the lines unchanged and in order.
struct Lookahead<'a, I> {
    trg_off: &'a [usize],
    src_off: &'a [usize],
    line_pair: &'a [u32],
    tokens: &'a [u32],
    sums: &'a [f64],
    /// Prior tables (`-d` only; empty otherwise).
    pairs: &'a LengthPairs,
    a: &'a [f64],
    inner: I,
    ring: [(u32, u16); 2 * AHEAD],
    head: usize,
    len: usize,
}

impl<'a, I: Iterator<Item = (u32, u16)>> Lookahead<'a, I> {
    fn new(ctx: &Ctx<'a>, sums: &'a [f64], inner: I) -> Self {
        let mut l = Lookahead {
            trg_off: ctx.trg.offsets(),
            src_off: ctx.src.offsets(),
            line_pair: &ctx.pairs.line_pair,
            tokens: ctx.trg.tokens(),
            sums,
            pairs: ctx.pairs,
            a: if ctx.diag { ctx.a } else { &[] },
            inner,
            ring: [(0, 0); 2 * AHEAD],
            head: 0,
            len: 0,
        };
        while l.len < 2 * AHEAD {
            let Some(x) = l.inner.next() else { break };
            l.push(x);
        }
        for d in 0..l.len.min(AHEAD) {
            l.near(l.ring[(l.head + d) % (2 * AHEAD)]);
        }
        l
    }

    #[inline]
    fn push(&mut self, x: (u32, u16)) {
        let line = x.0 as usize;
        prefetch(&self.trg_off[line]);
        prefetch(&self.src_off[line]);
        if let Some(p) = self.line_pair.get(line) {
            prefetch(p);
        }
        self.ring[(self.head + self.len) % (2 * AHEAD)] = x;
        self.len += 1;
    }

    #[inline]
    fn near(&self, (line, first): (u32, u16)) {
        let line = line as usize;
        if first != 0 && !self.a.is_empty() {
            let p = self.pairs.line_pair[line] as usize;
            if let Some(o) = self.pairs.a_off[p] {
                let m = self.pairs.pairs[p].0 as usize;
                if let Some(x) = self.a.get(o + (usize::from(first) - 1) * m) {
                    prefetch(x);
                }
            }
        }
        let o = self.trg_off[line];
        if let Some(t) = self.tokens.get(o) {
            prefetch(t);
        }
        if let Some(t) = self.sums.get(o) {
            prefetch(t);
        }
    }
}

impl<I: Iterator<Item = (u32, u16)>> Iterator for Lookahead<'_, I> {
    type Item = (u32, u16);

    #[inline]
    fn next(&mut self) -> Option<(u32, u16)> {
        if self.len == 0 {
            return None;
        }
        let x = self.ring[self.head];
        self.head = (self.head + 1) % (2 * AHEAD);
        self.len -= 1;
        if let Some(y) = self.inner.next() {
            self.push(y);
        }
        if self.len >= AHEAD {
            self.near(self.ring[(self.head + AHEAD - 1) % (2 * AHEAD)]);
        }
        Some(x)
    }
}

/// `Normalize` / `NormalizeVB` of one row (§11): total in hash order.
fn normalise(cnt: &[f64], order: RowOrder, prob: &mut [f64], opts: &TrainOptions) {
    match order {
        RowOrder::Short(o) => normalise_in(cnt, o, prob, opts),
        RowOrder::Long(o) => normalise_in(cnt, o, prob, opts),
    }
}

fn normalise_in<K: Copy + Into<u64>>(
    cnt: &[f64],
    order: &[K],
    prob: &mut [f64],
    opts: &TrainOptions,
) {
    let mut tot = 0.0f64;
    if opts.variational_bayes {
        let alpha = opts.alpha;
        for &k in order {
            tot += cnt[k.into() as usize] + alpha;
        }
        if tot == 0.0 {
            tot = 1.0;
        }
        let dt = math::digamma(tot);
        for (p, &c) in prob.iter_mut().zip(cnt) {
            *p = (math::digamma(c + alpha) - dt).exp();
        }
    } else {
        for &k in order {
            tot += cnt[k.into() as usize];
        }
        if tot == 0.0 {
            tot = 1.0;
        }
        for (p, &c) in prob.iter_mut().zip(cnt) {
            *p = c / tot;
        }
    }
}

/// Where one direction's log and alignments go.
pub struct Sinks<'a> {
    pub log: &'a mut (dyn Write + Send),
    pub out: &'a mut (dyn Write + Send),
}

/// Train two directions concurrently on one corpus, sharing the current rayon pool.
pub fn train_both<'c>(
    corpus: &'c Corpus,
    fwd: TrainOptions,
    rev: TrainOptions,
    popts: &ParallelOptions,
    fwd_sinks: Sinks<'_>,
    rev_sinks: Sinks<'_>,
    cancel: &Cancel,
) -> io::Result<(Trainer<'c>, Trainer<'c>)> {
    let run = |opts: TrainOptions, sinks: Sinks<'_>| -> io::Result<Trainer<'c>> {
        let mut t = Trainer::new(corpus, opts, popts.clone(), cancel, sinks.log)?;
        t.train(sinks.log, sinks.out)?;
        Ok(t)
    };
    let (a, b) = rayon::join(|| run(fwd, fwd_sinks), || run(rev, rev_sinks));
    Ok((a?, b?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::LoadOptions;

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

    /// Review #12: the row-line CSR parts are cut so that their `u32` offsets cannot overflow;
    /// every row's lines, single positions and costs are the same whatever the cut.
    #[test]
    fn row_lines_do_not_depend_on_the_part_limit() {
        let c = varied_corpus();
        let n_rows = c.vocab.len();
        let longest = (0..c.len()).map(|k| c.first.line_len(k)).max().unwrap();
        let cancel = Cancel::default();
        for use_null in [true, false] {
            // one thread: one part unless the limit cuts it
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let base =
                pool.install(|| RowLines::build(&c.first, &c.second, n_rows, use_null, &cancel));
            assert_eq!(base.parts.len(), 1);
            for limit in [longest, longest + 3, 50, 997] {
                let r = pool.install(|| {
                    RowLines::build_with(&c.first, &c.second, n_rows, use_null, limit, &cancel)
                });
                assert!(r.parts.len() > base.parts.len());
                for p in &r.parts {
                    assert!(p.lines.len() <= limit);
                }
                assert_eq!(r.cost, base.cost);
                assert_eq!(r.eps_in_src, base.eps_in_src);
                for e in 0..n_rows as u32 {
                    if use_null && e == NULL_ID {
                        continue;
                    }
                    assert!(base.lines(e).eq(r.lines(e)), "row {e}, limit {limit}");
                    let (mut a, mut b) = ((0, 0), (0, 0));
                    let x: Vec<_> = base.lines_before(e, &mut a, 200).collect();
                    let y: Vec<_> = r.lines_before(e, &mut b, 200).collect();
                    assert_eq!(x, y);
                }
            }
        }
    }
}
