//! Exact sequential trainer: a direct, single-threaded implementation of
//! `docs/SPEC-fast-align.md`, the test oracle for every faster implementation.
//!
//! Clarity over speed. Every floating-point value is computed with the same operations in the
//! same order as `OMP_NUM_THREADS=1 fast_align`, and everything fast_align writes (alignments,
//! scores, the `-p` table, the stderr log including progress dots and error messages) is
//! reproduced byte for byte. Section numbers (§) refer to the spec.

use std::io::{self, Write};

use rustc_hash::FxHashMap;

use crate::align::{align_pair, AlignOptions};
use crate::cancel::Cancel;
use crate::corpus::{Corpus, Vocab, NULL_ID};
use crate::math::{self, G};
use crate::ttable::{initial_pass, InitialPass, TTable, TableOptions};

/// Probability returned by `prob()` before the first M-step, and for unseen pairs in forced
/// alignment.
pub const UNINITIALISED_PROB: f64 = 1e-9;

/// `ln 2`, as GCC folds `log(2)` (spec §10.1).
const LN_2: f64 = std::f64::consts::LN_2;

/// fast_align's training options (spec §2), with fast_align's defaults.
#[derive(Clone, Debug)]
pub struct TrainOptions {
    /// `-r`
    pub reverse: bool,
    /// `-I` (no iterations if ≤ 0)
    pub iterations: i32,
    /// `-d`
    pub favor_diagonal: bool,
    /// `-o`
    pub optimize_tension: bool,
    /// `-v`
    pub variational_bayes: bool,
    /// `-a`
    pub alpha: f64,
    /// `false` for `-N`
    pub use_null: bool,
    /// `-q`
    pub p0: f64,
    /// `-T`: starting tension λ
    pub tension: f64,
    /// `-b`: lines per buffer; groups the per-iteration sums (§9). 0 behaves like 1.
    pub buffer_size: u64,
    /// `-s`
    pub print_scores: bool,
}

impl Default for TrainOptions {
    fn default() -> Self {
        TrainOptions {
            reverse: false,
            iterations: 5,
            favor_diagonal: false,
            optimize_tension: false,
            variational_bayes: false,
            alpha: 0.01,
            use_null: true,
            p0: 0.08,
            tension: 4.0,
            buffer_size: 10000,
            print_scores: false,
        }
    }
}

/// Totals of one iteration (spec §9, §10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IterationStats {
    pub likelihood: f64,
    /// Sum of NULL posteriors (0 in the final iteration).
    pub c0: f64,
    /// `emp_feat / n_target_tokens` (0 in the final iteration).
    pub emp_feat: f64,
    /// Tension used by this iteration's E-step.
    pub tension: f64,
    /// Tension after this iteration's optimisation (= `tension` if none ran).
    pub tension_after: f64,
}

/// fast_align's progress output: a dot every 1000 lines, ` [N]` and a newline every 50,000
/// (spec §15.4).
#[derive(Default)]
struct Progress {
    flag: bool,
}

impl Progress {
    fn tick(&mut self, lc: usize, log: &mut dyn Write) -> io::Result<()> {
        if lc % 1000 == 0 {
            log.write_all(b".")?;
            self.flag = true;
        }
        if lc % 50000 == 0 {
            writeln!(log, " [{lc}]")?;
            self.flag = false;
        }
        Ok(())
    }

    fn finish(&mut self, log: &mut dyn Write) -> io::Result<()> {
        if self.flag {
            log.write_all(b"\n")?;
        }
        Ok(())
    }
}

/// Per-buffer partial sums (spec §9).
#[derive(Default)]
struct Partial {
    likelihood: f64,
    c0: f64,
    emp_feat: f64,
}

/// The sequential trainer for one direction.
pub struct Oracle<'c> {
    corpus: &'c Corpus,
    opts: TrainOptions,
    pass: InitialPass,
    /// Probabilities, parallel to `pass.table.cols()` (fast_align's `ttable`).
    prob: Vec<f64>,
    /// Expected counts, parallel to `pass.table.cols()` (fast_align's `counts`).
    count: Vec<f64>,
    /// Whether an M-step has run (`prob()` returns 1e-9 until then).
    probs_initialized: bool,
    tension: f64,
    /// Scratch: cell values `probs[0..=n]` and their cell indices.
    scratch_vals: Vec<f64>,
    scratch_cells: Vec<usize>,
}

impl<'c> Oracle<'c> {
    /// Run the initial pass (§6), writing fast_align's initial-pass log to `log`.
    pub fn new(corpus: &'c Corpus, opts: TrainOptions, log: &mut dyn Write) -> io::Result<Self> {
        log.write_all(b"INITIAL PASS \n")?;
        let mut progress = Progress::default();
        let mut errors = corpus.empty_side_lines().iter().peekable();
        for k in 0..corpus.len() {
            let lc = k + 1;
            progress.tick(lc, log)?;
            if let Some((_, raw)) = errors.next_if(|(i, _)| *i == k) {
                writeln!(log, "Error in line {lc}")?;
                log.write_all(raw)?;
                log.write_all(b"\n")?;
            }
        }
        let pass = initial_pass(
            corpus,
            TableOptions {
                reverse: opts.reverse,
                use_null: opts.use_null,
            },
        );
        progress.finish(log)?;
        writeln!(
            log,
            "expected target length = source length * {}",
            G(pass.mean_srclen_multiplier)
        )?;
        let cells = pass.table.n_cells();
        Ok(Oracle {
            corpus,
            tension: opts.tension,
            opts,
            pass,
            prob: vec![0.0; cells],
            count: vec![0.0; cells],
            probs_initialized: false,
            scratch_vals: Vec::new(),
            scratch_cells: Vec::new(),
        })
    }

    /// The initial-pass results (table structure, length statistics).
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

    /// Current tension λ.
    pub fn tension(&self) -> f64 {
        self.tension
    }

    pub fn options(&self) -> &TrainOptions {
        &self.opts
    }

    /// Run all iterations (§7): statistics to `log`, alignments of the final iteration to `out`.
    pub fn train(&mut self, log: &mut dyn Write, out: &mut dyn Write) -> io::Result<()> {
        let iters = self.opts.iterations.max(0) as usize;
        for iter in 0..iters {
            self.iteration(iter, iter + 1 == iters, log, out)?;
        }
        Ok(())
    }

    /// One iteration: E-step over the corpus in buffers, statistics, then (unless `final_iter`)
    /// tension optimisation and the M-step. In the final iteration the alignments are written
    /// to `out` instead of collecting counts.
    pub fn iteration(
        &mut self,
        iter: usize,
        final_iter: bool,
        log: &mut dyn Write,
        out: &mut dyn Write,
    ) -> io::Result<IterationStats> {
        let corpus = self.corpus;
        writeln!(
            log,
            "ITERATION {}{}",
            iter + 1,
            if final_iter { " (FINAL)" } else { "" }
        )?;
        let n_lines = corpus.len();
        let buffer_size = self.opts.buffer_size.max(1);
        let mut progress = Progress::default();
        let (mut likelihood, mut c0, mut emp_feat) = (0.0f64, 0.0f64, 0.0f64);
        let mut errors = corpus.empty_side_lines().iter().peekable();
        let mut line_out = String::new();
        let mut start = 0;
        for k in 0..n_lines {
            let lc = k + 1;
            progress.tick(lc, log)?;
            let last = k + 1 == n_lines;
            if ((lc - start) as u64) < buffer_size && !last {
                continue;
            }
            // Process lines start..lc as one buffer; fast_align's line counter is lc.
            let mut part = Partial::default();
            for line in start..lc {
                if let Some((_, raw)) = errors.next_if(|(i, _)| *i == line) {
                    writeln!(log, "Error in line {lc}")?;
                    log.write_all(raw)?;
                    log.write_all(b"\n")?;
                }
                line_out.clear();
                self.e_step_line(line, final_iter, &mut part, &mut line_out);
                if final_iter {
                    out.write_all(line_out.as_bytes())?;
                }
            }
            likelihood += part.likelihood;
            c0 += part.c0;
            emp_feat += part.emp_feat;
            start = lc;
        }

        let t = self.pass.n_target_tokens;
        let base2 = likelihood / LN_2;
        progress.finish(log)?;
        let emp = emp_feat / t;
        writeln!(log, "  log_e likelihood: {}", G(likelihood))?;
        writeln!(log, "  log_2 likelihood: {}", G(base2))?;
        writeln!(log, "     cross entropy: {}", G(-base2 / t))?;
        writeln!(log, "        perplexity: {}", G(math::pow(2.0, -base2 / t)))?;
        writeln!(log, "      posterior p0: {}", G(c0 / t))?;
        writeln!(log, " posterior al-feat: {}", G(emp))?;
        writeln!(log, "       size counts: {}", self.pass.size_counts.len())?;
        let tension = self.tension;
        if !final_iter {
            if self.opts.favor_diagonal && self.opts.optimize_tension && iter > 0 {
                self.optimize_tension(emp, log)?;
            }
            self.m_step();
        }
        Ok(IterationStats {
            likelihood,
            c0,
            emp_feat: emp,
            tension,
            tension_after: self.tension,
        })
    }

    #[inline]
    fn prob_of(&self, cell: usize) -> f64 {
        if self.probs_initialized {
            self.prob[cell]
        } else {
            UNINITIALISED_PROB
        }
    }

    fn cell(&self, e: u32, f: u32) -> usize {
        self.pass
            .table
            .find(e, f)
            .expect("every (source, target) pair of the corpus is in the table")
    }

    /// E-step (or Viterbi decoding) of one line (§8). Appends the output line to `out` in the
    /// final iteration.
    fn e_step_line(&mut self, line: usize, final_iter: bool, part: &mut Partial, out: &mut String) {
        let (src, trg) = self.corpus.sides(self.opts.reverse);
        let (src, trg) = (src.line(line), trg.line(line));
        let n = src.len();
        let m = trg.len();
        let (n32, m32) = (n as u32, m as u32);
        let use_null = self.opts.use_null;
        let diag = self.opts.favor_diagonal;
        let p0 = self.opts.p0;
        let pnn = 1.0 - p0;
        let lambda = self.tension;
        let uniform = 1.0 / (n + usize::from(use_null)) as f64;

        let mut probs = std::mem::take(&mut self.scratch_vals);
        let mut cells = std::mem::take(&mut self.scratch_cells);
        probs.clear();
        probs.resize(n + 1, 0.0);
        cells.clear();
        cells.resize(n + 1, usize::MAX);

        let mut ll = 0.0f64;
        let mut first_link = true;
        for (j, &f) in trg.iter().enumerate() {
            let j32 = j as u32;
            let mut sum = 0.0f64;
            let mut a = uniform;
            if use_null {
                if diag {
                    a = p0;
                }
                cells[0] = self.cell(NULL_ID, f);
                probs[0] = self.prob_of(cells[0]) * a;
                sum += probs[0];
            }
            let mut az = 0.0;
            if diag {
                az = math::compute_z(j32 + 1, m32, n32, lambda) / pnn;
            }
            for i in 1..=n {
                if diag {
                    a = math::unnormalized_prob(j32 + 1, i as u32, m32, n32, lambda) / az;
                }
                cells[i] = self.cell(src[i - 1], f);
                probs[i] = self.prob_of(cells[i]) * a;
                sum += probs[i];
            }
            if final_iter {
                let (mut best, mut best_p) = if use_null { (0, probs[0]) } else { (-1, -1.0) };
                for (i, &p) in probs.iter().enumerate().skip(1) {
                    if p > best_p {
                        best = i as i64;
                        best_p = p;
                    }
                }
                if best > 0 {
                    if !first_link {
                        out.push(' ');
                    }
                    first_link = false;
                    let s = best - 1;
                    if self.opts.reverse {
                        out.push_str(&format!("{j}-{s}"));
                    } else {
                        out.push_str(&format!("{s}-{j}"));
                    }
                }
            } else {
                if use_null {
                    let c = probs[0] / sum;
                    part.c0 += c;
                    self.count[cells[0]] += c;
                }
                for i in 1..=n {
                    let p = probs[i] / sum;
                    self.count[cells[i]] += p;
                    part.emp_feat += math::feature(j32, i as u32, m32, n32) * p;
                }
            }
            ll += sum.ln();
        }
        part.likelihood += ll;
        if final_iter {
            if self.opts.print_scores {
                let lambda = 0.05 + n as f64 * self.pass.mean_srclen_multiplier;
                let lp = math::log_poisson(m32, lambda) + ll;
                out.push_str(" ||| ");
                math::write_g(out, lp);
            }
            out.push('\n');
        }
        self.scratch_vals = probs;
        self.scratch_cells = cells;
    }

    /// Eight gradient steps on λ (§10.2).
    fn optimize_tension(&mut self, emp: f64, log: &mut dyn Write) -> io::Result<()> {
        let t = self.pass.n_target_tokens;
        for ii in 1..=8 {
            let lambda = self.tension;
            let mut mod_feat = 0.0f64;
            for sc in &self.pass.size_counts {
                // fast_align's `short j` loop and short → unsigned conversions.
                let m = i32::from(sc.m);
                let (mu, nu) = (m as u32, i32::from(sc.n) as u32);
                for j in 1..=m {
                    mod_feat += f64::from(sc.count) * math::compute_dlogz(j as u32, mu, nu, lambda);
                }
            }
            mod_feat /= t;
            writeln!(
                log,
                "  {ii}  model al-feat: {} (tension={})",
                G(mod_feat),
                G(lambda)
            )?;
            // fast_align tests `<= 0.1` then `> 14`; `clamp` gives the same result for every
            // input, NaN included.
            self.tension = (lambda + (emp - mod_feat) * 20.0).clamp(0.1, 14.0);
        }
        writeln!(log, "     final tension: {}", G(self.tension))
    }

    /// `Normalize` / `NormalizeVB` (§11): row totals in hash order; counts reset to 0.
    fn m_step(&mut self) {
        let table = &self.pass.table;
        let alpha = self.opts.alpha;
        for e in 0..table.n_rows() as u32 {
            let row = table.row(e);
            if row.is_empty() {
                continue;
            }
            let mut tot = 0.0f64;
            if self.opts.variational_bayes {
                for c in table.iter_hash_order(e) {
                    tot += self.count[c] + alpha;
                }
                if tot == 0.0 {
                    tot = 1.0;
                }
                let dt = math::digamma(tot);
                for c in row.clone() {
                    self.prob[c] = (math::digamma(self.count[c] + alpha) - dt).exp();
                }
            } else {
                for c in table.iter_hash_order(e) {
                    tot += self.count[c];
                }
                if tot == 0.0 {
                    tot = 1.0;
                }
                for c in row.clone() {
                    self.prob[c] = self.count[c] / tot;
                }
            }
            self.count[row].fill(0.0);
        }
        self.probs_initialized = true;
    }

    /// Write fast_align's `-p` table (§15.3): rows in id order, entries in hash order, natural
    /// log probabilities at or above `log(max_p) · (−beam_threshold)`.
    pub fn write_table(
        &self,
        vocab: &Vocab,
        beam_threshold: f64,
        w: &mut dyn Write,
    ) -> io::Result<()> {
        let table = &self.pass.table;
        let cols = table.cols();
        let mut line = String::new();
        for e in 0..table.n_rows() as u32 {
            let mut max_p = -1.0f64;
            for c in table.iter_hash_order(e) {
                if self.prob[c] > max_p {
                    max_p = self.prob[c];
                }
            }
            let threshold = -max_p.ln() * beam_threshold;
            for c in table.iter_hash_order(e) {
                let lp = self.prob[c].ln();
                if lp >= threshold {
                    w.write_all(vocab.word(e))?;
                    w.write_all(b"\t")?;
                    w.write_all(vocab.word(cols[c]))?;
                    line.clear();
                    line.push('\t');
                    math::write_g(&mut line, lp);
                    line.push('\n');
                    w.write_all(line.as_bytes())?;
                }
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------
// Forced alignment with a text table (`-f`, spec §16)

/// A translation table read from fast_align's `-p` text format, keyed by the ids of one
/// corpus's vocabulary. Entries whose words do not occur in that corpus can never be looked
/// up and are dropped.
pub struct TextTable {
    probs: FxHashMap<(u32, u32), f64>,
    /// fast_align's count of records (one more than the records read, see spec §16).
    pub loaded: usize,
}

impl TextTable {
    /// Parse `bytes` as fast_align's `DeserializeLogProbsFromText` does, with C++ stream
    /// semantics (fields separated by C whitespace; `e`, `f`, then a number, repeated).
    pub fn parse(bytes: &[u8], vocab: &Vocab) -> Self {
        Self::parse_with(bytes, vocab, &Cancel::default())
    }

    /// [`TextTable::parse`], checking `cancel` every [`CANCEL_RECORDS`] records.
    pub fn parse_with(bytes: &[u8], vocab: &Vocab, cancel: &Cancel) -> Self {
        cancel.check();
        let mut ids: FxHashMap<&[u8], u32> = FxHashMap::default();
        for id in 1..vocab.len() as u32 {
            ids.insert(vocab.word(id), id);
        }
        let mut s = CStream {
            buf: bytes,
            pos: 0,
            fail: false,
            eof: false,
        };
        let (mut e, mut f): (&[u8], &[u8]) = (b"", b"");
        let mut p = 0.0f64;
        let mut probs = FxHashMap::default();
        let mut loaded = 0;
        // `while (*in)` tests failbit only: a field ending at end of input sets eofbit, and the
        // next `>> e` then fails, re-storing the previous record (records + 1).
        while !s.fail {
            if loaded % CANCEL_RECORDS == 0 {
                cancel.check();
            }
            if let Some(w) = s.word() {
                e = w;
            }
            if let Some(w) = s.word() {
                f = w;
            }
            if let Some(x) = s.number() {
                p = x;
            }
            if e.is_empty() {
                break;
            }
            loaded += 1;
            if let (Some(&ie), Some(&jf)) = (ids.get(e), ids.get(f)) {
                probs.insert((ie, jf), p.exp());
            }
        }
        TextTable { probs, loaded }
    }

    /// fast_align's `safe_prob`: the stored probability, or 1e-9.
    #[inline]
    pub fn prob(&self, e: u32, f: u32) -> f64 {
        self.probs
            .get(&(e, f))
            .copied()
            .unwrap_or(UNINITIALISED_PROB)
    }
}

/// Minimal emulation of `std::istream >> std::string` and `>> double` (C locale).
struct CStream<'a> {
    buf: &'a [u8],
    pos: usize,
    fail: bool,
    eof: bool,
}

impl<'a> CStream<'a> {
    fn is_space(c: u8) -> bool {
        matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
    }

    /// Sentry: skip whitespace; fail at end of input.
    fn sentry(&mut self) -> bool {
        if self.fail || self.eof {
            self.fail = true;
            return false;
        }
        while self.pos < self.buf.len() && Self::is_space(self.buf[self.pos]) {
            self.pos += 1;
        }
        if self.pos == self.buf.len() {
            self.eof = true;
            self.fail = true;
            return false;
        }
        true
    }

    fn word(&mut self) -> Option<&'a [u8]> {
        if !self.sentry() {
            return None;
        }
        let start = self.pos;
        while self.pos < self.buf.len() && !Self::is_space(self.buf[self.pos]) {
            self.pos += 1;
        }
        if self.pos == self.buf.len() {
            self.eof = true;
        }
        Some(&self.buf[start..self.pos])
    }

    /// `>> double`: libstdc++ accumulates `[+-]digits[.digits][(e|E)[+-]digits]` and converts it
    /// with `strtod`; an invalid accumulation stores 0 and sets failbit. Returns the stored
    /// value, or `None` if nothing was stored (sentry failed).
    fn number(&mut self) -> Option<f64> {
        if !self.sentry() {
            return None;
        }
        let b = self.buf;
        let start = self.pos;
        let mut p = self.pos;
        let digits = |p: &mut usize| {
            let s = *p;
            while *p < b.len() && b[*p].is_ascii_digit() {
                *p += 1;
            }
            *p > s
        };
        if p < b.len() && (b[p] == b'+' || b[p] == b'-') {
            p += 1;
        }
        let mut mant = digits(&mut p);
        if p < b.len() && b[p] == b'.' {
            p += 1;
            mant |= digits(&mut p);
        }
        if mant && p < b.len() && (b[p] == b'e' || b[p] == b'E') {
            p += 1;
            if p < b.len() && (b[p] == b'+' || b[p] == b'-') {
                p += 1;
            }
            digits(&mut p);
        }
        self.pos = p;
        if p == b.len() {
            self.eof = true;
        }
        let text = std::str::from_utf8(&b[start..p]).unwrap_or("");
        match text.parse::<f64>() {
            Ok(x) if x.is_finite() => Some(x),
            _ => {
                self.fail = true;
                Some(0.0)
            }
        }
    }
}

/// Text-table records read between two cancellation checks.
pub const CANCEL_RECORDS: usize = 1 << 16;

/// Lines aligned by [`force_align`] between two cancellation checks.
pub const CANCEL_FORCE_LINES: usize = 256;

/// Options of forced alignment (the training options that apply, spec §16).
#[derive(Clone, Debug)]
pub struct ForceOptions {
    pub reverse: bool,
    pub favor_diagonal: bool,
    pub use_null: bool,
    pub p0: f64,
    pub tension: f64,
    /// `-m`
    pub mean_srclen_multiplier: f64,
}

/// Outcome of [`force_align`].
#[derive(Debug, PartialEq)]
pub enum ForceResult {
    /// All lines aligned; total log probability.
    Done(f64),
    /// A line (1-based number) had an empty side; fast_align exits with status 1 there.
    EmptySide(usize),
}

/// fast_align's `-f` mode over `corpus` (spec §16). Writes `SRC ||| TGT ||| LINKS ||| LOGPROB`
/// lines to `out` and the messages to `log` (except `Loaded …`, which the caller prints).
/// `cancel` is checked every [`CANCEL_FORCE_LINES`] lines (see [`crate::cancel`]).
pub fn force_align(
    corpus: &Corpus,
    table: &TextTable,
    opts: &ForceOptions,
    cancel: &Cancel,
    log: &mut dyn Write,
    out: &mut dyn Write,
) -> io::Result<ForceResult> {
    let vocab = &corpus.vocab;
    let mut tlp = 0.0f64;
    let mut line = Vec::new();
    let mut links = Vec::new();
    for k in 0..corpus.len() {
        if k % CANCEL_FORCE_LINES == 0 {
            cancel.check();
        }
        let lc = k + 1;
        let (first, second) = (corpus.first.line(k), corpus.second.line(k));
        line.clear();
        for &s in first {
            line.extend_from_slice(vocab.word(s));
            line.push(b' ');
        }
        line.extend_from_slice(b"|||");
        for &t in second {
            line.push(b' ');
            line.extend_from_slice(vocab.word(t));
        }
        line.extend_from_slice(b" |||");
        let (src, trg) = if opts.reverse {
            (second, first)
        } else {
            (first, second)
        };
        if src.is_empty() || trg.is_empty() {
            out.write_all(&line)?;
            writeln!(log, "Error in line {lc}")?;
            return Ok(ForceResult::EmptySide(lc));
        }
        let aopts = AlignOptions {
            reverse: opts.reverse,
            favor_diagonal: opts.favor_diagonal,
            use_null: opts.use_null,
            p0: opts.p0,
            tension: opts.tension,
            mean_srclen_multiplier: opts.mean_srclen_multiplier,
            training_argmax: false,
        };
        links.clear();
        let lp = align_pair(&|e, f| table.prob(e, f), first, second, &aopts, &mut links)
            .expect("both sides are non-empty");
        for &(x, y) in &links {
            write!(line, " {x}-{y}")?;
        }
        tlp += lp;
        line.extend_from_slice(b" ||| ");
        line.extend_from_slice(math::fmt_g(lp).as_bytes());
        line.push(b'\n');
        out.write_all(&line)?;
    }
    writeln!(log, "TOTAL LOG PROB {}", G(tlp))?;
    Ok(ForceResult::Done(tlp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::LoadOptions;

    fn corpus(text: &[u8]) -> Corpus {
        Corpus::from_joined_bytes(text, &LoadOptions::default()).unwrap()
    }

    #[test]
    fn text_table_stream_semantics() {
        let c = corpus(b"a b ||| x y\n");
        // a=3 b=4 x=5 y=6
        let t = TextTable::parse(b"a\tx\t-1\nb x -0.5\n<eps>\ty\t-2\n", &c.vocab);
        assert_eq!(t.loaded, 4); // records + 1
        assert_eq!(t.prob(3, 5), (-1.0f64).exp());
        assert_eq!(t.prob(4, 5), (-0.5f64).exp());
        assert_eq!(t.prob(NULL_ID, 6), (-2.0f64).exp());
        assert_eq!(t.prob(3, 6), 1e-9);
        // -inf stops reading; that record is stored with p = 0 (probability 1).
        let t = TextTable::parse(b"a\tx\t-inf\nb\tx\t-1\n", &c.vocab);
        assert_eq!(t.loaded, 1);
        assert_eq!(t.prob(3, 5), 1.0);
        assert_eq!(t.prob(4, 5), 1e-9);
        assert_eq!(TextTable::parse(b"", &c.vocab).loaded, 0);
        assert_eq!(TextTable::parse(b"\n\n", &c.vocab).loaded, 0);
    }

    #[test]
    fn small_run_is_consistent() {
        let c = corpus(b"a b ||| x y\nb ||| y\na ||| x\n");
        let opts = TrainOptions {
            favor_diagonal: true,
            optimize_tension: true,
            variational_bayes: true,
            print_scores: true,
            ..Default::default()
        };
        let mut log = Vec::new();
        let mut out = Vec::new();
        let mut o = Oracle::new(&c, opts, &mut log).unwrap();
        o.train(&mut log, &mut out).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert_eq!(out.lines().count(), 3);
        assert!(out.starts_with("0-0 1-1 ||| "), "{out}");
        let log = String::from_utf8(log).unwrap();
        assert!(log.starts_with("INITIAL PASS \nexpected target length"));
        assert_eq!(log.matches("ITERATION").count(), 5);
        assert!(log.contains("ITERATION 5 (FINAL)\n"));
        assert!(log.contains("final tension"));
    }
}
