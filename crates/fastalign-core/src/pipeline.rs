//! The training pipeline: one or both directions of a loaded corpus (one after the other, or
//! concurrently), with per-iteration progress events, per-operation cooperative cancellation
//! (see [`crate::cancel`]) and the results packaged as a [`Model`] plus Pharaoh alignments.
//!
//! Both directions share one rayon pool (the global pool, or the one [`with_threads`] installs).
//! Nothing here changes a computed bit: the result is that of fast_align run once per direction
//! with `OMP_NUM_THREADS=1`, whatever the thread count.

use std::fmt;
use std::io;
use std::time::{Duration, Instant};

use crate::cancel::Cancel;
use rayon::prelude::*;

use crate::corpus::{Corpus, EmptySide, Vocab};
use crate::io::{DirModel, Direction, Model, SubwordInfo};
use crate::math;
use crate::oracle::{IterationStats, TrainOptions};
use crate::subword::project_text;
use crate::symmetrize::{symmetrize_text, Method};
use crate::train::{ParallelOptions, Trainer};

const LN_2: f64 = std::f64::consts::LN_2;

/// Which directions to train.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Directions {
    Both,
    Forward,
    Inverse,
}

impl Directions {
    pub fn list(self) -> Vec<Direction> {
        match self {
            Directions::Both => vec![Direction::Forward, Direction::Inverse],
            Directions::Forward => vec![Direction::Forward],
            Directions::Inverse => vec![Direction::Inverse],
        }
    }
}

/// All options of a training run. The default is fast_align's recommended `-d -o -v`, five
/// iterations, both directions.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// fast_align's options (`reverse` is ignored: `directions` decides).
    pub train: TrainOptions,
    pub directions: Directions,
    /// Tuning knobs of the parallel trainer; none changes any output bit.
    pub parallel: ParallelOptions,
    /// With `Both`: train the two directions at the same time, sharing the pool, instead of
    /// one after the other with the whole pool each. Off by default: at 4 or more threads it is
    /// no faster and it holds both trainers in memory at once (Step 9b, `RESULTS.md`).
    pub concurrent_directions: bool,
    /// Keep each direction's trained model in [`Trained::model`]. A caller that only needs the
    /// alignments (the CLI without `--save` or `--table`) turns it off, so the finished first
    /// direction's table and probabilities are freed before the second direction trains.
    pub keep_models: bool,
    /// The corpus is segmented into pieces (see [`crate::subword`]): every direction's alignments
    /// are projected to words (so the symmetrisation of [`Trained::symmetrized`] is
    /// project-then-symmetrise), and the model remembers this. `None` changes nothing.
    pub subword: Option<SubwordInfo>,
    /// With `subword`: also keep each direction's piece-level alignments
    /// ([`DirOutput::subword_alignments`]).
    pub keep_subword_output: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            train: TrainOptions {
                favor_diagonal: true,
                optimize_tension: true,
                variational_bayes: true,
                ..TrainOptions::default()
            },
            directions: Directions::Both,
            parallel: ParallelOptions::default(),
            concurrent_directions: false,
            keep_models: true,
            subword: None,
            keep_subword_output: false,
        }
    }
}

/// One progress event: an iteration of one direction has finished.
#[derive(Clone, Debug)]
pub struct Progress {
    pub direction: Direction,
    /// 1-based.
    pub iteration: usize,
    pub iterations: usize,
    /// Natural-log likelihood of the corpus under this iteration's E-step.
    pub log_likelihood: f64,
    pub perplexity: f64,
    /// Posterior mass of the NULL word per target token (0 in the final, decode-only iteration).
    pub p0: f64,
    /// Tension after this iteration.
    pub tension: f64,
    /// Time since this direction started training.
    pub elapsed: Duration,
}

impl fmt::Display for Progress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] iteration {}/{}: log-likelihood {:.1}, perplexity {:.4}, p0 {:.4}, tension {:.4} ({:.1} s)",
            self.direction.name(),
            self.iteration,
            self.iterations,
            self.log_likelihood,
            self.perplexity,
            self.p0,
            self.tension,
            self.elapsed.as_secs_f64()
        )
    }
}

/// What a direction's training wrote besides its model.
#[derive(Clone, Debug)]
pub struct DirOutput {
    pub direction: Direction,
    /// Whether a decoding iteration ran (`iterations` ≥ 1). Without one (fast_align's `-I 0`)
    /// there are no alignments at all: `alignments` is empty (zero lines, as fast_align's
    /// stdout), whatever filtering or subword projection was asked for.
    pub decoded: bool,
    /// Pharaoh alignments of the corpus, one line per *input* line (lines dropped by `max_len`
    /// or [`crate::corpus::EmptyPairs::Skip`] are empty lines), in the corpus's own orientation
    /// (first side index first). With `-s` each line ends with ` ||| score`. Empty when not
    /// [`DirOutput::decoded`].
    pub alignments: Vec<u8>,
    /// With a subword corpus and [`TrainConfig::keep_subword_output`]: the piece-level
    /// alignments `alignments` was projected from (positions are pieces; one line per input
    /// line too).
    pub subword_alignments: Option<Vec<u8>>,
    /// fast_align's stderr statistics for this direction (without the `ARG=` option echo).
    pub log: Vec<u8>,
    /// Totals of every iteration.
    pub stats: Vec<IterationStats>,
    /// Trained parameters that ended nonfinite (see [`Nonfinite`]).
    pub nonfinite: Nonfinite,
}

/// The trained parameters of a direction that are not finite numbers. fast_align's arithmetic
/// produces them from degenerate input (an empty source side: infinite length ratio, NaN tension;
/// a blank line: NaN length ratio; see [`crate::corpus::EmptyPairs`]); training completes and
/// the reference-format log shows them as they are. [`Trained::warnings`] reports them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Nonfinite {
    /// The final tension, if nonfinite (only with the diagonal prior).
    pub tension: Option<f64>,
    /// The mean target/source length ratio used by scores, if nonfinite.
    pub mean_srclen_multiplier: Option<f64>,
    /// How many translation probabilities are nonfinite, of how many.
    pub probabilities: (usize, usize),
}

impl Nonfinite {
    fn of(tension: f64, favor_diagonal: bool, mean: f64, probs: &[f64]) -> Self {
        let bad = |x: f64| (!x.is_finite()).then_some(x);
        Nonfinite {
            tension: bad(tension).filter(|_| favor_diagonal),
            mean_srclen_multiplier: bad(mean),
            probabilities: (
                probs
                    .par_chunks(1 << 16)
                    .map(|c| c.iter().filter(|p| !p.is_finite()).count())
                    .sum(),
                probs.len(),
            ),
        }
    }

    pub fn any(&self) -> bool {
        self.tension.is_some() || self.mean_srclen_multiplier.is_some() || self.probabilities.0 > 0
    }
}

impl fmt::Display for Nonfinite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(t) = self.tension {
            parts.push(format!("tension {}", math::fmt_g(t)));
        }
        if let Some(m) = self.mean_srclen_multiplier {
            parts.push(format!("mean length ratio {}", math::fmt_g(m)));
        }
        let (bad, all) = self.probabilities;
        if bad > 0 {
            parts.push(format!("{bad} of {all} translation probabilities"));
        }
        write!(f, "{}", parts.join(", "))
    }
}

/// The result of [`train`]. `model` has no direction when [`TrainConfig::keep_models`] is off.
#[derive(Clone, Debug)]
pub struct Trained {
    pub model: Model,
    pub fwd: Option<DirOutput>,
    pub inv: Option<DirOutput>,
}

impl Trained {
    /// The direction's output, if it was trained.
    pub fn output(&self, d: Direction) -> Option<&DirOutput> {
        match d {
            Direction::Forward => self.fwd.as_ref(),
            Direction::Inverse => self.inv.as_ref(),
        }
    }

    /// Symmetrised alignments (needs both directions), identical to `atools -c METHOD` on the
    /// two directional files.
    pub fn symmetrized(&self, method: Method) -> Result<Vec<u8>, String> {
        let (Some(f), Some(i)) = (&self.fwd, &self.inv) else {
            return Err("symmetrisation needs both directions".into());
        };
        if !f.decoded || !i.decoded {
            return Ok(Vec::new());
        }
        symmetrize_pharaoh(&f.alignments, &i.alignments, method)
    }

    /// [`Trained::symmetrized`], written to `w` instead of collected (no second copy of the
    /// text in memory).
    pub fn write_symmetrized(&self, method: Method, w: &mut dyn io::Write) -> Result<(), String> {
        let (Some(f), Some(i)) = (&self.fwd, &self.inv) else {
            return Err("symmetrisation needs both directions".into());
        };
        if !f.decoded || !i.decoded {
            return Ok(());
        }
        symmetrize_text(&f.alignments, Some(&i.alignments), method, w).map_err(|e| e.to_string())
    }

    /// Warnings about a finished training on `corpus` (the one it was trained on), for the
    /// high-level front ends (never part of the reference-format logs): the corpus has no pair
    /// to train on, or a direction's parameters ended nonfinite (with the empty-sided pairs that
    /// explain it). Empty when there is nothing to report.
    pub fn warnings(&self, corpus: &Corpus) -> Vec<String> {
        if corpus.is_empty() {
            let n = corpus.n_input_lines();
            return vec![if n == 0 {
                "the input has no line pairs: nothing was trained".to_owned()
            } else {
                format!(
                    "{} dropped (max_len / empty pairs 'skip'): nothing was trained",
                    if n == 1 {
                        "the only input line pair was".to_owned()
                    } else {
                        format!("all {n} input line pairs were")
                    }
                )
            }];
        }
        let mut out = Vec::new();
        for o in [&self.fwd, &self.inv].into_iter().flatten() {
            if o.nonfinite.any() {
                out.push(format!(
                    "{}: training ended with nonfinite parameters ({}); this direction's \
                     alignments and scores are unreliable{}",
                    o.direction.name(),
                    o.nonfinite,
                    empty_pairs_hint(corpus)
                ));
            }
        }
        out
    }
}

/// `"; the corpus has N line pairs with an empty side (...)"`, or `""` if it has none.
fn empty_pairs_hint(corpus: &Corpus) -> String {
    let lines = corpus.empty_side_lines();
    if lines.is_empty() {
        return String::new();
    }
    let mut report = crate::corpus::EmptyPairReport::default();
    for &(k, _) in lines {
        let side = match (
            corpus.first.line_len(k) == 0,
            corpus.second.line_len(k) == 0,
        ) {
            (true, true) => EmptySide::Both,
            (true, false) => EmptySide::Source,
            _ => EmptySide::Target,
        };
        match side {
            EmptySide::Source => report.source_empty += 1,
            EmptySide::Target => report.target_empty += 1,
            EmptySide::Both => report.both_empty += 1,
        }
        if report.first.len() < crate::corpus::EmptyPairReport::MAX_LISTED {
            report.first.push((corpus.line_number(k), side));
        }
    }
    format!("; the corpus has {report}, which fast_align keeps: empty pairs 'skip' drops them")
}

/// Symmetrise two directional Pharaoh texts (any layout `symmetrize::parse_line` accepts, so
/// ` ||| score` suffixes are ignored).
pub fn symmetrize_pharaoh(fwd: &[u8], inv: &[u8], method: Method) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(fwd.len());
    symmetrize_text(fwd, Some(inv), method, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// Run `f` on a rayon pool of `threads` threads (0 = the global pool, all cores). A pool that
/// cannot be built is an error: running on the ambient pool instead would silently ignore the
/// requested budget. A command makes one call around all its parallel phases (load, train,
/// export), so every phase runs under the same pool.
pub fn with_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> Result<T, String> {
    if threads == 0 {
        return Ok(f());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| format!("cannot start {threads} worker threads: {e}"))?;
    Ok(pool.install(f))
}

/// Return free heap memory to the operating system (glibc keeps freed memory of every thread's
/// arena otherwise). Allocation only: no value changes.
pub fn release_free_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: glibc's malloc_trim only releases free memory; it is thread-safe.
        unsafe {
            malloc_trim(0);
        }
    }
}

/// One line per input line (see [`Corpus::expand_lines`]), without copying an unfiltered text.
fn expand(corpus: &Corpus, text: Vec<u8>) -> Result<Vec<u8>, String> {
    Ok(match corpus.expand_lines(&text)? {
        std::borrow::Cow::Borrowed(_) => text,
        std::borrow::Cow::Owned(v) => v,
    })
}

fn train_direction(
    corpus: &Corpus,
    cfg: &TrainConfig,
    d: Direction,
    want_model: bool,
    progress: &(dyn Fn(&Progress) + Sync),
    cancel: &Cancel,
) -> io::Result<(Option<DirModel>, DirOutput)> {
    let opts = TrainOptions {
        reverse: d.reverse(),
        ..cfg.train.clone()
    };
    let start = Instant::now();
    let mut log = Vec::new();
    let mut out = Vec::new();
    let mut trainer = Trainer::new(corpus, opts, cfg.parallel.clone(), cancel, &mut log)?;
    // The initial pass freed its per-row vectors and build arrays on many threads' malloc
    // arenas; give that memory back before the iterations (PERF-TODO 4).
    release_free_memory();
    let iters = cfg.train.iterations.max(0) as usize;
    let decoded = iters > 0;
    let t = trainer.pass().n_target_tokens;
    let mut stats = Vec::with_capacity(iters);
    for iter in 0..iters {
        cancel.check();
        let s = trainer.iteration(iter, iter + 1 == iters, &mut log, &mut out)?;
        stats.push(s);
        progress(&Progress {
            direction: d,
            iteration: iter + 1,
            iterations: iters,
            log_likelihood: s.likelihood,
            perplexity: math::pow(2.0, -(s.likelihood / LN_2) / t),
            p0: s.c0 / t,
            tension: s.tension_after,
            elapsed: start.elapsed(),
        });
    }
    let invalid = |e: String| io::Error::new(io::ErrorKind::InvalidData, e);
    let mut subword_alignments = None;
    if decoded {
        if let Some(sw) = &cfg.subword {
            let words =
                project_text(corpus, &sw.boundary, &out, d.reverse(), cancel).map_err(invalid)?;
            let raw = std::mem::replace(&mut out, words);
            if cfg.keep_subword_output {
                subword_alignments = Some(expand(corpus, raw).map_err(invalid)?);
            }
        }
        out = expand(corpus, out).map_err(invalid)?;
    } else if cfg.subword.is_some() && cfg.keep_subword_output {
        subword_alignments = Some(Vec::new());
    }
    let nonfinite = Nonfinite::of(
        trainer.tension(),
        cfg.train.favor_diagonal,
        trainer.pass().mean_srclen_multiplier,
        trainer.probs(),
    );
    let model = want_model.then(|| DirModel::from_trainer(trainer.into_parts(), d.reverse()));
    release_free_memory();
    Ok((
        model,
        DirOutput {
            direction: d,
            decoded,
            alignments: out,
            subword_alignments,
            log,
            stats,
            nonfinite,
        },
    ))
}

/// A direction that has just finished training, as handed to a [`train_with_sink`] sink.
pub struct Finished<'a> {
    pub direction: Direction,
    /// The corpus's vocabulary (the words the model's ids refer to).
    pub vocab: &'a Vocab,
    pub output: &'a DirOutput,
    pub model: &'a DirModel,
}

/// Called on the calling thread once a direction has finished, before the next one starts.
pub type Sink<'s> = &'s mut dyn FnMut(&Finished<'_>) -> io::Result<()>;

/// Train the configured directions on `corpus` (with `Both`, one after the other unless
/// [`TrainConfig::concurrent_directions`]).
/// `progress` is called (from worker threads) after every iteration. `cancel` is this
/// operation's token, checked during setup, every iteration, every phase task and the subword
/// projection (see [`crate::cancel`]); `&Cancel::default()` cannot be cancelled.
pub fn train(
    corpus: &Corpus,
    cfg: &TrainConfig,
    progress: &(dyn Fn(&Progress) + Sync),
    cancel: &Cancel,
) -> io::Result<Trained> {
    train_with_sink(corpus, cfg, progress, cancel, None)
}

/// [`train`] with a per-direction output sink: `sink` sees each finished direction's model
/// and output (a table export, say) and the model is then dropped, unless
/// [`TrainConfig::keep_models`], before the next direction trains. So a job that only writes
/// files never holds two directions' tables. With concurrent directions both have finished
/// before the sink is called for either. A sink error stops the run.
pub fn train_with_sink(
    corpus: &Corpus,
    cfg: &TrainConfig,
    progress: &(dyn Fn(&Progress) + Sync),
    cancel: &Cancel,
    mut sink: Option<Sink<'_>>,
) -> io::Result<Trained> {
    cancel.check();
    let want = cfg.keep_models || sink.is_some();
    let run = |d| train_direction(corpus, cfg, d, want, progress, cancel);
    let mut finish = |d: Direction,
                      (model, out): (Option<DirModel>, DirOutput)|
     -> io::Result<(Option<DirModel>, DirOutput)> {
        if let (Some(s), Some(m)) = (sink.as_mut(), model.as_ref()) {
            s(&Finished {
                direction: d,
                vocab: &corpus.vocab,
                output: &out,
                model: m,
            })?;
        }
        if cfg.keep_models {
            return Ok((model, out));
        }
        drop(model);
        release_free_memory();
        Ok((None, out))
    };
    let (mut fwd, mut inv) = match cfg.directions {
        Directions::Both if !cfg.concurrent_directions => {
            let a = finish(Direction::Forward, run(Direction::Forward)?)?;
            let b = finish(Direction::Inverse, run(Direction::Inverse)?)?;
            (Some(a), Some(b))
        }
        Directions::Both => {
            let (a, b) = rayon::join(|| run(Direction::Forward), || run(Direction::Inverse));
            let (a, b) = (a?, b?);
            (
                Some(finish(Direction::Forward, a)?),
                Some(finish(Direction::Inverse, b)?),
            )
        }
        Directions::Forward => (
            Some(finish(Direction::Forward, run(Direction::Forward)?)?),
            None,
        ),
        Directions::Inverse => (
            None,
            Some(finish(Direction::Inverse, run(Direction::Inverse)?)?),
        ),
    };
    let (fm, fo) = fwd.take().map_or((None, None), |(m, o)| (m, Some(o)));
    let (im, io_) = inv.take().map_or((None, None), |(m, o)| (m, Some(o)));
    let mut model = Model::new(corpus.vocab.clone(), fm, im);
    model.subword = cfg.subword.clone();
    Ok(Trained {
        model,
        fwd: fo,
        inv: io_,
    })
}
