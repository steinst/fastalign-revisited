//! The training pipeline: one or both directions of a loaded corpus (one after the other, or
//! concurrently), with per-iteration progress events, cooperative cancellation (see
//! [`crate::cancel`]) and the results packaged as a [`Model`] plus Pharaoh alignments.
//!
//! Both directions share one rayon pool (the global pool, or the one [`with_threads`] installs).
//! Nothing here changes a computed bit: the result is that of fast_align run once per direction
//! with `OMP_NUM_THREADS=1`, whatever the thread count.

use std::fmt;
use std::io;
use std::time::{Duration, Instant};

use crate::cancel;
use crate::corpus::Corpus;
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
    /// Pharaoh alignments of the corpus, one line per line, in the corpus's own orientation
    /// (first side index first). With `-s` each line ends with ` ||| score`.
    pub alignments: Vec<u8>,
    /// With a subword corpus and [`TrainConfig::keep_subword_output`]: the piece-level
    /// alignments `alignments` was projected from (positions are pieces).
    pub subword_alignments: Option<Vec<u8>>,
    /// fast_align's stderr statistics for this direction (without the `ARG=` option echo).
    pub log: Vec<u8>,
    /// Totals of every iteration.
    pub stats: Vec<IterationStats>,
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
        symmetrize_pharaoh(&f.alignments, &i.alignments, method)
    }
}

/// Symmetrise two directional Pharaoh texts, ignoring ` ||| score` suffixes.
pub fn symmetrize_pharaoh(fwd: &[u8], inv: &[u8], method: Method) -> Result<Vec<u8>, String> {
    let (f, i) = (crate::io::strip_scores(fwd), crate::io::strip_scores(inv));
    let mut out = Vec::with_capacity(f.len());
    symmetrize_text(&f, Some(&i), method, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// Run `f` on a rayon pool of `threads` threads (0 = the global pool, all cores).
pub fn with_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    if threads == 0 {
        return f();
    }
    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
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

fn train_direction(
    corpus: &Corpus,
    cfg: &TrainConfig,
    d: Direction,
    progress: &(dyn Fn(&Progress) + Sync),
) -> io::Result<(Option<DirModel>, DirOutput)> {
    let opts = TrainOptions {
        reverse: d.reverse(),
        ..cfg.train.clone()
    };
    let start = Instant::now();
    let mut log = Vec::new();
    let mut out = Vec::new();
    let mut trainer = Trainer::new(corpus, opts, cfg.parallel.clone(), &mut log)?;
    // The initial pass freed its per-row vectors and build arrays on many threads' malloc
    // arenas; give that memory back before the iterations (PERF-TODO 4).
    release_free_memory();
    let iters = cfg.train.iterations.max(0) as usize;
    let t = trainer.pass().n_target_tokens;
    let mut stats = Vec::with_capacity(iters);
    for iter in 0..iters {
        cancel::check();
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
    let mut subword_alignments = None;
    if let Some(sw) = &cfg.subword {
        let words = project_text(corpus, &sw.boundary, &out, d.reverse())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let raw = std::mem::replace(&mut out, words);
        if cfg.keep_subword_output {
            subword_alignments = Some(raw);
        }
    }
    let model = cfg
        .keep_models
        .then(|| DirModel::from_trainer(trainer.into_parts(), d.reverse()));
    release_free_memory();
    Ok((
        model,
        DirOutput {
            direction: d,
            alignments: out,
            subword_alignments,
            log,
            stats,
        },
    ))
}

/// Train the configured directions on `corpus` (with `Both`, one after the other unless
/// [`TrainConfig::concurrent_directions`]).
/// `progress` is called (from worker threads) after every iteration.
pub fn train(
    corpus: &Corpus,
    cfg: &TrainConfig,
    progress: &(dyn Fn(&Progress) + Sync),
) -> io::Result<Trained> {
    cancel::check();
    let (mut fwd, mut inv) = match cfg.directions {
        Directions::Both if !cfg.concurrent_directions => {
            let a = train_direction(corpus, cfg, Direction::Forward, progress)?;
            let b = train_direction(corpus, cfg, Direction::Inverse, progress)?;
            (Some(a), Some(b))
        }
        Directions::Both => {
            let (a, b) = rayon::join(
                || train_direction(corpus, cfg, Direction::Forward, progress),
                || train_direction(corpus, cfg, Direction::Inverse, progress),
            );
            (Some(a?), Some(b?))
        }
        Directions::Forward => (
            Some(train_direction(corpus, cfg, Direction::Forward, progress)?),
            None,
        ),
        Directions::Inverse => (
            None,
            Some(train_direction(corpus, cfg, Direction::Inverse, progress)?),
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
