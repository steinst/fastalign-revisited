//! PyO3 layer over fastalign-core. Exposed to Python as `fastalign._core`.
//!
//! Deliberately thin: the documented API lives in the Python package `fastalign` (`train`,
//! `Model`, ...), which normalises its inputs and calls the classes and functions here. All
//! alignment output is the Pharaoh *text* that the core produces (the same bytes as the native
//! `fastalign` binary); [`PyAlignments`] only indexes into it. Every long computation runs
//! without the GIL on a helper thread, while the calling thread polls for Ctrl-C (see
//! [`run_interruptible`]). Each call has its own cancellation token ([`Op`]), so one call's
//! interrupt or callback failure never stops another.

use std::borrow::Cow;
use std::io::Write;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastalign_core::cancel::{self, Cancel};
use fastalign_core::corpus::{Corpus, CorpusError, EmptyPairs, InputBytes, LoadOptions};
use fastalign_core::io::{Direction, Model, ModelError, SubwordInfo};
use fastalign_core::oracle::{force_align, ForceOptions, ForceResult, TextTable, TrainOptions};
use fastalign_core::pipeline::{
    symmetrize_pharaoh, with_threads, Directions, Progress, TrainConfig,
};
use fastalign_core::subword::Boundary;
use fastalign_core::symmetrize::{self, format_links, Link, Method};
use pyo3::exceptions::{
    PyFileNotFoundError, PyIndexError, PyKeyboardInterrupt, PyOSError, PyRuntimeError, PyTypeError,
    PyValueError,
};
use pyo3::prelude::*;
use pyo3::pymodule;
use pyo3::types::{PyBytes, PySlice};
use rayon::prelude::*;

type Links = Vec<Link>;

// ---------------------------------------------------------------------------------------------
// Errors

fn io_err(e: &std::io::Error, what: String) -> PyErr {
    if e.kind() == std::io::ErrorKind::NotFound {
        PyFileNotFoundError::new_err(what)
    } else {
        PyOSError::new_err(what)
    }
}

fn corpus_err(e: CorpusError) -> PyErr {
    match &e {
        CorpusError::Io(_, source) => io_err(source, e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

fn model_err(e: ModelError, path: Option<&Path>) -> PyErr {
    let at = |s: String| match path {
        Some(p) => format!("{}: {s}", p.display()),
        None => s,
    };
    match &e {
        ModelError::Io(source) => io_err(source, at(source.to_string())),
        ModelError::Format(s) => PyValueError::new_err(at(s.clone())),
        ModelError::Unsupported(s) => PyValueError::new_err(s.clone()),
    }
}

// ---------------------------------------------------------------------------------------------
// Interruptible execution

/// One Python call's long operation: its cancellation token and the first Python exception
/// raised for it (Ctrl-C seen by the polling thread, or an error from a callback).
struct Op {
    cancel: Cancel,
    error: Mutex<Option<PyErr>>,
}

impl Op {
    fn new() -> Self {
        Op {
            cancel: Cancel::new(),
            error: Mutex::new(None),
        }
    }

    /// Record `e` (the first one wins) and ask this operation, and only it, to stop.
    fn fail(&self, e: PyErr) {
        self.error.lock().unwrap().get_or_insert(e);
        self.cancel.request();
    }
}

/// Run `f` on a helper thread with the GIL released. The calling thread wakes every 50 ms to
/// check for pending signals; on Ctrl-C (or when a callback has failed the operation) the
/// operation's token is requested, and once the helper has unwound the recorded exception is
/// returned (a `KeyboardInterrupt` only if none was recorded).
fn run_interruptible<T: Send>(
    py: Python<'_>,
    op: &Op,
    f: impl FnOnce(&Cancel) -> T + Send,
) -> PyResult<T> {
    let (tx, rx) = mpsc::channel();
    let rx = Mutex::new(rx);
    let cancel = &op.cancel;
    let result = std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn_scoped(s, move || {
                let r = catch_unwind(AssertUnwindSafe(|| f(cancel)));
                let _ = tx.send(r);
            })
            .map_err(|e| PyRuntimeError::new_err(format!("cannot start a thread: {e}")))?;
        loop {
            match py.detach(|| rx.lock().unwrap().recv_timeout(Duration::from_millis(50))) {
                Ok(r) => return Ok(r),
                Err(RecvTimeoutError::Timeout) => {
                    if !cancel.is_requested() {
                        if let Err(e) = py.check_signals() {
                            op.fail(e);
                        }
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(PyRuntimeError::new_err("worker thread died"))
                }
            }
        }
    });
    let r = result?;
    let pending = op.error.lock().unwrap().take();
    match r {
        Ok(v) => match pending {
            Some(e) => Err(e),
            None => Ok(v),
        },
        Err(p) if cancel::is_cancelled(&*p) => {
            Err(pending.unwrap_or_else(|| PyKeyboardInterrupt::new_err("interrupted")))
        }
        Err(p) => resume_unwind(p),
    }
}

// ---------------------------------------------------------------------------------------------
// Inputs

/// A text argument: a file path, in-memory sentences (one whitespace-separated string each),
/// or raw bytes already read (the command line's `-`, standard input).
enum Text {
    File(PathBuf),
    Lines(Vec<String>),
    Bytes(Vec<u8>),
}

fn text_arg(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Text> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(Text::Bytes(b.as_bytes().to_vec()));
    }
    if let Ok(p) = obj.extract::<PathBuf>() {
        return Ok(Text::File(p));
    }
    obj.extract::<Vec<String>>().map(Text::Lines).map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a file path or a list of sentences"))
    })
}

impl Text {
    /// The bytes of the text, one line per sentence. A file is memory-mapped when it is a
    /// regular file and read to end of file otherwise (pipes, FIFOs).
    fn into_input(self) -> PyResult<InputBytes> {
        match self {
            Text::File(p) => InputBytes::open(&p).map_err(corpus_err),
            Text::Bytes(b) => Ok(InputBytes::from_vec(b)),
            Text::Lines(l) => {
                let mut v = Vec::with_capacity(l.iter().map(|s| s.len() + 1).sum());
                for s in &l {
                    v.extend_from_slice(s.as_bytes());
                    v.push(b'\n');
                }
                Ok(InputBytes::from_vec(v))
            }
        }
    }
}

/// Read a corpus: one `source ||| target` text (`tgt` is `None`), or two line-aligned texts
/// (each one side). Reading checks `cancel`.
fn load_corpus(
    src: Text,
    tgt: Option<Text>,
    max_len: Option<usize>,
    empty_pairs: EmptyPairs,
    cancel: &Cancel,
) -> PyResult<Corpus> {
    let opts = LoadOptions {
        max_len,
        empty_pairs,
        cancel: cancel.clone(),
        ..LoadOptions::default()
    };
    match (src, tgt) {
        (Text::Lines(_), None) => Err(PyTypeError::new_err(
            "a single text must be the path of a file of `source ||| target` lines",
        )),
        (a, None) => Corpus::from_joined_bytes(&a.into_input()?, &opts).map_err(corpus_err),
        (a, Some(b)) => {
            let (a, b) = (a.into_input()?, b.into_input()?);
            Corpus::from_pair_bytes(&a, &b, &opts).map_err(corpus_err)
        }
    }
}

fn parse_directions(s: &str) -> PyResult<Directions> {
    match s {
        "both" => Ok(Directions::Both),
        "fwd" => Ok(Directions::Forward),
        "inv" => Ok(Directions::Inverse),
        _ => Err(PyValueError::new_err(format!(
            "directions must be 'both', 'fwd' or 'inv', not '{s}'"
        ))),
    }
}

fn parse_direction(s: &str) -> PyResult<Direction> {
    match s {
        "fwd" => Ok(Direction::Forward),
        "inv" => Ok(Direction::Inverse),
        _ => Err(PyValueError::new_err(format!(
            "direction must be 'fwd' or 'inv', not '{s}'"
        ))),
    }
}

/// A symmetrisation method of two directions (`invert` is a different operation).
fn parse_method(s: &str) -> PyResult<Method> {
    let m: Method = s.parse().map_err(PyValueError::new_err)?;
    if m == Method::Invert {
        return Err(PyValueError::new_err(
            "'invert' is not a symmetrisation of two directions",
        ));
    }
    Ok(m)
}

// ---------------------------------------------------------------------------------------------
// Alignment results

/// Parse the links of one Pharaoh line, ignoring a ` ||| score` suffix.
fn parse_links(line: &[u8]) -> Links {
    let cut = line
        .windows(5)
        .position(|w| w == b" ||| ")
        .unwrap_or(line.len());
    line[..cut]
        .split(|&b| b == b' ')
        .filter(|t| !t.is_empty())
        .filter_map(|t| {
            let dash = t.iter().position(|&b| b == b'-')?;
            let num = |s: &[u8]| std::str::from_utf8(s).ok()?.parse::<u32>().ok();
            Some((num(&t[..dash])?, num(&t[dash + 1..])?))
        })
        .collect()
}

/// A read-only sequence of alignments, one list of `(source, target)` links per line, backed by
/// the Pharaoh text itself (so that writing it out gives exactly the core's bytes).
#[pyclass(name = "Alignments", module = "fastalign._core", frozen)]
struct PyAlignments {
    text: Vec<u8>,
    /// Start offset of every line, plus the end of the text.
    bounds: Vec<usize>,
}

impl PyAlignments {
    fn from_text(text: Vec<u8>) -> Self {
        let mut bounds = vec![0];
        bounds.extend(
            text.iter()
                .enumerate()
                .filter(|(_, &b)| b == b'\n')
                .map(|(k, _)| k + 1),
        );
        if *bounds.last().unwrap() != text.len() {
            bounds.push(text.len());
        }
        PyAlignments { text, bounds }
    }

    fn n(&self) -> usize {
        self.bounds.len() - 1
    }

    fn line(&self, i: usize) -> &[u8] {
        let (a, b) = (self.bounds[i], self.bounds[i + 1]);
        let l = &self.text[a..b];
        l.strip_suffix(b"\n").unwrap_or(l)
    }

    fn index(&self, i: isize) -> PyResult<usize> {
        let n = self.n() as isize;
        let j = if i < 0 { i + n } else { i };
        if (0..n).contains(&j) {
            Ok(j as usize)
        } else {
            Err(PyIndexError::new_err("alignment index out of range"))
        }
    }
}

#[pymethods]
impl PyAlignments {
    fn __len__(&self) -> usize {
        self.n()
    }

    fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(slice) = index.cast::<PySlice>() {
            let ix = slice.indices(self.n() as isize)?;
            let mut out = Vec::with_capacity(ix.slicelength);
            let mut i = ix.start;
            for _ in 0..ix.slicelength {
                out.push(parse_links(self.line(i as usize)));
                i += ix.step;
            }
            return Ok(out.into_pyobject(py)?.into_any().unbind());
        }
        let i: isize = index
            .extract()
            .map_err(|_| PyTypeError::new_err("indices must be integers or slices"))?;
        let links = parse_links(self.line(self.index(i)?));
        Ok(links.into_pyobject(py)?.into_any().unbind())
    }

    fn __iter__(slf: Py<Self>) -> AlignmentsIter {
        AlignmentsIter { src: slf, pos: 0 }
    }

    /// Pharaoh-format line `i` (`"0-0 1-2"`), without any score.
    fn pharaoh(&self, i: isize) -> PyResult<String> {
        let mut out = Vec::new();
        format_links(&parse_links(self.line(self.index(i)?)), &mut out);
        Ok(String::from_utf8(out).expect("ascii"))
    }

    /// All alignments as a list of lists of `(source, target)` pairs.
    fn to_list(&self) -> Vec<Links> {
        (0..self.n()).map(|i| parse_links(self.line(i))).collect()
    }

    /// The Pharaoh text, exactly as written by [`write`](Self::write).
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.text)
    }

    /// Write the alignments in Pharaoh format, one line per sentence pair.
    fn write(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| write_file(&path, &self.text))
    }
}

#[pyclass(module = "fastalign._core")]
struct AlignmentsIter {
    src: Py<PyAlignments>,
    pos: usize,
}

#[pymethods]
impl AlignmentsIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<Links> {
        let a = self.src.get();
        if self.pos >= a.n() {
            return None;
        }
        let _ = py;
        let l = parse_links(a.line(self.pos));
        self.pos += 1;
        Some(l)
    }
}

fn write_file(path: &Path, bytes: &[u8]) -> PyResult<()> {
    fastalign_core::atomic::write_atomic(path, bytes)
        .map_err(|e| io_err(&e, format!("{}: {e}", path.display())))
}

// ---------------------------------------------------------------------------------------------
// Model

type SubwordTuple<'py> = (
    String,
    String,
    Vec<Bound<'py, PyBytes>>,
    Bound<'py, PyBytes>,
);

/// The `subword` argument of `train`: `(kind, marker, tokenizer models, settings)`.
type SubwordArg = (String, Vec<u8>, Vec<Vec<u8>>, Vec<u8>);

/// What training produced besides the model.
struct TrainData {
    fwd: Option<Vec<u8>>,
    inv: Option<Vec<u8>>,
    /// Piece-level alignments (subword training with `subword_output`).
    fwd_pieces: Option<Vec<u8>>,
    inv_pieces: Option<Vec<u8>>,
    fwd_log: Vec<u8>,
    inv_log: Vec<u8>,
    /// Number of lines of the input (alignment texts have one line per input line).
    n_lines: usize,
    /// `Trained::warnings` of the run (no pairs to train on, nonfinite parameters).
    warnings: Vec<String>,
}

struct Inner {
    model: Model,
    train: Option<TrainData>,
    /// Trained for its output files only (`train(keep_models=False)`): the directions' models
    /// were dropped as soon as each finished, so only the training alignments and logs exist.
    outputs_only: bool,
    /// The worker budget of the training that produced the model (0 = all cores): the
    /// symmetrisation of its training alignments runs under it.
    threads: usize,
}

enum Sel {
    Dir(Direction),
    Sym(Method),
}

impl Inner {
    fn has(&self, d: Direction) -> bool {
        if self.outputs_only {
            return self.train.as_ref().is_some_and(|t| match d {
                Direction::Forward => t.fwd.is_some(),
                Direction::Inverse => t.inv.is_some(),
            });
        }
        self.model.dir(d).is_some()
    }

    /// An error unless the model parameters exist (not an outputs-only training result).
    fn require_model(&self) -> PyResult<()> {
        if self.outputs_only {
            return Err(PyValueError::new_err(
                "this model was trained without keeping the translation tables (outputs only): \
                 it holds the training alignments but cannot be saved or used to align",
            ));
        }
        Ok(())
    }

    /// Resolve the `symmetrize` / `direction` arguments (see `fastalign.Model.alignments`).
    fn select(&self, symmetrize: Option<&str>, direction: Option<&str>) -> PyResult<Sel> {
        let both = self.has(Direction::Forward) && self.has(Direction::Inverse);
        match (symmetrize, direction) {
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "give either symmetrize or direction, not both",
            )),
            (Some(m), None) => {
                if !both {
                    return Err(PyValueError::new_err(
                        "symmetrisation needs a model trained in both directions",
                    ));
                }
                Ok(Sel::Sym(parse_method(m)?))
            }
            (None, Some(d)) => {
                let d = parse_direction(d)?;
                if !self.has(d) {
                    return Err(PyValueError::new_err(format!(
                        "the model has no '{}' direction",
                        d.name()
                    )));
                }
                Ok(Sel::Dir(d))
            }
            (None, None) => {
                if both {
                    Ok(Sel::Sym(Method::GrowDiagFinalAnd))
                } else if self.has(Direction::Forward) {
                    Ok(Sel::Dir(Direction::Forward))
                } else {
                    Ok(Sel::Dir(Direction::Inverse))
                }
            }
        }
    }

    fn train_data(&self) -> PyResult<&TrainData> {
        self.train.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "this model was loaded from a file and holds no training alignments; \
                 use align() or align_many() to align sentences with it",
            )
        })
    }

    /// The Pharaoh text of the training alignments for a selection.
    fn training_text(&self, sel: &Sel, pieces: bool) -> PyResult<Cow<'_, [u8]>> {
        let td = self.train_data()?;
        if pieces {
            let raw = match sel {
                Sel::Dir(Direction::Forward) => td.fwd_pieces.as_ref(),
                Sel::Dir(Direction::Inverse) => td.inv_pieces.as_ref(),
                Sel::Sym(_) => {
                    return Err(PyValueError::new_err(
                        "piece-level alignments are not symmetrised: give direction=",
                    ))
                }
            };
            return raw.map(|r| Cow::Borrowed(r.as_slice())).ok_or_else(|| {
                PyValueError::new_err(
                    "this model holds no piece-level alignments (train with subword and \
                     subword_output=True)",
                )
            });
        }
        let dir = |d: Direction| match d {
            Direction::Forward => td.fwd.as_ref(),
            Direction::Inverse => td.inv.as_ref(),
        };
        match sel {
            Sel::Dir(d) => Ok(Cow::Borrowed(
                dir(*d).expect("direction checked").as_slice(),
            )),
            Sel::Sym(m) => {
                let (f, i) = (
                    dir(Direction::Forward).expect("both directions checked"),
                    dir(Direction::Inverse).expect("both directions checked"),
                );
                with_threads(self.threads, || symmetrize_pharaoh(f, i, *m))
                    .map_err(PyRuntimeError::new_err)?
                    .map(Cow::Owned)
                    .map_err(PyValueError::new_err)
            }
        }
    }
}

/// A trained model (both directions, or one) and, right after training, the alignments of the
/// training corpus.
#[pyclass(name = "Model", module = "fastalign._core", frozen)]
struct PyModel {
    inner: Arc<Inner>,
}

#[pymethods]
impl PyModel {
    /// Directions the model holds: a subset of `("fwd", "inv")`.
    #[getter]
    fn directions(&self) -> Vec<&'static str> {
        [Direction::Forward, Direction::Inverse]
            .into_iter()
            .filter(|&d| self.inner.has(d))
            .map(Direction::name)
            .collect()
    }

    /// Number of distinct words (one dictionary is shared by both languages).
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.model.vocab.len().saturating_sub(3)
    }

    #[getter]
    fn has_training_alignments(&self) -> bool {
        self.inner.train.is_some()
    }

    /// Number of input lines of the training corpus, or `None` for a loaded model.
    #[getter]
    fn n_lines(&self) -> Option<usize> {
        self.inner.train.as_ref().map(|t| t.n_lines)
    }

    /// Warnings of the training run (nothing to train on, nonfinite parameters); empty for a
    /// loaded model.
    #[getter]
    fn training_warnings(&self) -> Vec<String> {
        self.inner
            .train
            .as_ref()
            .map_or_else(Vec::new, |t| t.warnings.clone())
    }

    /// The segmentation the model was trained with, as `(kind, marker, tokenizers, settings)`
    /// (`kind` is `"start"` or `"continue"`), or `None` for a word-level model.
    #[getter]
    fn subword<'py>(&self, py: Python<'py>) -> Option<SubwordTuple<'py>> {
        self.inner.model.subword.as_ref().map(|sw| {
            (
                sw.boundary.kind().to_owned(),
                String::from_utf8_lossy(sw.boundary.marker()).into_owned(),
                sw.tokenizers.iter().map(|t| PyBytes::new(py, t)).collect(),
                PyBytes::new(py, &sw.settings),
            )
        })
    }

    /// Final parameters of one direction: `(tension, p0, iterations, mean_srclen_multiplier)`.
    fn parameters(&self, direction: &str) -> PyResult<(f64, f64, u32, f64)> {
        self.inner.require_model()?;
        let d = parse_direction(direction)?;
        let dm = self.inner.model.dir(d).ok_or_else(|| {
            PyValueError::new_err(format!("the model has no '{}' direction", d.name()))
        })?;
        let o = &dm.opts;
        Ok((o.tension, o.p0, o.iterations, o.mean_srclen_multiplier))
    }

    /// fast_align's statistics (its stderr, without the option echo) for one direction of the
    /// training run.
    fn training_log<'py>(&self, py: Python<'py>, direction: &str) -> PyResult<Bound<'py, PyBytes>> {
        let d = parse_direction(direction)?;
        let td = self.inner.train_data()?;
        if !self.inner.has(d) {
            return Err(PyValueError::new_err(format!(
                "the model has no '{}' direction",
                d.name()
            )));
        }
        let log = match d {
            Direction::Forward => &td.fwd_log,
            Direction::Inverse => &td.inv_log,
        };
        Ok(PyBytes::new(py, log))
    }

    /// Write the model to `path`: staged in a uniquely named temporary file in the same
    /// directory and renamed over `path` when complete, so `path` is never left partial.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let inner = &self.inner;
        inner.require_model()?;
        py.detach(|| inner.model.save(&path))
            .map_err(|e| model_err(e, Some(&path)))
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<PyModel> {
        let model = py
            .detach(|| Model::load(&path))
            .map_err(|e| model_err(e, Some(&path)))?;
        Ok(PyModel {
            inner: Arc::new(Inner {
                model,
                train: None,
                outputs_only: false,
                threads: 0,
            }),
        })
    }

    /// Write one direction's table in fast_align's `-p` text format.
    #[pyo3(signature = (path, direction="fwd", beam_threshold=-4.0, threads=0))]
    fn write_table(
        &self,
        py: Python<'_>,
        path: PathBuf,
        direction: &str,
        beam_threshold: f64,
        threads: usize,
    ) -> PyResult<()> {
        let d = parse_direction(direction)?;
        self.inner.require_model()?;
        let model = &self.inner.model;
        if model.dir(d).is_none() {
            return Err(PyValueError::new_err(format!(
                "the model has no '{}' direction",
                d.name()
            )));
        }
        let op = Op::new();
        run_interruptible(py, &op, |cancel| -> PyResult<()> {
            // streamed into the staged file: the table text is never held whole
            with_threads(threads, || {
                fastalign_core::atomic::write_with(&path, |w| {
                    model
                        .write_table(d, beam_threshold, cancel, w)
                        .map_err(|e| model_err(e, Some(&path)))
                })
            })
            .map_err(PyRuntimeError::new_err)?
        })?
    }

    /// Alignments of the training corpus, one entry per input line.
    #[pyo3(signature = (symmetrize=None, direction=None, pieces=false))]
    fn training_alignments(
        &self,
        py: Python<'_>,
        symmetrize: Option<&str>,
        direction: Option<&str>,
        pieces: bool,
    ) -> PyResult<PyAlignments> {
        let inner = &self.inner;
        let sel = inner.select(symmetrize, direction)?;
        inner.train_data()?;
        // the materialising API: the Alignments object owns its text
        let text = py.detach(|| inner.training_text(&sel, pieces).map(Cow::into_owned))?;
        Ok(PyAlignments::from_text(text))
    }

    /// Write the training alignments to `path` in Pharaoh format.
    #[pyo3(signature = (path, symmetrize=None, direction=None, pieces=false))]
    fn write_training_alignments(
        &self,
        py: Python<'_>,
        path: PathBuf,
        symmetrize: Option<&str>,
        direction: Option<&str>,
        pieces: bool,
    ) -> PyResult<()> {
        let inner = &self.inner;
        let sel = inner.select(symmetrize, direction)?;
        inner.train_data()?;
        py.detach(|| write_file(&path, &inner.training_text(&sel, pieces)?))
    }

    #[allow(clippy::too_many_arguments)]
    /// Forced alignment of new sentence pairs (file paths or lists of sentences; with `tgt`
    /// omitted, `src` is a `source ||| target` file).
    #[pyo3(signature = (src, tgt=None, symmetrize=None, direction=None, threads=0, scores=false))]
    fn align_many(
        &self,
        py: Python<'_>,
        src: &Bound<'_, PyAny>,
        tgt: Option<&Bound<'_, PyAny>>,
        symmetrize: Option<&str>,
        direction: Option<&str>,
        threads: usize,
        scores: bool,
    ) -> PyResult<PyAlignments> {
        let inner = &self.inner;
        inner.require_model()?;
        let sel = inner.select(symmetrize, direction)?;
        if scores && matches!(sel, Sel::Sym(_)) {
            return Err(PyValueError::new_err(
                "scores belong to one direction: give direction= instead of symmetrize=",
            ));
        }
        let src = text_arg(src, "src")?;
        let tgt = tgt.map(|t| text_arg(t, "tgt")).transpose()?;
        let op = Op::new();
        let text = run_interruptible(py, &op, |cancel| -> PyResult<Vec<u8>> {
            with_threads(threads, || {
                let corpus = load_corpus(src, tgt, None, EmptyPairs::Keep, cancel)?;
                // one mapping of the new words to the model's (cached) index for both directions
                let mapped = inner.model.map_corpus(&corpus);
                let align = |d| {
                    inner
                        .model
                        .align_words_mapped(&mapped, d, scores, cancel)
                        .map_err(|e| model_err(e, None))
                };
                match sel {
                    Sel::Dir(d) => align(d),
                    Sel::Sym(m) => {
                        let f = align(Direction::Forward)?;
                        let i = align(Direction::Inverse)?;
                        symmetrize_pharaoh(&f, &i, m).map_err(PyValueError::new_err)
                    }
                }
            })
            .map_err(PyRuntimeError::new_err)?
        })??;
        Ok(PyAlignments::from_text(text))
    }

    /// Forced alignment of one sentence pair, each side one line of tokens separated by spaces
    /// (pieces for a subword model): the links `align_many` gives for a one-line input, computed
    /// directly (no corpus, no thread, no Pharaoh text) with the model's cached word index.
    #[pyo3(signature = (src, tgt, symmetrize=None, direction=None))]
    fn align_pair(
        &self,
        py: Python<'_>,
        src: &str,
        tgt: &str,
        symmetrize: Option<&str>,
        direction: Option<&str>,
    ) -> PyResult<Links> {
        let inner = &self.inner;
        inner.require_model()?;
        let sel = inner.select(symmetrize, direction)?;
        py.detach(|| {
            let (a, b) = (line_words(src), line_words(tgt));
            let align = |d| {
                inner
                    .model
                    .align_tokens_words(d, &a, &b)
                    .map_err(|e| model_err(e, None))
            };
            match sel {
                Sel::Dir(d) => align(d),
                Sel::Sym(m) => {
                    // as `symmetrize_pharaoh` reads each direction's line: sorted, no duplicates
                    let (mut f, mut i) = (align(Direction::Forward)?, align(Direction::Inverse)?);
                    for l in [&mut f, &mut i] {
                        l.sort_unstable();
                        l.dedup();
                    }
                    Ok(symmetrize::symmetrize(&f, &i, m))
                }
            }
        })
    }

    /// Whether the model stores a tokeniser (a non-empty blob), without copying the blobs.
    #[getter]
    fn has_tokenizer(&self) -> bool {
        self.inner
            .model
            .subword
            .as_ref()
            .is_some_and(|sw| sw.tokenizers.iter().any(|t| !t.is_empty()))
    }
}

/// The words of one side's line, as the core's paired reader splits them (space and tab).
fn line_words(line: &str) -> Vec<&[u8]> {
    line.as_bytes()
        .split(|&b| b == b' ' || b == b'\t')
        .filter(|w| !w.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Training

/// A trained direction's alignment text (one line per input line, or none without a decoding
/// iteration) and statistics log.
fn split_output(o: Option<fastalign_core::pipeline::DirOutput>) -> (Option<Vec<u8>>, Vec<u8>) {
    match o {
        Some(o) => (Some(o.alignments), o.log),
        None => (None, Vec::new()),
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (src, tgt, threads, directions, iterations, diagonal, optimize_tension,
                    variational_bayes, alpha, tension, p0, null, buffer_size, max_len,
                    empty_pairs, scores,
                    progress=None, lookup_index=true, subword=None, subword_output=false,
                    keep_models=true, table_files=None, table_beam=-4.0, offset_cache_mb=0))]
fn train(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    tgt: Option<&Bound<'_, PyAny>>,
    threads: usize,
    directions: &str,
    iterations: i32,
    diagonal: bool,
    optimize_tension: bool,
    variational_bayes: bool,
    alpha: f64,
    tension: f64,
    p0: f64,
    null: bool,
    buffer_size: u64,
    max_len: Option<usize>,
    empty_pairs: &str,
    scores: bool,
    progress: Option<Py<PyAny>>,
    lookup_index: bool,
    subword: Option<SubwordArg>,
    subword_output: bool,
    keep_models: bool,
    table_files: Option<Vec<(String, PathBuf)>>,
    table_beam: f64,
    offset_cache_mb: usize,
) -> PyResult<PyModel> {
    let empty_pairs = EmptyPairs::parse(empty_pairs).ok_or_else(|| {
        PyValueError::new_err(format!(
            "empty_pairs must be 'keep', 'skip' or 'error', not {empty_pairs:?}"
        ))
    })?;
    if diagonal {
        fastalign_core::math::check_tension(tension).map_err(PyValueError::new_err)?;
    }
    let subword = subword
        .map(
            |(kind, marker, tokenizers, settings)| -> PyResult<SubwordInfo> {
                Ok(SubwordInfo {
                    boundary: Boundary::from_parts(&kind, &marker)
                        .map_err(PyValueError::new_err)?,
                    tokenizers,
                    settings,
                })
            },
        )
        .transpose()?;
    let cfg = TrainConfig {
        train: TrainOptions {
            reverse: false,
            iterations,
            favor_diagonal: diagonal,
            optimize_tension,
            variational_bayes,
            alpha,
            use_null: null,
            p0,
            tension,
            buffer_size,
            print_scores: scores,
        },
        directions: parse_directions(directions)?,
        subword,
        keep_subword_output: subword_output,
        keep_models,
        ..TrainConfig::default()
    };
    let table_files = table_files
        .unwrap_or_default()
        .into_iter()
        .map(|(d, p)| Ok((parse_direction(&d)?, p)))
        .collect::<PyResult<Vec<_>>>()?;
    let mut cfg = cfg;
    if !lookup_index {
        cfg.parallel.dense_index_entries = 0;
    }
    cfg.parallel.offset_cache_bytes = offset_cache_mb.saturating_mul(1 << 20);
    let src = text_arg(src, "src")?;
    let tgt = tgt.map(|t| text_arg(t, "tgt")).transpose()?;

    let op = Op::new();
    let callback = |p: &Progress| {
        let Some(cb) = &progress else { return };
        Python::attach(|py| {
            let args = (
                p.direction.name(),
                p.iteration,
                p.iterations,
                p.log_likelihood,
                p.perplexity,
                p.p0,
                p.tension,
                p.elapsed.as_secs_f64(),
            );
            if let Err(e) = cb.call1(py, args) {
                op.fail(e);
            }
        });
    };
    let cfg_ref = &cfg;
    let out = run_interruptible(py, &op, |cancel| -> PyResult<PyModel> {
        with_threads(threads, || {
            let corpus = load_corpus(src, tgt, max_len, empty_pairs, cancel)?;
            // tables are written as each direction finishes (its model can then be dropped) into
            // staged files, published once the whole training has succeeded
            let mut staged: Vec<fastalign_core::atomic::AtomicFile> = Vec::new();
            let mut sink = |f: &fastalign_core::pipeline::Finished<'_>| -> std::io::Result<()> {
                for (d, path) in &table_files {
                    if *d == f.direction {
                        let mut w = fastalign_core::atomic::AtomicFile::create(path)?;
                        f.model.write_table(f.vocab, table_beam, cancel, &mut w)?;
                        staged.push(w);
                    }
                }
                Ok(())
            };
            let sink_ref: Option<fastalign_core::pipeline::Sink<'_>> =
                (!table_files.is_empty()).then_some(&mut sink);
            let trained = fastalign_core::pipeline::train_with_sink(
                &corpus, cfg_ref, &callback, cancel, sink_ref,
            )
            .map_err(|e| io_err(&e, e.to_string()))?;
            for w in staged {
                w.commit()?;
            }
            let n_lines = corpus.n_input_lines();
            let warnings = trained.warnings(&corpus);
            let fastalign_core::pipeline::Trained {
                model,
                mut fwd,
                mut inv,
            } = trained;
            let (fwd_pieces, inv_pieces) = (
                fwd.as_mut().and_then(|o| o.subword_alignments.take()),
                inv.as_mut().and_then(|o| o.subword_alignments.take()),
            );
            let (fwd_text, fwd_log) = split_output(fwd);
            let (inv_text, inv_log) = split_output(inv);
            let data = TrainData {
                fwd: fwd_text,
                inv: inv_text,
                fwd_pieces,
                inv_pieces,
                fwd_log,
                inv_log,
                n_lines,
                warnings,
            };
            Ok(PyModel {
                inner: Arc::new(Inner {
                    model,
                    train: Some(data),
                    outputs_only: !keep_models,
                    threads,
                }),
            })
        })
        .map_err(PyRuntimeError::new_err)?
    })??;
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// fast_align's `-f` mode, symmetrisation

/// fast_align's `-f` mode with a `-p` text table. Returns `(output, log, failed_line)`:
/// `SRC ||| TGT ||| LINKS ||| SCORE` lines, fast_align's stderr text, and the 1-based number of
/// the line with an empty side that stopped the run (fast_align exits with status 1 there).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn align_table<'py>(
    py: Python<'py>,
    table: PathBuf,
    src: &Bound<'_, PyAny>,
    tgt: Option<&Bound<'_, PyAny>>,
    reverse: bool,
    diagonal: bool,
    null: bool,
    p0: f64,
    tension: f64,
    mean_length_multiplier: f64,
    threads: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Option<usize>)> {
    if diagonal {
        fastalign_core::math::check_tension(tension).map_err(PyValueError::new_err)?;
    }
    let src = text_arg(src, "src")?;
    let tgt = tgt.map(|t| text_arg(t, "tgt")).transpose()?;
    let op = Op::new();
    let (out, log, failed) = run_interruptible(py, &op, |cancel| -> PyResult<_> {
        with_threads(threads, || {
            let corpus = load_corpus(src, tgt, None, EmptyPairs::Keep, cancel)?;
            let bytes = std::fs::read(&table)
                .map_err(|e| io_err(&e, format!("{}: {e}", table.display())))?;
            let table = TextTable::parse_with(&bytes, &corpus.vocab, cancel);
            let mut log = Vec::new();
            writeln!(log, "Loaded {} translation parameters.", table.loaded).unwrap();
            let opts = ForceOptions {
                reverse,
                favor_diagonal: diagonal,
                use_null: null,
                p0,
                tension,
                mean_srclen_multiplier: mean_length_multiplier,
            };
            let mut out = Vec::new();
            let res = force_align(&corpus, &table, &opts, cancel, &mut log, &mut out)
                .map_err(|e| PyOSError::new_err(e.to_string()))?;
            let failed = match res {
                ForceResult::Done(_) => None,
                ForceResult::EmptySide(n) => Some(n),
            };
            Ok((out, log, failed))
        })
        .map_err(PyRuntimeError::new_err)?
    })??;
    Ok((PyBytes::new(py, &out), PyBytes::new(py, &log), failed))
}

/// Symmetrise directional alignments pair by pair, as `atools -c METHOD`. `fwd` and `inv` hold
/// links in `(source, target)` orientation. Batches of pairs are symmetrised in parallel on
/// `threads` workers (0 = all cores); the result does not depend on it.
#[pyfunction]
#[pyo3(signature = (fwd, inv, method, threads=0))]
fn symmetrize_many(
    py: Python<'_>,
    fwd: Vec<Links>,
    inv: Vec<Links>,
    method: &str,
    threads: usize,
) -> PyResult<Vec<Links>> {
    let method = parse_method(method)?;
    if fwd.len() != inv.len() {
        return Err(PyValueError::new_err(format!(
            "fwd and inv must have the same size ({} and {})",
            fwd.len(),
            inv.len()
        )));
    }
    // enough pairs per task that a task outweighs scheduling it
    const BATCH: usize = 256;
    py.detach(|| {
        with_threads(threads, || {
            fwd.into_par_iter()
                .zip(inv.into_par_iter())
                .with_min_len(BATCH)
                .map(|(a, b)| symmetrize::symmetrize(&a, &b, method))
                .collect::<Vec<_>>()
        })
    })
    .map_err(PyRuntimeError::new_err)
}

/// Symmetrise two Pharaoh texts line by line, exactly as `atools -c METHOD -i FWD -j INV` (only
/// `fwd` is used for `invert`), on `threads` workers (0 = all cores).
#[pyfunction]
#[pyo3(signature = (fwd, inv, method, threads=0))]
fn symmetrize_text<'py>(
    py: Python<'py>,
    fwd: &[u8],
    inv: Option<&[u8]>,
    method: &str,
    threads: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let method: Method = method.parse().map_err(PyValueError::new_err)?;
    let mut out = Vec::new();
    py.detach(|| {
        with_threads(threads, || {
            symmetrize::symmetrize_text(fwd, inv, method, &mut out).map_err(|e| e.to_string())
        })
        .and_then(|r| r)
    })
    .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &out))
}

/// Swap the two sides of alignments (atools `invert`).
#[pyfunction]
fn invert_many(fwd: Vec<Links>) -> Vec<Links> {
    fwd.iter().map(|a| symmetrize::invert(a)).collect()
}

/// Parse a Pharaoh alignment line in any layout the core accepts (links, `links ||| score`,
/// atools-prefixed); sorted, no duplicates.
#[pyfunction]
fn parse_pharaoh(line: &str) -> PyResult<Links> {
    let text = line.strip_suffix('\n').unwrap_or(line);
    symmetrize::parse_line(text.as_bytes()).ok_or_else(|| {
        PyValueError::new_err(format!(
            "bad alignment line {line:?} (expected 'i-j' links)"
        ))
    })
}

#[pymodule]
mod _core {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::align_table;
    #[pymodule_export]
    use super::invert_many;
    #[pymodule_export]
    use super::parse_pharaoh;
    #[pymodule_export]
    use super::symmetrize_many;
    #[pymodule_export]
    use super::symmetrize_text;
    #[pymodule_export]
    use super::train;
    #[pymodule_export]
    use super::AlignmentsIter;
    #[pymodule_export]
    use super::PyAlignments;
    #[pymodule_export]
    use super::PyModel;

    /// Version of the Rust core, from Cargo metadata.
    #[pyfunction]
    fn version() -> &'static str {
        fastalign_core::version()
    }
}
