//! PyO3 layer over fastalign-core. Exposed to Python as `fastalign._core`.
//!
//! Deliberately thin: the documented API lives in the Python package `fastalign` (`train`,
//! `Model`, ...), which normalises its inputs and calls the classes and functions here. All
//! alignment output is the Pharaoh *text* that the core produces (the same bytes as the native
//! `fastalign` binary); [`PyAlignments`] only indexes into it. Every long computation runs
//! without the GIL on a helper thread, while the calling thread polls for Ctrl-C (see
//! [`run_interruptible`]).

use std::io::Write;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastalign_core::cancel;
use fastalign_core::corpus::{Corpus, CorpusError, LoadOptions};
use fastalign_core::io::{Direction, Model, ModelError, SubwordInfo};
use fastalign_core::oracle::{force_align, ForceOptions, ForceResult, TextTable, TrainOptions};
use fastalign_core::pipeline::{
    symmetrize_pharaoh, train as core_train, with_threads, Directions, Progress, TrainConfig,
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

/// Run `f` on a helper thread with the GIL released. The calling thread wakes every 50 ms to
/// check for pending signals; on Ctrl-C (or when `slot` receives an error from a callback) it
/// asks the computation to cancel and returns that error once the helper has unwound.
fn run_interruptible<T: Send>(
    py: Python<'_>,
    slot: &Mutex<Option<PyErr>>,
    f: impl FnOnce() -> T + Send,
) -> PyResult<T> {
    cancel::reset();
    let (tx, rx) = mpsc::channel();
    let rx = Mutex::new(rx);
    let result = std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn_scoped(s, move || {
                let r = catch_unwind(AssertUnwindSafe(f));
                let _ = tx.send(r);
            })
            .map_err(|e| PyRuntimeError::new_err(format!("cannot start a thread: {e}")))?;
        loop {
            match py.detach(|| rx.lock().unwrap().recv_timeout(Duration::from_millis(50))) {
                Ok(r) => return Ok(r),
                Err(RecvTimeoutError::Timeout) => {
                    if !cancel::is_requested() {
                        if let Err(e) = py.check_signals() {
                            slot.lock().unwrap().get_or_insert(e);
                            cancel::request();
                        }
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(PyRuntimeError::new_err("worker thread died"))
                }
            }
        }
    });
    cancel::reset();
    let r = result?;
    let pending = slot.lock().unwrap().take();
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

/// A text argument: a file path or in-memory sentences (one whitespace-separated string each).
enum Text {
    File(PathBuf),
    Lines(Vec<String>),
}

fn text_arg(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Text> {
    if let Ok(p) = obj.extract::<PathBuf>() {
        return Ok(Text::File(p));
    }
    obj.extract::<Vec<String>>().map(Text::Lines).map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a file path or a list of sentences"))
    })
}

impl Text {
    /// The bytes of the text, one line per sentence.
    fn into_bytes(self) -> PyResult<Vec<u8>> {
        match self {
            Text::File(p) => {
                std::fs::read(&p).map_err(|e| io_err(&e, format!("{}: {e}", p.display())))
            }
            Text::Lines(l) => {
                let mut v = Vec::with_capacity(l.iter().map(|s| s.len() + 1).sum());
                for s in &l {
                    v.extend_from_slice(s.as_bytes());
                    v.push(b'\n');
                }
                Ok(v)
            }
        }
    }
}

/// Read a corpus: one `source ||| target` file (`tgt` is `None`), or two line-aligned texts.
fn load_corpus(src: Text, tgt: Option<Text>, max_len: Option<usize>) -> PyResult<Corpus> {
    let opts = LoadOptions {
        max_len,
        ..LoadOptions::default()
    };
    match (src, tgt) {
        (Text::File(a), None) => Corpus::read_joined(&a, &opts).map_err(corpus_err),
        (Text::Lines(_), None) => Err(PyTypeError::new_err(
            "a single text must be the path of a file of `source ||| target` lines",
        )),
        (Text::File(a), Some(Text::File(b))) => {
            Corpus::read_pair(&a, &b, &opts).map_err(corpus_err)
        }
        (a, Some(b)) => {
            let (a, b) = (a.into_bytes()?, b.into_bytes()?);
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
    std::fs::write(path, bytes).map_err(|e| io_err(&e, format!("{}: {e}", path.display())))
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
}

struct Inner {
    model: Model,
    train: Option<TrainData>,
}

enum Sel {
    Dir(Direction),
    Sym(Method),
}

impl Inner {
    fn has(&self, d: Direction) -> bool {
        self.model.dir(d).is_some()
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
    fn training_text(&self, sel: &Sel, pieces: bool) -> PyResult<Vec<u8>> {
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
            return raw.cloned().ok_or_else(|| {
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
            Sel::Dir(d) => Ok(dir(*d).expect("direction checked").clone()),
            Sel::Sym(m) => symmetrize_pharaoh(
                dir(Direction::Forward).expect("both directions checked"),
                dir(Direction::Inverse).expect("both directions checked"),
                *m,
            )
            .map_err(PyValueError::new_err),
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

    /// Write the model to `path` (through a temporary file, renamed when complete).
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let inner = &self.inner;
        let mut tmp = path.clone().into_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        py.detach(|| -> PyResult<()> {
            let r = inner.model.save(&tmp);
            if let Err(e) = r {
                let _ = std::fs::remove_file(&tmp);
                return Err(model_err(e, Some(&path)));
            }
            std::fs::rename(&tmp, &path).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                io_err(&e, format!("{}: {e}", path.display()))
            })
        })
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<PyModel> {
        let model = py
            .detach(|| Model::load(&path))
            .map_err(|e| model_err(e, Some(&path)))?;
        Ok(PyModel {
            inner: Arc::new(Inner { model, train: None }),
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
        let model = &self.inner.model;
        if model.dir(d).is_none() {
            return Err(PyValueError::new_err(format!(
                "the model has no '{}' direction",
                d.name()
            )));
        }
        let slot = Mutex::new(None);
        run_interruptible(py, &slot, || -> PyResult<()> {
            let mut buf = Vec::new();
            with_threads(threads, || model.write_table(d, beam_threshold, &mut buf))
                .map_err(|e| model_err(e, None))?;
            write_file(&path, &buf)
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
        let text = py.detach(|| inner.training_text(&sel, pieces))?;
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
        let sel = inner.select(symmetrize, direction)?;
        if scores && matches!(sel, Sel::Sym(_)) {
            return Err(PyValueError::new_err(
                "scores belong to one direction: give direction= instead of symmetrize=",
            ));
        }
        let src = text_arg(src, "src")?;
        let tgt = tgt.map(|t| text_arg(t, "tgt")).transpose()?;
        let slot = Mutex::new(None);
        let text = run_interruptible(py, &slot, || -> PyResult<Vec<u8>> {
            with_threads(threads, || {
                let corpus = load_corpus(src, tgt, None)?;
                let align = |d| {
                    inner
                        .model
                        .align_words(&corpus, d, scores)
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
        })??;
        Ok(PyAlignments::from_text(text))
    }
}

// ---------------------------------------------------------------------------------------------
// Training

/// A trained direction's alignment text (one line per input line) and statistics log.
fn split_output(
    o: Option<fastalign_core::pipeline::DirOutput>,
    corpus: &Corpus,
) -> (Option<Vec<u8>>, Vec<u8>) {
    match o {
        Some(o) => (
            Some(match corpus.expand_lines(&o.alignments) {
                std::borrow::Cow::Borrowed(_) => o.alignments,
                std::borrow::Cow::Owned(v) => v,
            }),
            o.log,
        ),
        None => (None, Vec::new()),
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (src, tgt, threads, directions, iterations, diagonal, optimize_tension,
                    variational_bayes, alpha, tension, p0, null, buffer_size, max_len, scores,
                    progress=None, lookup_index=true, subword=None, subword_output=false))]
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
    scores: bool,
    progress: Option<Py<PyAny>>,
    lookup_index: bool,
    subword: Option<SubwordArg>,
    subword_output: bool,
) -> PyResult<PyModel> {
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
        ..TrainConfig::default()
    };
    let mut cfg = cfg;
    if !lookup_index {
        cfg.parallel.dense_index_entries = 0;
    }
    let src = text_arg(src, "src")?;
    let tgt = tgt.map(|t| text_arg(t, "tgt")).transpose()?;

    let slot: Mutex<Option<PyErr>> = Mutex::new(None);
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
                slot.lock().unwrap().get_or_insert(e);
                cancel::request();
            }
        });
    };
    let cfg_ref = &cfg;
    let out = run_interruptible(py, &slot, || -> PyResult<PyModel> {
        with_threads(threads, || {
            let corpus = load_corpus(src, tgt, max_len)?;
            if corpus.is_empty() {
                return Err(PyValueError::new_err("the corpus has no sentence pairs"));
            }
            let trained =
                core_train(&corpus, cfg_ref, &callback).map_err(|e| io_err(&e, e.to_string()))?;
            let n_lines = corpus.n_input_lines();
            let fastalign_core::pipeline::Trained { model, fwd, inv } = trained;
            let (fwd_pieces, inv_pieces) = (
                fwd.as_ref().and_then(|o| o.subword_alignments.clone()),
                inv.as_ref().and_then(|o| o.subword_alignments.clone()),
            );
            let expand = |t: Option<Vec<u8>>| {
                t.map(|t| match corpus.expand_lines(&t) {
                    std::borrow::Cow::Borrowed(_) => t,
                    std::borrow::Cow::Owned(v) => v,
                })
            };
            let (fwd_pieces, inv_pieces) = (expand(fwd_pieces), expand(inv_pieces));
            let (fwd_text, fwd_log) = split_output(fwd, &corpus);
            let (inv_text, inv_log) = split_output(inv, &corpus);
            let data = TrainData {
                fwd: fwd_text,
                inv: inv_text,
                fwd_pieces,
                inv_pieces,
                fwd_log,
                inv_log,
                n_lines,
            };
            Ok(PyModel {
                inner: Arc::new(Inner {
                    model,
                    train: Some(data),
                }),
            })
        })
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
    let src = text_arg(src, "src")?;
    let tgt = tgt.map(|t| text_arg(t, "tgt")).transpose()?;
    let slot = Mutex::new(None);
    let (out, log, failed) = run_interruptible(py, &slot, || -> PyResult<_> {
        with_threads(threads, || {
            let corpus = load_corpus(src, tgt, None)?;
            let bytes = std::fs::read(&table)
                .map_err(|e| io_err(&e, format!("{}: {e}", table.display())))?;
            let table = TextTable::parse(&bytes, &corpus.vocab);
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
            let res = force_align(&corpus, &table, &opts, &mut log, &mut out)
                .map_err(|e| PyOSError::new_err(e.to_string()))?;
            let failed = match res {
                ForceResult::Done(_) => None,
                ForceResult::EmptySide(n) => Some(n),
            };
            Ok((out, log, failed))
        })
    })??;
    Ok((PyBytes::new(py, &out), PyBytes::new(py, &log), failed))
}

/// Symmetrise directional alignments pair by pair, as `atools -c METHOD`. `fwd` and `inv` hold
/// links in `(source, target)` orientation.
#[pyfunction]
fn symmetrize_many(
    py: Python<'_>,
    fwd: Vec<Links>,
    inv: Vec<Links>,
    method: &str,
) -> PyResult<Vec<Links>> {
    let method = parse_method(method)?;
    if fwd.len() != inv.len() {
        return Err(PyValueError::new_err(format!(
            "fwd and inv must have the same size ({} and {})",
            fwd.len(),
            inv.len()
        )));
    }
    py.detach(|| {
        fwd.into_iter()
            .zip(inv)
            .enumerate()
            .map(|(k, (mut a, mut b))| {
                a.sort_unstable();
                a.dedup();
                b.sort_unstable();
                b.dedup();
                symmetrize::symmetrize(&a, &b, method).ok_or_else(|| {
                    PyValueError::new_err(format!("pair {k}: the alignment grid is too large"))
                })
            })
            .collect()
    })
}

/// Symmetrise two Pharaoh texts line by line, exactly as `atools -c METHOD -i FWD -j INV` (only
/// `fwd` is used for `invert`).
#[pyfunction]
#[pyo3(signature = (fwd, inv, method))]
fn symmetrize_text<'py>(
    py: Python<'py>,
    fwd: &[u8],
    inv: Option<&[u8]>,
    method: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let method: Method = method.parse().map_err(PyValueError::new_err)?;
    let mut out = Vec::new();
    py.detach(|| {
        symmetrize::symmetrize_text(fwd, inv, method, &mut out).map_err(|e| e.to_string())
    })
    .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &out))
}

/// Swap the two sides of alignments (atools `invert`).
#[pyfunction]
fn invert_many(fwd: Vec<Links>) -> Vec<Links> {
    fwd.iter().map(|a| symmetrize::invert(a)).collect()
}

/// Parse a Pharaoh alignment line (`"0-0 1-2"`); a ` ||| score` suffix is ignored.
#[pyfunction]
fn parse_pharaoh(line: &str) -> PyResult<Links> {
    let bytes = line.as_bytes();
    let cut = bytes
        .windows(5)
        .position(|w| w == b" ||| ")
        .unwrap_or(bytes.len());
    let mut links = Vec::new();
    for tok in line[..cut].split_whitespace() {
        let bad = || PyValueError::new_err(format!("bad alignment link {tok:?} (expected 'i-j')"));
        let (a, b) = tok.split_once('-').ok_or_else(bad)?;
        links.push((
            a.parse::<u32>().map_err(|_| bad())?,
            b.parse::<u32>().map_err(|_| bad())?,
        ));
    }
    Ok(links)
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
