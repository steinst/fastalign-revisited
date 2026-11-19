//! Model files and forced alignment with a trained model.
//!
//! # Model file format (version 2, little endian)
//!
//! ```text
//! magic    "FASTALGN"                 8 bytes
//! version  u32                        2 (version 1 files are read too: no subword section)
//! vocab    u64 n, then n words (u32 length + bytes), ids 0..n (0 = <bad0>, 1 = <eps>, 2 = |||)
//! dirs     forward, then inverse: u8 present, then a DirModel
//! subword  (version 2 only) u8 present; if 1: u8 kind (0 start marker, 1 continuation marker),
//!          u32 marker length + bytes, u32 n tokeniser models (each u64 length + bytes, opaque to
//!          this crate), u32 settings length + bytes (opaque to this crate)
//! end      "ENDFALGN"                 8 bytes (detects truncation)
//! ```
//!
//! A model with a subword section was trained on a segmented corpus: its vocabulary holds pieces,
//! and alignments with it are projected to words with the stored [`Boundary`]. The tokeniser
//! models (serialised SentencePiece protos, written by the Python layer) are only carried along;
//! a model without them behaves like one without a tokeniser.
//!
//! A `DirModel` is its options (`u8` flags: reverse, use_null, favor_diagonal, optimize_tension,
//! variational_bayes; `u32` iterations; `f64` p0, tension, alpha, mean_srclen_multiplier), then
//! the translation table: `u64` rows and cells, row offsets (`u64` × rows+1), target ids (`u32`
//! × cells), probabilities (`f64` × cells, exactly as trained), hash-iteration order (`u32` ×
//! cells) and `u64` bucket counts (× rows). The order is kept so that `-p` tables exported from
//! a model are byte-identical to the ones fast_align (and a training run) writes. Word ids are
//! those of the training corpus's single dictionary, shared by both languages.
//!
//! # Forced alignment with a model
//!
//! New sentences are mapped to the model's vocabulary by word bytes. Words the model does not
//! know, and pairs it has no cell for, have probability `1e-9`, as in fast_align's `-f` mode.
//! The alignment of the training corpus itself is reproduced byte for byte (for `-I` ≥ 2; with
//! `-I 1` the table is all zero, as in fast_align's `-p` export).

use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::align::{align_pair, AlignOptions};
use crate::corpus::{Corpus, Vocab};
use crate::math;
use crate::oracle::{TrainOptions, UNINITIALISED_PROB};
use crate::subword::Boundary;
use crate::symmetrize::{format_links, Link};
use crate::train::{write_text_table, TrainerParts};
use crate::ttable::TTable;

const MAGIC: &[u8; 8] = b"FASTALGN";
const END: &[u8; 8] = b"ENDFALGN";
/// Current model file format version.
pub const FORMAT_VERSION: u32 = 2;
/// Oldest model file format version that is still read.
const MIN_FORMAT_VERSION: u32 = 1;

/// Marker for words a model does not know.
const UNKNOWN: u32 = u32::MAX;

/// Errors reading or writing model files.
#[derive(Debug)]
pub enum ModelError {
    Io(io::Error),
    /// The file is not a (complete, supported) model.
    Format(String),
    /// The model cannot be used as asked.
    Unsupported(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Io(e) => write!(f, "{e}"),
            ModelError::Format(s) | ModelError::Unsupported(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<io::Error> for ModelError {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            ModelError::Format("the model file is truncated".into())
        } else {
            ModelError::Io(e)
        }
    }
}

fn bad<T>(msg: &str) -> Result<T, ModelError> {
    Err(ModelError::Format(msg.into()))
}

/// Which direction of the corpus a model conditions on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// First corpus side generates the second (fast_align without `-r`).
    Forward,
    /// Second side generates the first (`-r`).
    Inverse,
}

impl Direction {
    pub fn name(self) -> &'static str {
        match self {
            Direction::Forward => "fwd",
            Direction::Inverse => "inv",
        }
    }

    pub fn reverse(self) -> bool {
        self == Direction::Inverse
    }
}

/// The training options a model remembers; the ones forced alignment needs plus information.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOptions {
    pub reverse: bool,
    pub use_null: bool,
    pub favor_diagonal: bool,
    pub optimize_tension: bool,
    pub variational_bayes: bool,
    pub iterations: u32,
    pub p0: f64,
    /// The tension the last iteration ended with (what `-T` needs to reproduce the alignments).
    pub tension: f64,
    pub alpha: f64,
    pub mean_srclen_multiplier: f64,
}

impl ModelOptions {
    /// The options of forced alignment with this model.
    pub fn align_options(&self) -> AlignOptions {
        AlignOptions {
            reverse: self.reverse,
            favor_diagonal: self.favor_diagonal,
            use_null: self.use_null,
            p0: self.p0,
            tension: self.tension,
            mean_srclen_multiplier: self.mean_srclen_multiplier,
        }
    }
}

/// One trained direction: options, table structure and probabilities (parallel to
/// `table.cols()`).
#[derive(Clone, Debug)]
pub struct DirModel {
    pub opts: ModelOptions,
    pub table: TTable,
    pub probs: Vec<f64>,
}

impl DirModel {
    /// Package a finished trainer.
    pub fn from_trainer(parts: TrainerParts, reverse: bool) -> Self {
        let o: &TrainOptions = &parts.opts;
        DirModel {
            opts: ModelOptions {
                reverse,
                use_null: o.use_null,
                favor_diagonal: o.favor_diagonal,
                optimize_tension: o.optimize_tension,
                variational_bayes: o.variational_bayes,
                iterations: o.iterations.max(0) as u32,
                p0: o.p0,
                tension: parts.tension,
                alpha: o.alpha,
                mean_srclen_multiplier: parts.mean_srclen_multiplier,
            },
            table: parts.table,
            probs: parts.prob,
        }
    }

    /// Heap bytes of the table and probabilities.
    pub fn memory_bytes(&self) -> usize {
        self.table.memory_bytes() + self.probs.capacity() * 8
    }
}

/// What a model trained on a segmented corpus remembers about its segmentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubwordInfo {
    /// How word boundaries are marked in the pieces.
    pub boundary: Boundary,
    /// Serialised tokeniser models (one joint model, or source then target), opaque here.
    pub tokenizers: Vec<Vec<u8>>,
    /// Tokeniser settings (opaque here; the Python layer stores JSON).
    pub settings: Vec<u8>,
}

/// A trained model: the shared vocabulary and one or both directions.
#[derive(Clone, Debug)]
pub struct Model {
    pub vocab: Vocab,
    pub fwd: Option<DirModel>,
    pub inv: Option<DirModel>,
    /// Set when the vocabulary holds subword pieces (see [`SubwordInfo`]).
    pub subword: Option<SubwordInfo>,
}

fn w_u32s(w: &mut dyn Write, v: &[u32]) -> io::Result<()> {
    let mut buf = Vec::with_capacity(1 << 18);
    for chunk in v.chunks(1 << 16) {
        buf.clear();
        for x in chunk {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        w.write_all(&buf)?;
    }
    Ok(())
}

fn w_u32s_iter(w: &mut dyn Write, v: impl Iterator<Item = u32>) -> io::Result<()> {
    let mut buf = Vec::with_capacity(1 << 18);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
        if buf.len() >= 1 << 18 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)
}

fn w_u64s(w: &mut dyn Write, v: impl Iterator<Item = u64>) -> io::Result<()> {
    let mut buf = Vec::with_capacity(1 << 19);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
        if buf.len() >= 1 << 19 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)
}

fn w_f64s(w: &mut dyn Write, v: &[f64]) -> io::Result<()> {
    w_u64s(w, v.iter().map(|x| x.to_bits()))
}

/// Reader that knows how many bytes are left, so counts from a corrupt header cannot make us
/// allocate more than the file can hold.
struct Rd<R: Read> {
    r: R,
    left: u64,
}

impl<R: Read> Rd<R> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, ModelError> {
        if n as u64 > self.left {
            return bad("the model file is truncated or corrupt");
        }
        let mut v = vec![0u8; n];
        self.r.read_exact(&mut v)?;
        self.left -= n as u64;
        Ok(v)
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N], ModelError> {
        let v = self.bytes(N)?;
        Ok(v.try_into().unwrap())
    }

    fn u8(&mut self) -> Result<u8, ModelError> {
        Ok(self.arr::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        Ok(u32::from_le_bytes(self.arr()?))
    }

    fn u64(&mut self) -> Result<u64, ModelError> {
        Ok(u64::from_le_bytes(self.arr()?))
    }

    fn f64(&mut self) -> Result<f64, ModelError> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// `n` elements of `size` bytes each, read in chunks and converted by `f`.
    fn vec<T>(&mut self, n: u64, size: u64, f: impl Fn(&[u8]) -> T) -> Result<Vec<T>, ModelError> {
        if n.checked_mul(size).is_none_or(|b| b > self.left) {
            return bad("the model file is truncated or corrupt");
        }
        let n = n as usize;
        let size = size as usize;
        let mut out = Vec::with_capacity(n);
        let mut buf = vec![0u8; size << 16];
        let mut todo = n;
        while todo > 0 {
            let k = todo.min(1 << 16);
            self.r.read_exact(&mut buf[..k * size])?;
            out.extend(buf[..k * size].chunks_exact(size).map(&f));
            todo -= k;
        }
        self.left -= (n * size) as u64;
        Ok(out)
    }
}

impl Model {
    /// A model of the given directions over `vocab`.
    pub fn new(vocab: Vocab, fwd: Option<DirModel>, inv: Option<DirModel>) -> Self {
        Model {
            vocab,
            fwd,
            inv,
            subword: None,
        }
    }

    /// The direction's model, if present.
    pub fn dir(&self, d: Direction) -> Option<&DirModel> {
        match d {
            Direction::Forward => self.fwd.as_ref(),
            Direction::Inverse => self.inv.as_ref(),
        }
    }

    /// Write the model file.
    pub fn write(&self, w: &mut dyn Write) -> io::Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&FORMAT_VERSION.to_le_bytes())?;
        w.write_all(&(self.vocab.len() as u64).to_le_bytes())?;
        for id in 0..self.vocab.len() as u32 {
            let word = self.vocab.word(id);
            w.write_all(&(word.len() as u32).to_le_bytes())?;
            w.write_all(word)?;
        }
        for d in [&self.fwd, &self.inv] {
            match d {
                None => w.write_all(&[0])?,
                Some(d) => {
                    w.write_all(&[1])?;
                    let o = &d.opts;
                    w.write_all(&[
                        u8::from(o.reverse),
                        u8::from(o.use_null),
                        u8::from(o.favor_diagonal),
                        u8::from(o.optimize_tension),
                        u8::from(o.variational_bayes),
                    ])?;
                    w.write_all(&o.iterations.to_le_bytes())?;
                    for x in [o.p0, o.tension, o.alpha, o.mean_srclen_multiplier] {
                        w.write_all(&x.to_bits().to_le_bytes())?;
                    }
                    let t = &d.table;
                    w.write_all(&(t.n_rows() as u64).to_le_bytes())?;
                    w.write_all(&(t.n_cells() as u64).to_le_bytes())?;
                    w_u64s(w, t.row_ptr().iter().map(|&x| x as u64))?;
                    w_u32s(w, t.cols())?;
                    w_f64s(w, &d.probs)?;
                    w_u32s_iter(w, t.order())?;
                    w_u64s(w, t.buckets().iter().copied())?;
                }
            }
        }
        match &self.subword {
            None => w.write_all(&[0])?,
            Some(sw) => {
                w.write_all(&[1, u8::from(matches!(sw.boundary, Boundary::Continue(_)))])?;
                let m = sw.boundary.marker();
                w.write_all(&(m.len() as u32).to_le_bytes())?;
                w.write_all(m)?;
                w.write_all(&(sw.tokenizers.len() as u32).to_le_bytes())?;
                for t in &sw.tokenizers {
                    w.write_all(&(t.len() as u64).to_le_bytes())?;
                    w.write_all(t)?;
                }
                w.write_all(&(sw.settings.len() as u32).to_le_bytes())?;
                w.write_all(&sw.settings)?;
            }
        }
        w.write_all(END)
    }

    /// Save to a file.
    pub fn save(&self, path: &Path) -> Result<(), ModelError> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        self.write(&mut w)?;
        w.flush()?;
        Ok(())
    }

    /// Load from a file.
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let f = File::open(path)?;
        let len = f.metadata()?.len();
        Self::read(BufReader::with_capacity(1 << 20, f), len)
    }

    /// Read a model of `len` bytes from `r`. Corrupt or truncated input gives
    /// [`ModelError::Format`], never a panic or an allocation larger than `len`.
    pub fn read<R: Read>(r: R, len: u64) -> Result<Self, ModelError> {
        let mut r = Rd { r, left: len };
        if r.arr::<8>()
            .map_err(|_| ModelError::Format("not a fastalign model file".into()))?
            != *MAGIC
        {
            return bad("not a fastalign model file");
        }
        let version = r.u32()?;
        if !(MIN_FORMAT_VERSION..=FORMAT_VERSION).contains(&version) {
            return Err(ModelError::Unsupported(format!(
                "model file version {version} is not supported (this build reads versions \
                 {MIN_FORMAT_VERSION} to {FORMAT_VERSION})"
            )));
        }
        let n_words = r.u64()?;
        if n_words < 3 || n_words > r.left / 4 {
            return bad("the model file is corrupt (vocabulary size)");
        }
        let mut words = Vec::with_capacity(n_words as usize);
        for _ in 0..n_words {
            let l = r.u32()? as usize;
            words.push(r.bytes(l)?);
        }
        let vocab = Vocab::from_words(words.iter().map(Vec::as_slice));
        if vocab.word(1) != b"<eps>" {
            return bad("the model file is corrupt (vocabulary)");
        }
        let mut dirs = Vec::new();
        for _ in 0..2 {
            dirs.push(match r.u8()? {
                0 => None,
                1 => Some(Self::read_dir(&mut r, vocab.len())?),
                _ => return bad("the model file is corrupt (direction flag)"),
            });
        }
        let subword = if version >= 2 {
            Self::read_subword(&mut r)?
        } else {
            None
        };
        if r.arr::<8>()? != *END || r.left != 0 {
            return bad("the model file is corrupt (end marker)");
        }
        let inv = dirs.pop().unwrap();
        let fwd = dirs.pop().unwrap();
        for (d, rev) in [(&fwd, false), (&inv, true)] {
            if d.as_ref().is_some_and(|d| d.opts.reverse != rev) {
                return bad("the model file is corrupt (direction)");
            }
        }
        Ok(Model {
            vocab,
            fwd,
            inv,
            subword,
        })
    }

    fn read_subword<R: Read>(r: &mut Rd<R>) -> Result<Option<SubwordInfo>, ModelError> {
        match r.u8()? {
            0 => return Ok(None),
            1 => {}
            _ => return bad("the model file is corrupt (subword flag)"),
        }
        let kind = match r.u8()? {
            0 => "start",
            1 => "continue",
            _ => return bad("the model file is corrupt (subword boundary)"),
        };
        let n = r.u32()? as usize;
        let marker = r.bytes(n)?;
        let boundary = Boundary::from_parts(kind, &marker)
            .map_err(|e| ModelError::Format(format!("the model file is corrupt ({e})")))?;
        let n_models = r.u32()?;
        if u64::from(n_models) * 8 > r.left {
            return bad("the model file is truncated or corrupt");
        }
        let mut tokenizers = Vec::new();
        for _ in 0..n_models {
            let len = r.u64()?;
            let len = usize::try_from(len)
                .map_err(|_| ModelError::Format("the model file is corrupt".into()))?;
            tokenizers.push(r.bytes(len)?);
        }
        let n = r.u32()? as usize;
        let settings = r.bytes(n)?;
        Ok(Some(SubwordInfo {
            boundary,
            tokenizers,
            settings,
        }))
    }

    fn read_dir<R: Read>(r: &mut Rd<R>, vocab_len: usize) -> Result<DirModel, ModelError> {
        let flags = r.arr::<5>()?;
        if flags.iter().any(|&b| b > 1) {
            return bad("the model file is corrupt (options)");
        }
        let iterations = r.u32()?;
        let (p0, tension, alpha, mean) = (r.f64()?, r.f64()?, r.f64()?, r.f64()?);
        let opts = ModelOptions {
            reverse: flags[0] == 1,
            use_null: flags[1] == 1,
            favor_diagonal: flags[2] == 1,
            optimize_tension: flags[3] == 1,
            variational_bayes: flags[4] == 1,
            iterations,
            p0,
            tension,
            alpha,
            mean_srclen_multiplier: mean,
        };
        let (rows, cells) = (r.u64()?, r.u64()?);
        let row_ptr = r.vec(rows.saturating_add(1), 8, |b| {
            u64::from_le_bytes(b.try_into().unwrap())
        })?;
        let row_ptr: Vec<usize> = row_ptr
            .into_iter()
            .map(|x| usize::try_from(x).unwrap_or(usize::MAX))
            .collect();
        let u32le = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        let cols = r.vec(cells, 4, u32le)?;
        let probs = r.vec(cells, 8, |b| {
            f64::from_bits(u64::from_le_bytes(b.try_into().unwrap()))
        })?;
        let order = r.vec(cells, 4, u32le)?;
        let buckets = r.vec(rows, 8, |b| u64::from_le_bytes(b.try_into().unwrap()))?;
        let table = TTable::from_parts(row_ptr, cols, order, buckets, vocab_len)
            .map_err(|e| ModelError::Format(format!("the model file is corrupt ({e})")))?;
        Ok(DirModel { opts, table, probs })
    }

    /// Write direction `d`'s conditional probability table in fast_align's `-p` text format.
    pub fn write_table(
        &self,
        d: Direction,
        beam_threshold: f64,
        w: &mut dyn Write,
    ) -> Result<(), ModelError> {
        let dm = self.dir(d).ok_or_else(|| {
            ModelError::Unsupported(format!("the model has no {} direction", d.name()))
        })?;
        write_text_table(&dm.table, &dm.probs, &self.vocab, beam_threshold, w)?;
        Ok(())
    }

    /// Word bytes → model id, for mapping new text.
    pub fn word_index(&self) -> WordIndex<'_> {
        let mut map = FxHashMap::default();
        for id in 1..self.vocab.len() as u32 {
            map.insert(self.vocab.word(id), id);
        }
        WordIndex { map }
    }

    /// Forced alignment of every line of `corpus` (read with the same tokenisation as training)
    /// in direction `d`, as Pharaoh text, one line per corpus line. With `scores`, each line
    /// ends with ` ||| ` and the log score. Lines with an empty side give an empty alignment.
    pub fn align_corpus(
        &self,
        corpus: &Corpus,
        d: Direction,
        scores: bool,
    ) -> Result<Vec<u8>, ModelError> {
        let dm = self.dir(d).ok_or_else(|| {
            ModelError::Unsupported(format!("the model has no {} direction", d.name()))
        })?;
        let index = self.word_index();
        let map: Vec<u32> = (0..corpus.vocab.len() as u32)
            .map(|id| {
                let w = corpus.vocab.word(id);
                if id == 0 {
                    UNKNOWN
                } else {
                    index.map.get(w).copied().unwrap_or(UNKNOWN)
                }
            })
            .collect();
        let aopts = dm.opts.align_options();
        let prob = |e: u32, f: u32| lookup(dm, &map, e, f);
        const CHUNK: usize = 2048;
        let n = corpus.len();
        let starts: Vec<usize> = (0..n).step_by(CHUNK).collect();
        let window = 4 * rayon::current_num_threads();
        let mut out = Vec::new();
        for win in starts.chunks(window) {
            let texts: Vec<Vec<u8>> = win
                .par_iter()
                .map(|&s| {
                    crate::cancel::check();
                    let mut buf = Vec::new();
                    let mut links = Vec::new();
                    for k in s..(s + CHUNK).min(n) {
                        links.clear();
                        let lp = align_pair(
                            &prob,
                            corpus.first.line(k),
                            corpus.second.line(k),
                            &aopts,
                            &mut links,
                        );
                        format_links(&links, &mut buf);
                        if scores {
                            if let Some(lp) = lp {
                                buf.extend_from_slice(b" ||| ");
                                buf.extend_from_slice(math::fmt_g(lp).as_bytes());
                            }
                        }
                        buf.push(b'\n');
                    }
                    buf
                })
                .collect();
            for t in texts {
                out.extend_from_slice(&t);
            }
        }
        Ok(out)
    }

    /// Like [`Model::align_corpus`], but for a model trained on a segmented corpus the piece links
    /// are projected to words (with the model's boundary), so the lines are word alignments. For
    /// any other model it is the same as `align_corpus`. `corpus` must be segmented as in training.
    pub fn align_words(
        &self,
        corpus: &Corpus,
        d: Direction,
        scores: bool,
    ) -> Result<Vec<u8>, ModelError> {
        let raw = self.align_corpus(corpus, d, scores)?;
        match &self.subword {
            None => Ok(raw),
            Some(sw) => crate::subword::project_text(corpus, &sw.boundary, &raw, d.reverse())
                .map_err(ModelError::Unsupported),
        }
    }

    /// Forced alignment of one sentence pair given as tokens (bytes). Returns the links
    /// `(first index, second index)` in increasing target position and the log score; `None`
    /// if a side is empty.
    pub fn align_tokens(
        &self,
        d: Direction,
        index: &WordIndex<'_>,
        first: &[&[u8]],
        second: &[&[u8]],
    ) -> Result<Option<(Vec<Link>, f64)>, ModelError> {
        let dm = self.dir(d).ok_or_else(|| {
            ModelError::Unsupported(format!("the model has no {} direction", d.name()))
        })?;
        let ids = |ws: &[&[u8]]| -> Vec<u32> {
            ws.iter()
                .map(|w| index.map.get(w).copied().unwrap_or(UNKNOWN))
                .collect()
        };
        let (a, b) = (ids(first), ids(second));
        let mut links = Vec::new();
        let ident = |e: u32, f: u32| lookup_ids(dm, e, f);
        Ok(align_pair(&ident, &a, &b, &dm.opts.align_options(), &mut links).map(|lp| (links, lp)))
    }
}

/// Word bytes → model id (see [`Model::word_index`]).
pub struct WordIndex<'m> {
    map: FxHashMap<&'m [u8], u32>,
}

#[inline]
fn lookup_ids(dm: &DirModel, e: u32, f: u32) -> f64 {
    if e == UNKNOWN || f == UNKNOWN {
        return UNINITIALISED_PROB;
    }
    dm.table
        .find(e, f)
        .map_or(UNINITIALISED_PROB, |c| dm.probs[c])
}

#[inline]
fn lookup(dm: &DirModel, map: &[u32], e: u32, f: u32) -> f64 {
    lookup_ids(dm, map[e as usize], map[f as usize])
}

/// Remove ` ||| score` suffixes from Pharaoh lines (output made with scores), leaving links.
pub fn strip_scores(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for line in text.split_inclusive(|&b| b == b'\n') {
        let body = line.strip_suffix(b"\n").unwrap_or(line);
        let cut = body
            .windows(5)
            .position(|w| w == b" ||| ")
            .unwrap_or(body.len());
        out.extend_from_slice(&body[..cut]);
        if line.ends_with(b"\n") {
            out.push(b'\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Model {
        let vocab = Vocab::from_words([&b"<bad0>"[..], b"<eps>", b"|||", b"a", b"x"]);
        let table = TTable::from_parts(
            vec![0, 0, 1, 1, 2, 2],
            vec![4, 4],
            vec![0, 0],
            vec![1, 1, 1, 1, 1],
            5,
        );
        // rows: 0 empty, 1 (NULL) -> x, 2 empty, 3 (a) -> x, 4 empty
        let table = table.unwrap();
        let o = ModelOptions {
            reverse: false,
            use_null: true,
            favor_diagonal: true,
            optimize_tension: true,
            variational_bayes: true,
            iterations: 5,
            p0: 0.08,
            tension: 4.0,
            alpha: 0.01,
            mean_srclen_multiplier: 1.0,
        };
        Model::new(
            vocab,
            Some(DirModel {
                opts: o,
                table,
                probs: vec![0.25, 0.75],
            }),
            None,
        )
    }

    #[test]
    fn save_load_roundtrip_and_corruption() {
        let m = tiny();
        let mut bytes = Vec::new();
        m.write(&mut bytes).unwrap();
        let l = Model::read(&bytes[..], bytes.len() as u64).unwrap();
        assert_eq!(l.vocab, m.vocab);
        let (a, b) = (m.fwd.as_ref().unwrap(), l.fwd.as_ref().unwrap());
        assert_eq!(a.opts, b.opts);
        assert_eq!(a.table, b.table);
        assert_eq!(a.probs, b.probs);
        assert!(l.inv.is_none());
        // every truncation and every single-byte corruption is an error or a valid model, never a panic
        for cut in 0..bytes.len() {
            assert!(Model::read(&bytes[..cut], cut as u64).is_err());
        }
        for k in 0..bytes.len() {
            let mut c = bytes.clone();
            c[k] ^= 0xff;
            let _ = Model::read(&c[..], c.len() as u64);
        }
        let mut c = bytes.clone();
        c.extend_from_slice(b"x");
        assert!(Model::read(&c[..], c.len() as u64).is_err());
    }

    #[test]
    fn version_1_files_still_load() {
        let m = tiny();
        let mut v2 = Vec::new();
        m.write(&mut v2).unwrap();
        // version 1 = version 2 minus the subword flag byte before the end marker
        let mut v1 = v2[..v2.len() - 9].to_vec();
        v1.extend_from_slice(END);
        v1[8..12].copy_from_slice(&1u32.to_le_bytes());
        let l = Model::read(&v1[..], v1.len() as u64).unwrap();
        assert!(l.subword.is_none());
        assert_eq!(l.fwd.unwrap().probs, m.fwd.unwrap().probs);
        // versions that do not exist are refused
        let mut v3 = v2.clone();
        v3[8..12].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(
            Model::read(&v3[..], v3.len() as u64),
            Err(ModelError::Unsupported(_))
        ));
    }

    #[test]
    fn subword_section_round_trips_and_survives_corruption() {
        for boundary in [
            Boundary::sentencepiece(),
            Boundary::Continue(b"@@".to_vec()),
        ] {
            let mut m = tiny();
            m.subword = Some(SubwordInfo {
                boundary,
                tokenizers: vec![b"model-one".to_vec(), Vec::new(), vec![0, 255, 7]],
                settings: b"{\"a\": 1}".to_vec(),
            });
            let mut bytes = Vec::new();
            m.write(&mut bytes).unwrap();
            let l = Model::read(&bytes[..], bytes.len() as u64).unwrap();
            assert_eq!(l.subword, m.subword);
            for cut in 0..bytes.len() {
                assert!(Model::read(&bytes[..cut], cut as u64).is_err());
            }
            for k in 0..bytes.len() {
                let mut c = bytes.clone();
                c[k] ^= 0xff;
                let _ = Model::read(&c[..], c.len() as u64);
            }
        }
        // a subword section without tokenisers is a model that behaves as before
        let mut m = tiny();
        m.subword = Some(SubwordInfo {
            boundary: Boundary::sentencepiece(),
            tokenizers: vec![],
            settings: vec![],
        });
        let mut bytes = Vec::new();
        m.write(&mut bytes).unwrap();
        assert_eq!(
            Model::read(&bytes[..], bytes.len() as u64).unwrap().subword,
            m.subword
        );
    }

    #[test]
    fn strip_scores_keeps_links() {
        assert_eq!(
            strip_scores(b"0-0 1-1 ||| -3.5\n ||| -2\n0-1\n"),
            b"0-0 1-1\n\n0-1\n"
        );
    }

    #[test]
    fn aligns_tokens_with_unknown_words() {
        let m = tiny();
        let idx = m.word_index();
        let (links, _) = m
            .align_tokens(Direction::Forward, &idx, &[b"a"], &[b"x", b"zzz"])
            .unwrap()
            .unwrap();
        assert_eq!(links[0], (0, 0));
        assert!(m
            .align_tokens(Direction::Forward, &idx, &[], &[b"x"])
            .unwrap()
            .is_none());
        assert!(m
            .align_tokens(Direction::Inverse, &idx, &[b"a"], &[b"x"])
            .is_err());
    }
}
