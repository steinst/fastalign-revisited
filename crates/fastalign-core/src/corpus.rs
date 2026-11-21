//! Corpus readers, shared-dictionary interning and flat token arrays.
//!
//! Parsing follows `docs/SPEC-fast-align.md` §3–4 exactly:
//!
//! * lines end at `\n`; a final piece without `\n` is a line if it is non-empty; `\r` is an
//!   ordinary byte;
//! * only space and tab separate tokens, and a tab also acts as a `|||` separator;
//! * the first separator (tab or the token `|||`) splits source from target; later separators are
//!   target words (id 2, spelled `|||`);
//! * one dictionary for both sides: `<eps>` = 1, `|||` = 2, then words in order of first
//!   occurrence (line order, left to right); id 0 is `<bad0>`.
//!
//! Two line-aligned files are the two sides (decision B of the code review): in each file space
//! and tab separate words, and a `|||` word is an ordinary word of its side (id 2, as a later
//! `|||` on the target side of a joined line). fast_align has no such mode; ids are assigned as
//! if the pair were the joined line `SRC ||| TGT`, so a pair of files without tabs or `|||` words
//! gives exactly the corpus of their joined file.
//!
//! Inputs: a regular nonempty file is memory-mapped; anything else (pipes, FIFOs, process
//! substitutions, character devices, files whose metadata size is 0) is read to end of file
//! ([`InputBytes`]).
//!
//! Limits ([`Limits`]): the trainer stores word ids, line ids and positions as `u32`; reading
//! checks every such quantity against its limit and fails with an error instead of wrapping.
//!
//! Reading is parallel: the input is cut into line-aligned chunks, each chunk is tokenised and
//! interned with a chunk-local dictionary, and the local dictionaries are merged in chunk order.
//! Because a chunk's local ids are in first-occurrence order, the merged ids are the global
//! first-occurrence ids whatever the number of chunks or threads.

use std::fmt;
use std::fs::File;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};

use memchr::{memchr, memchr_iter};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::cancel::Cancel;

/// Id of the NULL word `<eps>`.
pub const NULL_ID: u32 = 1;
/// Id of the separator `|||` (also the id of a tab separator or a later `|||` on the target side).
pub const SEP_ID: u32 = 2;

const SEP_WORD: &[u8] = b"|||";
const NULL_WORD: &[u8] = b"<eps>";
const BAD0_WORD: &[u8] = b"<bad0>";

/// Errors while reading a corpus.
#[derive(Debug)]
pub enum CorpusError {
    Io(PathBuf, std::io::Error),
    LineCountMismatch {
        src: usize,
        tgt: usize,
    },
    /// More distinct words (ids, counting the three reserved ones) than [`Limits::max_words`].
    TooManyWords {
        limit: usize,
    },
    /// More input lines than [`Limits::max_lines`].
    TooManyLines {
        limit: usize,
    },
    /// One side of input line `line` (1-based) has more tokens than
    /// [`Limits::max_line_tokens`].
    LineTooLong {
        line: usize,
        side: &'static str,
        tokens: usize,
        limit: usize,
    },
    /// [`EmptyPairs::Error`]: line pairs with an empty side.
    EmptyPairs(EmptyPairReport),
}

/// The line pairs with an empty side found while reading with [`EmptyPairs::Error`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmptyPairReport {
    /// Pairs whose source (first) side is empty and target side is not.
    pub source_empty: usize,
    /// Pairs whose target (second) side is empty and source side is not.
    pub target_empty: usize,
    /// Pairs with both sides empty (blank lines).
    pub both_empty: usize,
    /// The first [`EmptyPairReport::MAX_LISTED`] of them: (1-based input line, empty side).
    pub first: Vec<(usize, EmptySide)>,
}

impl EmptyPairReport {
    /// How many pairs [`EmptyPairReport::first`] lists at most.
    pub const MAX_LISTED: usize = 10;

    pub fn total(&self) -> usize {
        self.source_empty + self.target_empty + self.both_empty
    }

    fn add(&mut self, line: usize, side: EmptySide) {
        match side {
            EmptySide::Source => self.source_empty += 1,
            EmptySide::Target => self.target_empty += 1,
            EmptySide::Both => self.both_empty += 1,
        }
        if self.first.len() < Self::MAX_LISTED {
            self.first.push((line, side));
        }
    }
}

impl fmt::Display for EmptyPairReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.total();
        write!(
            f,
            "{n} line pair{} with an empty side ({} source, {} target, {} both)",
            if n == 1 { "" } else { "s" },
            self.source_empty,
            self.target_empty,
            self.both_empty
        )?;
        for (k, (line, side)) in self.first.iter().enumerate() {
            write!(
                f,
                "{}line {line} ({})",
                if k == 0 { ": " } else { ", " },
                side.name()
            )?;
        }
        if n > self.first.len() {
            write!(f, ", ...")?;
        }
        Ok(())
    }
}

/// Which side of a line pair is empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptySide {
    Source,
    Target,
    Both,
}

impl EmptySide {
    fn of(n_first: usize, n_second: usize) -> Option<Self> {
        match (n_first == 0, n_second == 0) {
            (true, true) => Some(EmptySide::Both),
            (true, false) => Some(EmptySide::Source),
            (false, true) => Some(EmptySide::Target),
            (false, false) => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            EmptySide::Source => "source side empty",
            EmptySide::Target => "target side empty",
            EmptySide::Both => "both sides empty",
        }
    }
}

/// What reading does with a line pair that has an empty side (decision C of the code review).
///
/// fast_align keeps such pairs, and they change the *other* pairs' results: an empty source side
/// makes the mean length ratio infinite and, with tension optimisation, the tension NaN (the
/// forward direction then aligns nothing); a blank line makes it NaN (every score NaN).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmptyPairs {
    /// Train on them, as fast_align does (the default; byte-identical to fast_align).
    #[default]
    Keep,
    /// Drop them before reading words (as `max_len` drops lines): ids, training and output are
    /// those of the file without these lines, and each gets an empty alignment line.
    Skip,
    /// Fail with [`CorpusError::EmptyPairs`] (line numbers, sides and counts).
    Error,
}

impl EmptyPairs {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "keep" => Some(EmptyPairs::Keep),
            "skip" => Some(EmptyPairs::Skip),
            "error" => Some(EmptyPairs::Error),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            EmptyPairs::Keep => "keep",
            EmptyPairs::Skip => "skip",
            EmptyPairs::Error => "error",
        }
    }
}

impl fmt::Display for CorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CorpusError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            CorpusError::LineCountMismatch { src, tgt } => write!(
                f,
                "source and target files have different numbers of lines ({src} vs {tgt})"
            ),
            CorpusError::TooManyWords { limit } => write!(
                f,
                "the corpus has more distinct words than the limit of {limit} ids"
            ),
            CorpusError::TooManyLines { limit } => {
                write!(f, "the corpus has more than the limit of {limit} lines")
            }
            CorpusError::LineTooLong {
                line,
                side,
                tokens,
                limit,
            } => write!(
                f,
                "line {line}: the {side} side has {tokens} tokens, more than the limit of {limit}"
            ),
            CorpusError::EmptyPairs(r) => write!(
                f,
                "the corpus has {r} (empty pairs: error; 'skip' drops them, 'keep' trains on \
                 them as fast_align does)"
            ),
        }
    }
}

impl std::error::Error for CorpusError {}

/// Options for reading a corpus.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// Drop line pairs whose source or target side has more than this many tokens. `None`
    /// (the default) keeps everything, as fast_align does. Dropped lines do not enter the
    /// dictionary; [`Corpus::line_number`] maps kept lines back to input line numbers.
    pub max_len: Option<usize>,
    /// Number of chunks to cut the input into (default: 4 × the rayon thread count). Results do
    /// not depend on it; tests use it to check that.
    pub chunks: Option<usize>,
    /// The operation's cancellation token, checked every [`CANCEL_LINES`] lines of every chunk
    /// and between the merge steps (see [`crate::cancel`]). The default cannot be cancelled.
    pub cancel: Cancel,
    /// Storage limits; the default is [`Limits::DEFAULT`]. Tests lower them.
    pub limits: Limits,
    /// Line pairs with an empty side: keep (fast_align, the default), skip or error. Checked
    /// after `max_len` (a line `max_len` drops is not reported).
    pub empty_pairs: EmptyPairs,
}

impl LoadOptions {
    /// Whether reading may drop lines (so kept lines need their input line numbers).
    fn filters(&self) -> bool {
        self.max_len.is_some() || self.empty_pairs == EmptyPairs::Skip
    }
}

/// Storage limits of a corpus. Word ids, line ids and token positions are `u32` everywhere
/// after reading, with `u32::MAX` (and `u32::MAX - 1` for ids) kept free as sentinels; reading
/// fails with a [`CorpusError`] when a corpus would exceed one. (fast_align's own length
/// statistics are signed 16-bit and wrap for lines over 32767 tokens; that is reproduced on
/// purpose by the trainer and is unrelated to these limits.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Most vocabulary ids, counting the reserved ids 0 (`<bad0>`), 1 (`<eps>`) and 2 (`|||`):
    /// ids stay at most `u32::MAX - 2`.
    pub max_words: usize,
    /// Most input lines (line numbers and line ids are `u32`).
    pub max_lines: usize,
    /// Most tokens on one side of one line (1-based positions are `u32`).
    pub max_line_tokens: usize,
}

impl Limits {
    pub const DEFAULT: Limits = Limits {
        max_words: u32::MAX as usize - 1,
        max_lines: u32::MAX as usize - 1,
        max_line_tokens: u32::MAX as usize - 1,
    };
}

impl Default for Limits {
    fn default() -> Self {
        Limits::DEFAULT
    }
}

/// Lines parsed between two cancellation checks.
pub const CANCEL_LINES: usize = 1024;

/// The vocabulary: id → word bytes. Id 0 is `<bad0>`, 1 `<eps>`, 2 `|||`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vocab {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}

impl Vocab {
    /// Build a vocabulary from the words of ids `0..`.
    pub fn from_words<'a>(words: impl IntoIterator<Item = &'a [u8]>) -> Self {
        let mut bytes = Vec::new();
        let mut offsets = vec![0];
        for w in words {
            bytes.extend_from_slice(w);
            offsets.push(bytes.len());
        }
        Vocab { bytes, offsets }
    }

    /// Number of ids, including the reserved id 0.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The word of `id` (`<bad0>` for 0).
    pub fn word(&self, id: u32) -> &[u8] {
        let i = id as usize;
        &self.bytes[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Heap bytes used.
    pub fn memory_bytes(&self) -> usize {
        self.bytes.capacity() + self.offsets.capacity() * std::mem::size_of::<usize>()
    }
}

/// One side of the corpus: the tokens of every line, concatenated, plus line offsets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Side {
    tokens: Vec<u32>,
    offsets: Vec<usize>,
}

impl Side {
    /// Tokens of line `k` (0-based).
    #[inline]
    pub fn line(&self, k: usize) -> &[u32] {
        &self.tokens[self.offsets[k]..self.offsets[k + 1]]
    }

    #[inline]
    pub fn line_len(&self, k: usize) -> usize {
        self.offsets[k + 1] - self.offsets[k]
    }

    /// All tokens of all lines, in order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Line offsets into [`Side::tokens`] (length = number of lines + 1).
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn memory_bytes(&self) -> usize {
        self.tokens.capacity() * 4 + self.offsets.capacity() * std::mem::size_of::<usize>()
    }
}

/// A parsed parallel corpus. `first` is the side before the separator (fast_align's source
/// without `-r`), `second` the side after it.
#[derive(Clone, Debug)]
pub struct Corpus {
    pub vocab: Vocab,
    pub first: Side,
    pub second: Side,
    /// 1-based input line number of each kept line, when lines may be dropped (`max_len`,
    /// [`EmptyPairs::Skip`]).
    line_numbers: Option<Vec<u32>>,
    /// Raw input lines (as fast_align would see them) of the lines with an empty side, for
    /// fast_align's `Error in line` messages: (0-based kept-line index, bytes).
    empty_side_lines: Vec<(usize, Vec<u8>)>,
    /// Number of input lines read (before `max_len` / [`EmptyPairs::Skip`] filtering).
    n_input_lines: usize,
}

impl Corpus {
    /// Read one file with `SRC ||| TGT` lines (fast_align's format).
    pub fn read_joined(path: &Path, opts: &LoadOptions) -> Result<Self, CorpusError> {
        let buf = InputBytes::open(path)?;
        Self::from_joined_bytes(&buf, opts)
    }

    /// Read two line-aligned files (the source and the target side).
    pub fn read_pair(src: &Path, tgt: &Path, opts: &LoadOptions) -> Result<Self, CorpusError> {
        let a = InputBytes::open(src)?;
        let b = InputBytes::open(tgt)?;
        Self::from_pair_bytes(&a, &b, opts)
    }

    /// Parse an in-memory `SRC ||| TGT` text.
    pub fn from_joined_bytes(text: &[u8], opts: &LoadOptions) -> Result<Self, CorpusError> {
        let n_chunks = n_chunks(opts, text.len());
        let bounds = chunk_bounds(text, n_chunks);
        let chunks: Vec<ChunkInput> = bounds
            .windows(2)
            .map(|w| ChunkInput::Joined(&text[w[0]..w[1]]))
            .collect();
        build(chunks, opts)
    }

    /// Parse two in-memory line-aligned texts (each one side).
    pub fn from_pair_bytes(
        src: &[u8],
        tgt: &[u8],
        opts: &LoadOptions,
    ) -> Result<Self, CorpusError> {
        let n_chunks = n_chunks(opts, src.len().max(tgt.len()));
        let a_bounds = chunk_bounds(src, n_chunks);
        // Lines per source chunk, then the matching byte boundaries in the target text.
        let a_lines: Vec<usize> = a_bounds
            .par_windows(2)
            .map(|w| count_lines(&src[w[0]..w[1]]))
            .collect();
        let total_a: usize = a_lines.iter().sum();
        let b_bounds = line_boundaries(tgt, &a_lines);
        let total_b = count_lines_parallel(tgt, n_chunks);
        if total_a != total_b {
            return Err(CorpusError::LineCountMismatch {
                src: total_a,
                tgt: total_b,
            });
        }
        let chunks: Vec<ChunkInput> = a_bounds
            .windows(2)
            .zip(b_bounds.windows(2))
            .map(|(a, b)| ChunkInput::Pair(&src[a[0]..a[1]], &tgt[b[0]..b[1]]))
            .collect();
        build(chunks, opts)
    }

    /// Number of (kept) line pairs.
    pub fn len(&self) -> usize {
        self.first.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of input lines read, including lines dropped by `max_len`.
    pub fn n_input_lines(&self) -> usize {
        self.n_input_lines
    }

    /// 1-based input line number of kept line `k`.
    pub fn line_number(&self, k: usize) -> usize {
        match &self.line_numbers {
            Some(v) => v[k] as usize,
            None => k + 1,
        }
    }

    /// Make the line-per-pair `text` of a filtered corpus (`max_len`, [`EmptyPairs::Skip`]) one
    /// line per *input* line: dropped lines become empty lines. An unfiltered corpus returns
    /// `text` borrowed. `text` must have exactly one `\n`-terminated line per kept pair; anything
    /// else is an error (nothing is invented: "no decoding iteration" is not expanded, see
    /// [`crate::pipeline::DirOutput::decoded`]).
    pub fn expand_lines<'a>(&self, text: &'a [u8]) -> Result<std::borrow::Cow<'a, [u8]>, String> {
        let n_text = memchr_iter(b'\n', text).count();
        if n_text != self.len() || text.last().is_some_and(|&b| b != b'\n') {
            return Err(format!(
                "the alignments have {n_text} lines but the corpus has {} line pairs",
                self.len()
            ));
        }
        if !self.is_filtered() {
            return Ok(std::borrow::Cow::Borrowed(text));
        }
        let mut out = Vec::with_capacity(text.len() + self.n_input_lines());
        let mut lines = text.split_inclusive(|&b| b == b'\n');
        let mut next = 1usize; // next input line number to write
        for k in 0..self.len() {
            let target = self.line_number(k);
            while next < target {
                out.push(b'\n');
                next += 1;
            }
            out.extend_from_slice(lines.next().expect("line count checked"));
            next += 1;
        }
        while next <= self.n_input_lines() {
            out.push(b'\n');
            next += 1;
        }
        Ok(std::borrow::Cow::Owned(out))
    }

    /// Whether lines may have been dropped (`max_len`, [`EmptyPairs::Skip`]).
    pub fn is_filtered(&self) -> bool {
        self.line_numbers.is_some()
    }

    /// Raw bytes of the lines with an empty side (0-based kept index, line as fast_align reads
    /// it: for two files, `SRC ||| TGT`).
    pub fn empty_side_lines(&self) -> &[(usize, Vec<u8>)] {
        &self.empty_side_lines
    }

    /// `(source, target)` sides for a direction (`reverse` = fast_align's `-r`).
    pub fn sides(&self, reverse: bool) -> (&Side, &Side) {
        if reverse {
            (&self.second, &self.first)
        } else {
            (&self.first, &self.second)
        }
    }

    /// Heap bytes used by the token arrays and the vocabulary.
    pub fn memory_bytes(&self) -> usize {
        self.vocab.memory_bytes()
            + self.first.memory_bytes()
            + self.second.memory_bytes()
            + self.line_numbers.as_ref().map_or(0, |v| v.capacity() * 4)
            + self
                .empty_side_lines
                .iter()
                .map(|(_, l)| l.capacity() + 32)
                .sum::<usize>()
    }
}

// ------------------------------------------------------------------------------------------
// Input buffers

/// The bytes of one input: a memory map of a regular nonempty file, or the whole content of
/// any other input read to end of file.
pub struct InputBytes(Buf);

enum Buf {
    Map(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl InputBytes {
    /// Open `path`. A regular file whose metadata size is nonzero is memory-mapped. Everything
    /// else is read to end of file: a pipe, FIFO or process substitution (`/dev/stdin`,
    /// `/dev/fd/N`) reports size 0 but has content, and a truly empty file reads as empty.
    pub fn open(path: &Path) -> Result<Self, CorpusError> {
        let err = |e| CorpusError::Io(path.to_path_buf(), e);
        let mut file = File::open(path).map_err(err)?;
        let meta = file.metadata().map_err(err)?;
        if meta.is_file() && meta.len() > 0 {
            // SAFETY: the file is only read; concurrent modification by another process is the
            // caller's responsibility, as for any memory-mapped reader.
            let map = unsafe { memmap2::Mmap::map(&file) }.map_err(err)?;
            return Ok(InputBytes(Buf::Map(map)));
        }
        let mut v = Vec::new();
        file.read_to_end(&mut v).map_err(err)?;
        Ok(InputBytes(Buf::Owned(v)))
    }

    /// Standard input, read to end of file.
    pub fn stdin() -> Result<Self, CorpusError> {
        let mut v = Vec::new();
        std::io::stdin()
            .lock()
            .read_to_end(&mut v)
            .map_err(|e| CorpusError::Io(PathBuf::from("<stdin>"), e))?;
        Ok(InputBytes(Buf::Owned(v)))
    }

    /// Bytes already in memory.
    pub fn from_vec(v: Vec<u8>) -> Self {
        InputBytes(Buf::Owned(v))
    }
}

impl std::ops::Deref for InputBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match &self.0 {
            Buf::Map(m) => m,
            Buf::Owned(v) => v,
        }
    }
}

fn n_chunks(opts: &LoadOptions, len: usize) -> usize {
    let n = opts.chunks.unwrap_or(4 * rayon::current_num_threads());
    // At least ~64 KiB per chunk unless the caller insists.
    let n = if opts.chunks.is_some() {
        n
    } else {
        n.min(len / 65536 + 1)
    };
    n.max(1)
}

/// Byte offsets `0 = b_0 ≤ … ≤ b_k = len` cutting `text` after newlines into about `n` pieces.
fn chunk_bounds(text: &[u8], n: usize) -> Vec<usize> {
    let mut b = vec![0];
    for c in 1..n {
        let target = text.len() * c / n;
        let prev = *b.last().unwrap();
        let start = target.max(prev);
        let cut = match memchr(b'\n', &text[start..]) {
            Some(p) => start + p + 1,
            None => text.len(),
        };
        if cut > prev && cut < text.len() {
            b.push(cut);
        }
    }
    b.push(text.len());
    b
}

/// Number of lines in a piece that is either all of the text or ends after a newline.
fn count_lines(piece: &[u8]) -> usize {
    let nl = memchr_iter(b'\n', piece).count();
    nl + usize::from(piece.last().is_some_and(|&c| c != b'\n'))
}

fn count_lines_parallel(text: &[u8], n: usize) -> usize {
    chunk_bounds(text, n)
        .par_windows(2)
        .map(|w| count_lines(&text[w[0]..w[1]]))
        .sum()
}

/// Byte boundaries in `text` after the given numbers of lines (cumulative), as chunk bounds.
/// If `text` has fewer lines the remaining bounds are `text.len()` (the caller reports the
/// mismatch).
fn line_boundaries(text: &[u8], lines_per_chunk: &[usize]) -> Vec<usize> {
    let mut targets = Vec::with_capacity(lines_per_chunk.len());
    let mut acc = 0;
    for &l in &lines_per_chunk[..lines_per_chunk.len().saturating_sub(1)] {
        acc += l;
        targets.push(acc);
    }
    // Newline positions are found with one parallel pass over coarse chunks.
    let coarse = chunk_bounds(text, rayon::current_num_threads().max(1) * 2);
    let counts: Vec<usize> = coarse
        .par_windows(2)
        .map(|w| memchr_iter(b'\n', &text[w[0]..w[1]]).count())
        .collect();
    let mut start_line = Vec::with_capacity(counts.len());
    let mut s = 0;
    for c in &counts {
        start_line.push(s);
        s += c;
    }
    let mut out = vec![0];
    out.par_extend(targets.par_iter().map(|&t| {
        // Offset just after the t-th newline (t ≥ 1), or text.len().
        let ci = start_line.partition_point(|&sl| sl < t);
        if ci == 0 {
            return 0;
        }
        let ci = ci - 1;
        let need = t - start_line[ci]; // 1-based newline within coarse chunk ci
        let w = coarse[ci]..coarse[ci + 1];
        match memchr_iter(b'\n', &text[w.clone()]).nth(need - 1) {
            Some(p) => w.start + p + 1,
            None => text.len(),
        }
    }));
    out.push(text.len());
    out
}

// ------------------------------------------------------------------------------------------
// Tokenisation and chunk-local interning

enum ChunkInput<'a> {
    Joined(&'a [u8]),
    Pair(&'a [u8], &'a [u8]),
}

/// Append the words of `line` to `out`: space and tab separate words. With `tab_is_sep` (a
/// joined line, spec §3) a tab is also a separator token, returned as the word `|||`.
fn split_words<'a>(line: &'a [u8], tab_is_sep: bool, out: &mut Vec<&'a [u8]>) {
    let mut start = None;
    for (p, &c) in line.iter().enumerate() {
        if c == b' ' || c == b'\t' {
            if let Some(s) = start.take() {
                out.push(&line[s..p]);
            }
            if c == b'\t' && tab_is_sep {
                out.push(SEP_WORD);
            }
        } else if start.is_none() {
            start = Some(p);
        }
    }
    if let Some(s) = start {
        out.push(&line[s..]);
    }
}

/// A limit hit inside a chunk, at the chunk's 0-based input line `.0`.
enum ChunkLimit {
    Words,
    Lines,
    LineTokens(&'static str, usize),
}

/// Output of one chunk: local dictionary and tokens in local ids.
struct ChunkOut<'a> {
    /// Local words, local id = index + 2 (local 0 = `<eps>`, local 1 = `|||`).
    words: Vec<&'a [u8]>,
    first: Vec<u32>,
    first_len: Vec<u32>,
    second: Vec<u32>,
    second_len: Vec<u32>,
    /// For each kept line: its 0-based index among the chunk's input lines.
    kept: Option<Vec<u32>>,
    n_input: usize,
    empty_side: Vec<(usize, Vec<u8>)>,
    /// The first limit exceeded (reading of the chunk stopped there).
    limit: Option<(usize, ChunkLimit)>,
    /// [`EmptyPairs::Error`]: the chunk's empty-sided pairs (lines are chunk-local, 1-based).
    empty_pairs: EmptyPairReport,
}

struct LocalDict<'a> {
    map: FxHashMap<&'a [u8], u32>,
    words: Vec<&'a [u8]>,
    /// Most local words before the chunk alone exceeds [`Limits::max_words`].
    max_words: usize,
    overflow: bool,
}

impl<'a> LocalDict<'a> {
    fn new(limits: &Limits) -> Self {
        let mut map = FxHashMap::default();
        map.insert(NULL_WORD, 0);
        map.insert(SEP_WORD, 1);
        LocalDict {
            map,
            words: Vec::new(),
            // the global ids add `<bad0>`, `<eps>` and `|||`
            max_words: limits.max_words.saturating_sub(3),
            overflow: false,
        }
    }

    #[inline]
    fn intern(&mut self, w: &'a [u8]) -> u32 {
        let n = self.words.len();
        let (max, overflow) = (self.max_words, &mut self.overflow);
        let words = &mut self.words;
        *self.map.entry(w).or_insert_with(|| {
            if n >= max {
                // the chunk is discarded with an error; the id only has to be in range
                *overflow = true;
                return 0;
            }
            words.push(w);
            n as u32 + 2
        })
    }
}

fn process_chunk<'a>(input: &ChunkInput<'a>, opts: &LoadOptions) -> ChunkOut<'a> {
    let limits = &opts.limits;
    let mut dict = LocalDict::new(limits);
    let mut out = ChunkOut {
        words: Vec::new(),
        first: Vec::new(),
        first_len: Vec::new(),
        second: Vec::new(),
        second_len: Vec::new(),
        kept: opts.filters().then(Vec::new),
        n_input: 0,
        empty_side: Vec::new(),
        limit: None,
        empty_pairs: EmptyPairReport::default(),
    };
    // The line's words; for a joined line separators are the word `|||`.
    let mut words: Vec<&'a [u8]> = Vec::new();
    // `words[..n_first]` is the first side, `words[second_start..]` the second.
    let handle = |n_first: usize,
                  second_start: usize,
                  words: &[&'a [u8]],
                  raw: &dyn Fn() -> Vec<u8>,
                  out: &mut ChunkOut<'a>,
                  dict: &mut LocalDict<'a>|
     -> Option<ChunkLimit> {
        let n_second = words.len() - second_start;
        let line_idx = out.n_input;
        if line_idx >= limits.max_lines {
            return Some(ChunkLimit::Lines);
        }
        out.n_input += 1;
        if let Some(max) = opts.max_len {
            if n_first > max || n_second > max {
                return None;
            }
        }
        let empty = EmptySide::of(n_first, n_second);
        if let Some(side) = empty {
            match opts.empty_pairs {
                EmptyPairs::Keep => {}
                EmptyPairs::Skip => return None,
                EmptyPairs::Error => {
                    // reading goes on (to count them all), the corpus is discarded
                    out.empty_pairs.add(line_idx + 1, side);
                    return None;
                }
            }
        }
        if let Some(kept) = out.kept.as_mut() {
            kept.push(line_idx as u32);
        }
        for (side, n) in [("source", n_first), ("target", n_second)] {
            if n > limits.max_line_tokens {
                return Some(ChunkLimit::LineTokens(side, n));
            }
        }
        let kept_idx = out.first_len.len();
        if empty.is_some() {
            out.empty_side.push((kept_idx, raw()));
        }
        for &w in &words[..n_first] {
            out.first.push(dict.intern(w));
        }
        for &w in &words[second_start..] {
            out.second.push(dict.intern(w));
        }
        if dict.overflow {
            return Some(ChunkLimit::Words);
        }
        out.first_len.push(n_first as u32);
        out.second_len.push(n_second as u32);
        None
    };
    let cancel = &opts.cancel;
    let mut stop = None;
    match *input {
        ChunkInput::Joined(text) => {
            for (k, line) in lines(text).enumerate() {
                if k % CANCEL_LINES == 0 {
                    cancel.check();
                }
                words.clear();
                split_words(line, true, &mut words);
                // The first separator splits; later ones are target words `|||` (id 2).
                let (n_first, second_start) = match words.iter().position(|&w| w == SEP_WORD) {
                    Some(p) => (p, p + 1),
                    None => (words.len(), words.len()),
                };
                let raw = || line.to_vec();
                if let Some(l) = handle(n_first, second_start, &words, &raw, &mut out, &mut dict) {
                    stop = Some((k, l));
                    break;
                }
            }
        }
        ChunkInput::Pair(a, b) => {
            let mut lb = lines(b);
            for (k, la) in lines(a).enumerate() {
                if k % CANCEL_LINES == 0 {
                    cancel.check();
                }
                let lb = lb.next().unwrap_or(&[]);
                words.clear();
                split_words(la, false, &mut words);
                let n_first = words.len();
                split_words(lb, false, &mut words);
                // fast_align's `Error in line` text shows the pair as one joined line
                let raw = || [la, b" ||| ", lb].concat();
                if let Some(l) = handle(n_first, n_first, &words, &raw, &mut out, &mut dict) {
                    stop = Some((k, l));
                    break;
                }
            }
        }
    }
    out.limit = stop;
    out.words = dict.words;
    out
}

/// Lines of a piece per `getline`: split on `\n`, a final non-empty piece is a line.
fn lines(text: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        match memchr(b'\n', rest) {
            Some(p) => {
                let l = &rest[..p];
                rest = &rest[p + 1..];
                Some(l)
            }
            None => {
                let l = rest;
                rest = &[];
                Some(l)
            }
        }
    })
}

// ------------------------------------------------------------------------------------------
// Merge

fn build(chunks: Vec<ChunkInput>, opts: &LoadOptions) -> Result<Corpus, CorpusError> {
    let cancel = &opts.cancel;
    let limits = &opts.limits;
    cancel.check();
    let outs: Vec<ChunkOut> = chunks.par_iter().map(|c| process_chunk(c, opts)).collect();
    cancel.check();

    // The first limit hit, in input order (every chunk before it was read completely).
    let mut input_base = 0usize;
    for o in &outs {
        if let Some((k, l)) = &o.limit {
            return Err(match *l {
                ChunkLimit::Words => CorpusError::TooManyWords {
                    limit: limits.max_words,
                },
                ChunkLimit::Lines => CorpusError::TooManyLines {
                    limit: limits.max_lines,
                },
                ChunkLimit::LineTokens(side, tokens) => CorpusError::LineTooLong {
                    line: input_base + k + 1,
                    side,
                    tokens,
                    limit: limits.max_line_tokens,
                },
            });
        }
        input_base += o.n_input;
        if input_base > limits.max_lines {
            return Err(CorpusError::TooManyLines {
                limit: limits.max_lines,
            });
        }
    }
    if opts.empty_pairs == EmptyPairs::Error {
        let mut report = EmptyPairReport::default();
        let mut input_base = 0usize;
        for o in &outs {
            let r = &o.empty_pairs;
            report.source_empty += r.source_empty;
            report.target_empty += r.target_empty;
            report.both_empty += r.both_empty;
            for &(line, side) in &r.first {
                if report.first.len() < EmptyPairReport::MAX_LISTED {
                    report.first.push((input_base + line, side));
                }
            }
            input_base += o.n_input;
        }
        if report.total() > 0 {
            return Err(CorpusError::EmptyPairs(report));
        }
    }

    // Global dictionary in chunk order: first-occurrence ids.
    let mut map: FxHashMap<&[u8], u32> = FxHashMap::default();
    let mut words: Vec<&[u8]> = vec![BAD0_WORD, NULL_WORD, SEP_WORD];
    map.insert(NULL_WORD, NULL_ID);
    map.insert(SEP_WORD, SEP_ID);
    let mut remaps: Vec<Vec<u32>> = Vec::with_capacity(outs.len());
    for o in &outs {
        cancel.check();
        let mut r = Vec::with_capacity(o.words.len() + 2);
        r.push(NULL_ID);
        r.push(SEP_ID);
        for &w in &o.words {
            let next = words.len() as u32;
            let id = *map.entry(w).or_insert_with(|| {
                words.push(w);
                next
            });
            r.push(id);
        }
        // Checked per chunk, off the hot path: an id that wrapped within this chunk (only
        // possible beyond the limit) is discarded with the error.
        if words.len() > limits.max_words {
            return Err(CorpusError::TooManyWords {
                limit: limits.max_words,
            });
        }
        remaps.push(r);
    }
    drop(map);
    let vocab = make_vocab(&words);
    cancel.check();

    let first = concat_side(&outs, &remaps, |o| (&o.first, &o.first_len));
    let second = concat_side(&outs, &remaps, |o| (&o.second, &o.second_len));
    cancel.check();

    let mut line_numbers = opts.filters().then(Vec::new);
    let mut empty_side_lines = Vec::new();
    let mut input_base = 0usize;
    let mut kept_base = 0usize;
    for o in &outs {
        if let (Some(v), Some(k)) = (line_numbers.as_mut(), o.kept.as_ref()) {
            // fits: input_base + i < the number of input lines <= max_lines < u32::MAX
            v.extend(k.iter().map(|&i| (input_base + i as usize + 1) as u32));
        }
        empty_side_lines.extend(o.empty_side.iter().map(|(k, l)| (kept_base + k, l.clone())));
        input_base += o.n_input;
        kept_base += o.first_len.len();
    }
    Ok(Corpus {
        vocab,
        first,
        second,
        line_numbers,
        empty_side_lines,
        n_input_lines: input_base,
    })
}

fn make_vocab(words: &[&[u8]]) -> Vocab {
    let total: usize = words.iter().map(|w| w.len()).sum();
    let mut bytes = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(words.len() + 1);
    offsets.push(0);
    for w in words {
        bytes.extend_from_slice(w);
        offsets.push(bytes.len());
    }
    Vocab { bytes, offsets }
}

fn concat_side<'a, F>(outs: &'a [ChunkOut], remaps: &[Vec<u32>], get: F) -> Side
where
    F: Fn(&'a ChunkOut) -> (&'a Vec<u32>, &'a Vec<u32>) + Sync,
{
    let n_lines: usize = outs.iter().map(|o| get(o).1.len()).sum();
    let n_tok: usize = outs.iter().map(|o| get(o).0.len()).sum();
    let mut tokens = vec![0u32; n_tok];
    let mut offsets = vec![0usize; n_lines + 1];
    // Split the outputs into disjoint per-chunk slices and fill them in parallel.
    let mut tok_slices = Vec::with_capacity(outs.len());
    let mut off_slices = Vec::with_capacity(outs.len());
    let (mut trest, mut orest) = (&mut tokens[..], &mut offsets[1..]);
    let (mut tbase, mut lbase) = (0usize, 0usize);
    for o in outs {
        let (t, l) = get(o);
        let (th, tt) = trest.split_at_mut(t.len());
        let (oh, ot) = orest.split_at_mut(l.len());
        tok_slices.push((th, tbase));
        off_slices.push(oh);
        trest = tt;
        orest = ot;
        tbase += t.len();
        lbase += l.len();
    }
    debug_assert_eq!(lbase, n_lines);
    tok_slices
        .into_par_iter()
        .zip(off_slices)
        .zip(outs.par_iter().zip(remaps))
        .for_each(|(((dst, base), odst), (o, remap))| {
            let (t, l) = get(o);
            for (d, &s) in dst.iter_mut().zip(t) {
                *d = remap[s as usize];
            }
            let mut acc = base;
            for (d, &len) in odst.iter_mut().zip(l) {
                acc += len as usize;
                *d = acc;
            }
        });
    Side { tokens, offsets }
}

/// Most entries of one corpus part's posting arrays (their offsets are `u32` prefix sums).
pub(crate) const MAX_PART_ENTRIES: usize = u32::MAX as usize;

/// Line ranges of the corpus parts that the per-part posting arrays (source-word postings of
/// the initial pass, the trainer's row lines) are built over: `n_parts` equal line ranges,
/// each further cut where its tokens on `side` would exceed `max_entries`, so that every
/// part's entries (at most one per token) fit its `u32` offsets. A single line longer than
/// `max_entries` is a part of its own; with the default limits that cannot happen
/// ([`Limits::max_line_tokens`] < [`MAX_PART_ENTRIES`]). Parts are in corpus order and cover
/// every line once.
pub(crate) fn part_ranges(side: &Side, n_parts: usize, max_entries: usize) -> Vec<Range<usize>> {
    let n_lines = side.offsets.len() - 1;
    let n_parts = n_parts.max(1);
    let off = &side.offsets;
    let mut out = Vec::with_capacity(n_parts);
    for p in 0..n_parts {
        let (lo, hi) = (n_lines * p / n_parts, n_lines * (p + 1) / n_parts);
        if off[hi] - off[lo] <= max_entries {
            out.push(lo..hi);
            continue;
        }
        let mut start = lo;
        for k in lo..hi {
            if k > start && off[k + 1] - off[start] > max_entries {
                out.push(start..k);
                start = k;
            }
        }
        out.push(start..hi);
    }
    out
}

/// Byte range helper for tests and tools: the `Range` of line `k`'s tokens on a side.
pub fn line_range(side: &Side, k: usize) -> Range<usize> {
    side.offsets[k]..side.offsets[k + 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(c: &Corpus, side: &Side, k: usize) -> Vec<String> {
        side.line(k)
            .iter()
            .map(|&i| String::from_utf8_lossy(c.vocab.word(i)).into_owned())
            .collect()
    }

    fn opts(chunks: usize) -> LoadOptions {
        LoadOptions {
            chunks: Some(chunks),
            ..Default::default()
        }
    }

    fn with_empty(e: EmptyPairs, chunks: usize) -> LoadOptions {
        LoadOptions {
            chunks: Some(chunks),
            empty_pairs: e,
            ..Default::default()
        }
    }

    /// Review #5 / decision C: `skip` reads the file as if the empty-sided pairs were not there
    /// (ids, tokens), keeps their input line numbers, and expands to one line per input line.
    #[test]
    fn empty_pairs_skip_is_the_file_without_them() {
        let text = b"a b ||| x y\n ||| z\nq ||| \n\nb c ||| y w\n\t\n";
        let clean = b"a b ||| x y\nb c ||| y w\n";
        for chunks in 1..=6 {
            let c = Corpus::from_joined_bytes(text, &with_empty(EmptyPairs::Skip, chunks)).unwrap();
            let r = Corpus::from_joined_bytes(clean, &opts(1)).unwrap();
            assert_eq!(c.vocab, r.vocab, "chunks {chunks}");
            assert_eq!(c.first.tokens(), r.first.tokens());
            assert_eq!(c.second.tokens(), r.second.tokens());
            assert_eq!((c.len(), c.n_input_lines()), (2, 6));
            assert_eq!((c.line_number(0), c.line_number(1)), (1, 5));
            assert!(c.empty_side_lines().is_empty());
            assert_eq!(
                c.expand_lines(b"0-0\n1-1\n").unwrap().as_ref(),
                b"0-0\n\n\n\n1-1\n\n"
            );
        }
        // paired files: same rule, sides are the files
        let c = Corpus::from_pair_bytes(b"a\n\nb\n", b"x\ny\n\n", &with_empty(EmptyPairs::Skip, 2))
            .unwrap();
        assert_eq!((c.len(), c.n_input_lines(), c.line_number(0)), (1, 3, 1));
        // keep is the default and fast_align's reading
        let k = Corpus::from_joined_bytes(text, &with_empty(EmptyPairs::Keep, 3)).unwrap();
        let d = Corpus::from_joined_bytes(text, &opts(3)).unwrap();
        assert_eq!((k.len(), k.vocab.clone()), (d.len(), d.vocab.clone()));
        assert_eq!(k.empty_side_lines().len(), 4);
        assert!(!k.is_filtered());
    }

    #[test]
    fn empty_pairs_error_lists_input_lines_sides_and_counts() {
        let mut text = Vec::new();
        for k in 1..=40 {
            text.extend_from_slice(match k {
                3 => b" ||| z\n".as_slice(),
                7 => b"q |||\n",
                8 | 30 => b"\n",
                k if k > 12 && k % 2 == 0 => b"a ||| \n",
                _ => b"a ||| x\n",
            });
        }
        for chunks in [1, 2, 5, 13] {
            let e = Corpus::from_joined_bytes(&text, &with_empty(EmptyPairs::Error, chunks))
                .unwrap_err();
            let CorpusError::EmptyPairs(r) = &e else {
                panic!("{e}")
            };
            // lines 14..40 even: 14 lines, 30 of them blank
            assert_eq!((r.source_empty, r.target_empty, r.both_empty), (1, 14, 2));
            assert_eq!(r.first.len(), EmptyPairReport::MAX_LISTED);
            assert_eq!(
                r.first[..5],
                [
                    (3, EmptySide::Source),
                    (7, EmptySide::Target),
                    (8, EmptySide::Both),
                    (14, EmptySide::Target),
                    (16, EmptySide::Target)
                ]
            );
            let msg = e.to_string();
            assert!(msg.contains("17 line pairs with an empty side (1 source, 14 target, 2 both)"));
            assert!(msg.contains("line 3 (source side empty), line 7 (target side empty)"));
            assert!(msg.ends_with(", ... (empty pairs: error; 'skip' drops them, 'keep' trains on them as fast_align does)"), "{msg}");
        }
        // a line dropped by max_len is not reported; no empty pair: no error
        let o = LoadOptions {
            max_len: Some(1),
            ..with_empty(EmptyPairs::Error, 2)
        };
        let c = Corpus::from_joined_bytes(b"a ||| x\nb c d |||\n", &o).unwrap();
        assert_eq!((c.len(), c.n_input_lines()), (1, 2));
    }

    /// Expansion never invents a line: text with a different line count is an error.
    #[test]
    fn expand_lines_is_strict() {
        let o = LoadOptions {
            max_len: Some(1),
            ..opts(1)
        };
        let c = Corpus::from_joined_bytes(b"a ||| x\nb c ||| y\n", &o).unwrap();
        assert_eq!(c.expand_lines(b"0-0\n").unwrap().as_ref(), b"0-0\n\n");
        for bad in [&b""[..], b"0-0", b"0-0\n\n"] {
            assert!(c.expand_lines(bad).is_err(), "{bad:?}");
        }
        let all = Corpus::from_joined_bytes(b"a b ||| x\nb c ||| y\n", &o).unwrap();
        assert_eq!(all.expand_lines(b"").unwrap().as_ref(), b"\n\n");
        let plain = Corpus::from_joined_bytes(b"a ||| x\n", &opts(1)).unwrap();
        assert!(matches!(
            plain.expand_lines(b"0-0\n").unwrap(),
            std::borrow::Cow::Borrowed(_)
        ));
        assert!(plain.expand_lines(b"").is_err());
    }

    #[test]
    fn ids_in_first_occurrence_order() {
        let c = Corpus::from_joined_bytes(b"a b ||| x a\nb c ||| y\n", &opts(1)).unwrap();
        let v: Vec<&[u8]> = (0..c.vocab.len() as u32).map(|i| c.vocab.word(i)).collect();
        assert_eq!(
            v,
            [
                &b"<bad0>"[..],
                b"<eps>",
                b"|||",
                b"a",
                b"b",
                b"x",
                b"c",
                b"y"
            ]
        );
        assert_eq!(c.first.line(0), [3, 4]);
        assert_eq!(c.second.line(0), [5, 3]);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn separators_tabs_and_extra_bars() {
        let c = Corpus::from_joined_bytes(b"a\tb ||| x\nc ||| d ||| e\n\n ||| y\nz\n", &opts(1))
            .unwrap();
        assert_eq!(words(&c, &c.first, 0), ["a"]);
        assert_eq!(words(&c, &c.second, 0), ["b", "|||", "x"]);
        assert_eq!(words(&c, &c.second, 1), ["d", "|||", "e"]);
        assert_eq!(c.first.line_len(2) + c.second.line_len(2), 0);
        assert_eq!(words(&c, &c.second, 3), ["y"]);
        assert_eq!(words(&c, &c.first, 4), ["z"]);
        assert_eq!(c.second.line_len(4), 0);
        let empties: Vec<usize> = c.empty_side_lines().iter().map(|e| e.0).collect();
        assert_eq!(empties, [2, 3, 4]);
        assert_eq!(c.empty_side_lines()[1].1, b" ||| y");
    }

    #[test]
    fn whitespace_rules() {
        // only space/tab separate; \r and other bytes are word bytes; no trailing newline
        let c = Corpus::from_joined_bytes(b"  a\x0bb  ||| x\r", &opts(1)).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(words(&c, &c.first, 0), ["a\u{b}b"]);
        assert_eq!(words(&c, &c.second, 0), ["x\r"]);
        assert!(Corpus::from_joined_bytes(b"", &opts(1)).unwrap().is_empty());
        assert_eq!(Corpus::from_joined_bytes(b"\n", &opts(1)).unwrap().len(), 1);
    }

    #[test]
    fn eps_token_is_null() {
        let c = Corpus::from_joined_bytes(b"<eps> a ||| |||\n", &opts(1)).unwrap();
        assert_eq!(c.first.line(0), [NULL_ID, 3]);
        assert_eq!(c.second.line(0), [SEP_ID]);
    }

    fn random_corpus(seed: u64, lines: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut x = seed;
        let mut rnd = move |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n
        };
        let (mut joined, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..lines {
            let mut side = |out: &mut Vec<u8>, out2: &mut Vec<u8>| {
                let n = rnd(6);
                for k in 0..n {
                    if k > 0 {
                        out.push(b' ');
                        out2.push(b' ');
                    }
                    let w = format!("w{}", rnd(30) * rnd(30));
                    out.extend_from_slice(w.as_bytes());
                    out2.extend_from_slice(w.as_bytes());
                }
            };
            side(&mut joined, &mut a);
            joined.extend_from_slice(b" ||| ");
            side(&mut joined, &mut b);
            joined.push(b'\n');
            a.push(b'\n');
            b.push(b'\n');
        }
        (joined, a, b)
    }

    fn same(x: &Corpus, y: &Corpus) {
        assert_eq!(x.vocab, y.vocab);
        assert_eq!(x.first, y.first);
        assert_eq!(x.second, y.second);
        assert_eq!(x.empty_side_lines, y.empty_side_lines);
    }

    #[test]
    fn chunking_does_not_change_anything() {
        for seed in 1..6 {
            let (joined, a, b) = random_corpus(seed * 7919, 300);
            let base = Corpus::from_joined_bytes(&joined, &opts(1)).unwrap();
            for chunks in [2, 3, 7, 64, 1000] {
                same(
                    &base,
                    &Corpus::from_joined_bytes(&joined, &opts(chunks)).unwrap(),
                );
                same(
                    &base,
                    &Corpus::from_pair_bytes(&a, &b, &opts(chunks)).unwrap(),
                );
            }
        }
    }

    #[test]
    fn pair_mismatch_is_an_error() {
        let e = Corpus::from_pair_bytes(b"a\nb\n", b"x\n", &opts(2)).unwrap_err();
        assert!(matches!(
            e,
            CorpusError::LineCountMismatch { src: 2, tgt: 1 }
        ));
        let e = Corpus::from_pair_bytes(b"a\nb", b"x\ny\nz", &opts(3)).unwrap_err();
        assert!(matches!(
            e,
            CorpusError::LineCountMismatch { src: 2, tgt: 3 }
        ));
    }

    #[test]
    fn max_len_drops_lines_before_interning() {
        let o = LoadOptions {
            max_len: Some(2),
            chunks: Some(2),
            ..Default::default()
        };
        let c = Corpus::from_joined_bytes(b"a ||| x\nlong long long ||| y\nb ||| z\n", &o).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c.n_input_lines(), 3);
        assert_eq!((c.line_number(0), c.line_number(1)), (1, 3));
        let joined = Corpus::from_joined_bytes(b"a ||| x\nb ||| z\n", &opts(1)).unwrap();
        assert_eq!(c.vocab, joined.vocab);
    }

    #[test]
    fn part_ranges_respect_the_entry_limit() {
        let (joined, _, _) = random_corpus(4242, 500);
        let c = Corpus::from_joined_bytes(&joined, &opts(3)).unwrap();
        let side = &c.first;
        let n = c.len();
        let longest = (0..n).map(|k| side.line_len(k)).max().unwrap();
        for parts in [1, 2, 7, 1000] {
            for limit in [
                longest,
                longest + 1,
                2 * longest,
                64,
                1000,
                MAX_PART_ENTRIES,
            ] {
                let r = part_ranges(side, parts, limit);
                // in order, contiguous, covering every line once
                assert_eq!(r.first().unwrap().start, 0);
                assert_eq!(r.last().unwrap().end, n);
                for w in r.windows(2) {
                    assert_eq!(w[0].end, w[1].start);
                }
                for x in &r {
                    assert!(side.offsets[x.end] - side.offsets[x.start] <= limit);
                }
                if limit == MAX_PART_ENTRIES {
                    // unchanged equal line split at real scale
                    let eq: Vec<_> = (0..parts)
                        .map(|p| n * p / parts..n * (p + 1) / parts)
                        .collect();
                    assert_eq!(r, eq);
                }
            }
        }
        // a line longer than the limit is a part of its own
        let c = Corpus::from_joined_bytes(b"a ||| x\na b c d ||| y\ne ||| z\n", &opts(1)).unwrap();
        assert_eq!(part_ranges(&c.first, 1, 2), [0..1, 1..2, 2..3]);
    }

    fn limited(limits: Limits) -> LoadOptions {
        LoadOptions {
            limits,
            chunks: Some(2),
            ..Default::default()
        }
    }

    #[test]
    fn word_limit() {
        // 5 distinct words + 3 reserved ids = 8
        let text = b"a b ||| x\nc ||| a y\n";
        let ok = Limits {
            max_words: 8,
            ..Limits::DEFAULT
        };
        assert_eq!(
            Corpus::from_joined_bytes(text, &limited(ok))
                .unwrap()
                .vocab
                .len(),
            8
        );
        let low = Limits {
            max_words: 7,
            ..Limits::DEFAULT
        };
        for chunks in [1, 2, 5] {
            let o = LoadOptions {
                chunks: Some(chunks),
                ..limited(low)
            };
            let e = Corpus::from_joined_bytes(text, &o).unwrap_err();
            assert!(matches!(e, CorpusError::TooManyWords { limit: 7 }), "{e}");
            let e = Corpus::from_pair_bytes(b"a b\nc\n", b"x\na y\n", &o).unwrap_err();
            assert!(matches!(e, CorpusError::TooManyWords { limit: 7 }), "{e}");
        }
        // words of lines dropped by max_len do not count
        let o = LoadOptions {
            max_len: Some(1),
            ..limited(Limits {
                max_words: 6,
                ..Limits::DEFAULT
            })
        };
        assert_eq!(Corpus::from_joined_bytes(text, &o).unwrap().len(), 0);
    }

    #[test]
    fn line_limit() {
        let text = b"a ||| x\nb ||| y\nc ||| z\n";
        let at = |n| Limits {
            max_lines: n,
            ..Limits::DEFAULT
        };
        assert_eq!(
            Corpus::from_joined_bytes(text, &limited(at(3)))
                .unwrap()
                .len(),
            3
        );
        for chunks in [1, 2, 3] {
            let o = LoadOptions {
                chunks: Some(chunks),
                ..limited(at(2))
            };
            let e = Corpus::from_joined_bytes(text, &o).unwrap_err();
            assert!(matches!(e, CorpusError::TooManyLines { limit: 2 }), "{e}");
            // dropped lines count: line numbers refer to input lines
            let o = LoadOptions {
                max_len: Some(0),
                ..o
            };
            assert!(Corpus::from_joined_bytes(text, &o).is_err());
        }
    }

    #[test]
    fn line_token_limit_names_the_line_and_side() {
        let text = b"a b ||| x\nc ||| y z w\n";
        let o = |n| LoadOptions {
            chunks: Some(1),
            ..limited(Limits {
                max_line_tokens: n,
                ..Limits::DEFAULT
            })
        };
        assert!(Corpus::from_joined_bytes(text, &o(3)).is_ok());
        let e = Corpus::from_joined_bytes(text, &o(2)).unwrap_err();
        assert_eq!(
            e.to_string(),
            "line 2: the target side has 3 tokens, more than the limit of 2"
        );
        let e = Corpus::from_pair_bytes(b"a b\nc\n", b"x\ny z w\n", &o(2)).unwrap_err();
        assert!(matches!(
            e,
            CorpusError::LineTooLong {
                line: 2,
                side: "target",
                tokens: 3,
                ..
            }
        ));
        let e = Corpus::from_joined_bytes(text, &o(1)).unwrap_err();
        assert!(matches!(
            e,
            CorpusError::LineTooLong {
                line: 1,
                side: "source",
                tokens: 2,
                ..
            }
        ));
        // a line dropped by max_len is never stored, so it cannot exceed the limit
        let o = LoadOptions {
            max_len: Some(2),
            ..o(2)
        };
        assert_eq!(Corpus::from_joined_bytes(text, &o).unwrap().len(), 1);
    }

    #[test]
    fn paired_files_are_the_sides() {
        // review #9: a tab or `|||` inside a file stays on that file's side
        let c = Corpus::from_pair_bytes(b"a\tb c\n", b"x y z\n", &opts(1)).unwrap();
        assert_eq!(words(&c, &c.first, 0), ["a", "b", "c"]);
        assert_eq!(words(&c, &c.second, 0), ["x", "y", "z"]);
        let c = Corpus::from_pair_bytes(
            b"a ||| b\tc\r\n\t\t|||  |||\n",
            b"x\t|||\ty\r\nz\n",
            &opts(1),
        )
        .unwrap();
        assert_eq!(words(&c, &c.first, 0), ["a", "|||", "b", "c\r"]);
        assert_eq!(words(&c, &c.second, 0), ["x", "|||", "y\r"]);
        assert_eq!(words(&c, &c.first, 1), ["|||", "|||"]);
        assert_eq!(c.first.line(1), [SEP_ID, SEP_ID]);
        assert_eq!(words(&c, &c.second, 1), ["z"]);
        // the same ids as the joined file of tab- and `|||`-free sides
        let j = Corpus::from_joined_bytes(b"a b c ||| x y z\n", &opts(1)).unwrap();
        let p = Corpus::from_pair_bytes(b"a\tb c\n", b"x y z\n", &opts(1)).unwrap();
        same(&j, &p);
        // an empty side is reported as fast_align would show the joined line
        let c = Corpus::from_pair_bytes(b"\t\n", b"x\n", &opts(1)).unwrap();
        assert_eq!(c.empty_side_lines(), [(0, b"\t ||| x".to_vec())]);
    }

    #[test]
    fn inputs_that_are_not_regular_files_are_read_to_the_end() {
        let dir = std::env::temp_dir().join(format!("fa-corpus-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let empty = dir.join("empty");
        std::fs::write(&empty, b"").unwrap();
        assert!(Corpus::read_joined(&empty, &opts(1)).unwrap().is_empty());
        let e = Corpus::read_pair(&empty, &dir.join("missing"), &opts(1)).unwrap_err();
        assert!(matches!(e, CorpusError::Io(..)));
        #[cfg(unix)]
        {
            // a FIFO reports size 0 but has content
            let fifo = dir.join("fifo");
            let st = std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap();
            assert!(st.success());
            let f2 = fifo.clone();
            let w = std::thread::spawn(move || std::fs::write(f2, b"a b ||| x y\n").unwrap());
            let c = Corpus::read_joined(&fifo, &opts(1)).unwrap();
            w.join().unwrap();
            assert_eq!(
                (c.len(), c.first.n_tokens(), c.second.n_tokens()),
                (1, 2, 2)
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
