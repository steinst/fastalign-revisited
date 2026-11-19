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
//! Two line-aligned files are read as if each line pair were joined into one line
//! `SRC ||| TGT` (so a tab or `|||` inside the source line splits there, exactly as fast_align
//! would on the joined file).
//!
//! Reading is parallel: the input is cut into line-aligned chunks, each chunk is tokenised and
//! interned with a chunk-local dictionary, and the local dictionaries are merged in chunk order.
//! Because a chunk's local ids are in first-occurrence order, the merged ids are the global
//! first-occurrence ids whatever the number of chunks or threads.

use std::fmt;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};

use memchr::{memchr, memchr_iter};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

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
    LineCountMismatch { src: usize, tgt: usize },
    TooManyTokens,
}

impl fmt::Display for CorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CorpusError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            CorpusError::LineCountMismatch { src, tgt } => write!(
                f,
                "source and target files have different numbers of lines ({src} vs {tgt})"
            ),
            CorpusError::TooManyTokens => {
                write!(f, "corpus has too many distinct words (> 2^32 - 2)")
            }
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
}

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
    /// 1-based input line number of each kept line, when lines were dropped (`max_len`).
    line_numbers: Option<Vec<u32>>,
    /// Raw input lines (as fast_align would see them) of the lines with an empty side, for
    /// fast_align's `Error in line` messages: (0-based kept-line index, bytes).
    empty_side_lines: Vec<(usize, Vec<u8>)>,
    /// Number of input lines read (before `max_len` filtering).
    n_input_lines: usize,
}

impl Corpus {
    /// Read one file with `SRC ||| TGT` lines (fast_align's format).
    pub fn read_joined(path: &Path, opts: &LoadOptions) -> Result<Self, CorpusError> {
        let buf = map_file(path)?;
        Ok(Self::from_joined_bytes(buf.as_slice(), opts))
    }

    /// Read two line-aligned files.
    pub fn read_pair(src: &Path, tgt: &Path, opts: &LoadOptions) -> Result<Self, CorpusError> {
        let a = map_file(src)?;
        let b = map_file(tgt)?;
        Self::from_pair_bytes(a.as_slice(), b.as_slice(), opts)
    }

    /// Parse an in-memory `SRC ||| TGT` text.
    pub fn from_joined_bytes(text: &[u8], opts: &LoadOptions) -> Self {
        let n_chunks = n_chunks(opts, text.len());
        let bounds = chunk_bounds(text, n_chunks);
        let chunks: Vec<ChunkInput> = bounds
            .windows(2)
            .map(|w| ChunkInput::Joined(&text[w[0]..w[1]]))
            .collect();
        build(chunks, opts)
    }

    /// Parse two in-memory line-aligned texts.
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
        Ok(build(chunks, opts))
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

    /// Make the line-per-pair `text` of a filtered corpus (`max_len`) one line per *input*
    /// line: dropped lines become empty lines. An unfiltered corpus returns `text` borrowed.
    pub fn expand_lines<'a>(&self, text: &'a [u8]) -> std::borrow::Cow<'a, [u8]> {
        if !self.is_filtered() {
            return std::borrow::Cow::Borrowed(text);
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
            out.extend_from_slice(lines.next().unwrap_or(b"\n"));
            next += 1;
        }
        while next <= self.n_input_lines() {
            out.push(b'\n');
            next += 1;
        }
        std::borrow::Cow::Owned(out)
    }

    /// Whether lines were dropped by `max_len`.
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

enum Buf {
    Map(memmap2::Mmap),
    Empty,
}

impl Buf {
    fn as_slice(&self) -> &[u8] {
        match self {
            Buf::Map(m) => m,
            Buf::Empty => &[],
        }
    }
}

fn map_file(path: &Path) -> Result<Buf, CorpusError> {
    let err = |e| CorpusError::Io(path.to_path_buf(), e);
    let file = File::open(path).map_err(err)?;
    let len = file.metadata().map_err(err)?.len();
    if len == 0 {
        return Ok(Buf::Empty);
    }
    // SAFETY: the file is only read; concurrent modification by another process is the
    // caller's responsibility, as for any memory-mapped reader.
    let map = unsafe { memmap2::Mmap::map(&file) }.map_err(err)?;
    Ok(Buf::Map(map))
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

/// A token of a line: a word (byte range in the line) or a separator.
#[derive(Clone, Copy)]
enum Tok {
    Word(u32, u32),
    Sep,
}

/// Tokenise `line` per spec §3, appending to `out`. The word `|||` is returned as `Sep`.
fn tokenize(line: &[u8], out: &mut Vec<Tok>) {
    let mut start = None;
    for (p, &c) in line.iter().enumerate() {
        if c == b' ' || c == b'\t' {
            if let Some(s) = start.take() {
                out.push(word_tok(line, s, p));
            }
            if c == b'\t' {
                out.push(Tok::Sep);
            }
        } else if start.is_none() {
            start = Some(p);
        }
    }
    if let Some(s) = start {
        out.push(word_tok(line, s, line.len()));
    }
}

#[inline]
fn word_tok(line: &[u8], s: usize, e: usize) -> Tok {
    if &line[s..e] == SEP_WORD {
        Tok::Sep
    } else {
        Tok::Word(s as u32, e as u32)
    }
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
}

struct LocalDict<'a> {
    map: FxHashMap<&'a [u8], u32>,
    words: Vec<&'a [u8]>,
}

impl<'a> LocalDict<'a> {
    fn new() -> Self {
        let mut map = FxHashMap::default();
        map.insert(NULL_WORD, 0);
        map.insert(SEP_WORD, 1);
        LocalDict {
            map,
            words: Vec::new(),
        }
    }

    #[inline]
    fn intern(&mut self, w: &'a [u8]) -> u32 {
        let next = self.words.len() as u32 + 2;
        *self.map.entry(w).or_insert_with(|| {
            self.words.push(w);
            next
        })
    }
}

fn process_chunk<'a>(input: &ChunkInput<'a>, opts: &LoadOptions) -> ChunkOut<'a> {
    let mut dict = LocalDict::new();
    let mut out = ChunkOut {
        words: Vec::new(),
        first: Vec::new(),
        first_len: Vec::new(),
        second: Vec::new(),
        second_len: Vec::new(),
        kept: opts.max_len.map(|_| Vec::new()),
        n_input: 0,
        empty_side: Vec::new(),
    };
    let mut toks: Vec<Tok> = Vec::new();
    let mut words: Vec<&'a [u8]> = Vec::new(); // the line's tokens as slices, Sep = "|||"
    let mut handle = |line_parts: &[&'a [u8]], out: &mut ChunkOut<'a>, dict: &mut LocalDict<'a>| {
        toks.clear();
        words.clear();
        for (k, part) in line_parts.iter().enumerate() {
            if k > 0 {
                toks.push(Tok::Sep);
                words.push(SEP_WORD);
            }
            let before = toks.len();
            tokenize(part, &mut toks);
            for t in &toks[before..] {
                words.push(match *t {
                    Tok::Word(s, e) => &part[s as usize..e as usize],
                    Tok::Sep => SEP_WORD,
                });
            }
        }
        let split = toks.iter().position(|t| matches!(t, Tok::Sep));
        let (n_first, n_second) = match split {
            Some(p) => (p, toks.len() - p - 1),
            None => (toks.len(), 0),
        };
        let line_idx = out.n_input;
        out.n_input += 1;
        if let Some(max) = opts.max_len {
            if n_first > max || n_second > max {
                return;
            }
            out.kept.as_mut().unwrap().push(line_idx as u32);
        }
        let kept_idx = out.first_len.len();
        if n_first == 0 || n_second == 0 {
            let mut raw = Vec::new();
            for (k, part) in line_parts.iter().enumerate() {
                if k > 0 {
                    raw.extend_from_slice(b" ||| ");
                }
                raw.extend_from_slice(part);
            }
            out.empty_side.push((kept_idx, raw));
        }
        for &w in &words[..n_first] {
            out.first.push(dict.intern(w));
        }
        if let Some(p) = split {
            for (t, &w) in toks[p + 1..].iter().zip(&words[p + 1..]) {
                out.second.push(match t {
                    Tok::Sep => 1,
                    Tok::Word(..) => dict.intern(w),
                });
            }
        }
        out.first_len.push(n_first as u32);
        out.second_len.push(n_second as u32);
    };
    match *input {
        ChunkInput::Joined(text) => {
            for line in lines(text) {
                handle(&[line], &mut out, &mut dict);
            }
        }
        ChunkInput::Pair(a, b) => {
            let mut lb = lines(b);
            for la in lines(a) {
                let lb = lb.next().unwrap_or(&[]);
                handle(&[la, lb], &mut out, &mut dict);
            }
        }
    }
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

fn build(chunks: Vec<ChunkInput>, opts: &LoadOptions) -> Corpus {
    let outs: Vec<ChunkOut> = chunks.par_iter().map(|c| process_chunk(c, opts)).collect();

    // Global dictionary in chunk order: first-occurrence ids.
    let mut map: FxHashMap<&[u8], u32> = FxHashMap::default();
    let mut words: Vec<&[u8]> = vec![BAD0_WORD, NULL_WORD, SEP_WORD];
    map.insert(NULL_WORD, NULL_ID);
    map.insert(SEP_WORD, SEP_ID);
    let remaps: Vec<Vec<u32>> = outs
        .iter()
        .map(|o| {
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
            r
        })
        .collect();
    drop(map);
    let vocab = make_vocab(&words);

    let first = concat_side(&outs, &remaps, |o| (&o.first, &o.first_len));
    let second = concat_side(&outs, &remaps, |o| (&o.second, &o.second_len));

    let mut line_numbers = opts.max_len.map(|_| Vec::new());
    let mut empty_side_lines = Vec::new();
    let mut input_base = 0usize;
    let mut kept_base = 0usize;
    for o in &outs {
        if let (Some(v), Some(k)) = (line_numbers.as_mut(), o.kept.as_ref()) {
            v.extend(k.iter().map(|&i| (input_base + i as usize + 1) as u32));
        }
        empty_side_lines.extend(o.empty_side.iter().map(|(k, l)| (kept_base + k, l.clone())));
        input_base += o.n_input;
        kept_base += o.first_len.len();
    }
    Corpus {
        vocab,
        first,
        second,
        line_numbers,
        empty_side_lines,
        n_input_lines: input_base,
    }
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

    #[test]
    fn ids_in_first_occurrence_order() {
        let c = Corpus::from_joined_bytes(b"a b ||| x a\nb c ||| y\n", &opts(1));
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
        let c = Corpus::from_joined_bytes(b"a\tb ||| x\nc ||| d ||| e\n\n ||| y\nz\n", &opts(1));
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
        let c = Corpus::from_joined_bytes(b"  a\x0bb  ||| x\r", &opts(1));
        assert_eq!(c.len(), 1);
        assert_eq!(words(&c, &c.first, 0), ["a\u{b}b"]);
        assert_eq!(words(&c, &c.second, 0), ["x\r"]);
        assert!(Corpus::from_joined_bytes(b"", &opts(1)).is_empty());
        assert_eq!(Corpus::from_joined_bytes(b"\n", &opts(1)).len(), 1);
    }

    #[test]
    fn eps_token_is_null() {
        let c = Corpus::from_joined_bytes(b"<eps> a ||| |||\n", &opts(1));
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
            let base = Corpus::from_joined_bytes(&joined, &opts(1));
            for chunks in [2, 3, 7, 64, 1000] {
                same(&base, &Corpus::from_joined_bytes(&joined, &opts(chunks)));
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
        };
        let c = Corpus::from_joined_bytes(b"a ||| x\nlong long long ||| y\nb ||| z\n", &o);
        assert_eq!(c.len(), 2);
        assert_eq!(c.n_input_lines(), 3);
        assert_eq!((c.line_number(0), c.line_number(1)), (1, 3));
        let joined = Corpus::from_joined_bytes(b"a ||| x\nb ||| z\n", &opts(1));
        assert_eq!(c.vocab, joined.vocab);
    }
}
