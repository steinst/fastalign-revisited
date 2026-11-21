"""Subword segmentation for word alignment.

fast_align only sees tokens, so subword alignment is: segment every word into pieces, align the
piece corpus with the ordinary trainer, and resolve the piece links back to words. Training,
projection and saving are done by :func:`fastalign.train` (``subword=...``) and the core; this
module holds the tokeniser side (SentencePiece) and, for tests and benchmarks, a pure-Python
reference of the projection.

* Segmentation is **per word**: each whitespace token is encoded on its own, so no piece crosses
  a word boundary. The first piece of a word starts with a word-start *marker* (``"▁"``).
* The piece → word map is kept by position (one array per line), never recovered from text.
* :func:`project` turns piece links into word links.

SentencePiece is an optional dependency (``pip install sentencepiece``); only
:func:`train_segmenter` and :class:`Segmenter` over a trained model need it.

Text handling. Corpora may be Latin-1, so each word is decoded as UTF-8 and, if that fails, as
Latin-1 -- for the tokeniser only; the position map never depends on the decoding. SentencePiece
normalisation is off (``identity``), so the pieces of a word give the decoded word back. A word
for which that is not true (it contains the marker character, an unseen character that is not
byte-fallback encoded, or a piece equal to the ``|||`` separator) is kept as **one piece**.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import tempfile
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

Link = tuple[int, int]

MARKER = "▁"  # "▁", SentencePiece's word-start marker
SEPARATOR = "|||"
_BYTE_PIECE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


# -- reading text ---------------------------------------------------------------------------


def decode_word(word: bytes) -> str:
    """UTF-8 if valid, else Latin-1 (never fails; used only to feed the tokeniser)."""
    try:
        return word.decode("utf-8")
    except UnicodeDecodeError:
        return word.decode("latin-1")


def read_lines(path: str) -> list[list[bytes]]:
    """Read a text file as lines of words (split on ASCII whitespace, so CRLF is harmless).

    A final line without a newline counts; a ``|||`` word is rejected, since it would be read
    as fast_align's separator in the encoded file.
    """
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    out = [line.split() for line in lines]
    sep = SEPARATOR.encode()
    for n, words in enumerate(out, 1):
        if sep in words:
            raise ValueError(f"line {n} of {path}: the token '|||' cannot be aligned")
    return out


# -- segmenters -----------------------------------------------------------------------------


class Segmenter(Protocol):
    """Anything that can cut one word into pieces. Must be deterministic."""

    def encode_word(self, word: bytes) -> list[str]: ...


class WholeWords:
    """The segmentation that never splits: one piece per word (the identity check)."""

    def __init__(self, marker: bool = False):
        self.marker = marker

    def encode_word(self, word: bytes) -> list[str]:
        text = decode_word(word)
        return [MARKER + text if self.marker else text]

    def scoped(self, cache_size: int | None = None) -> WholeWords:
        """Nothing to cache: the segmenter itself."""
        return self


def piece_text(pieces: Sequence[str]) -> str | None:
    """The text of a word's pieces, without the word-start marker; ``None`` if it is not a
    valid text (byte pieces that do not form UTF-8) or the first piece lacks the marker."""
    raw = bytearray()
    for p in pieces:
        m = _BYTE_PIECE.match(p)
        raw += bytes([int(m.group(1), 16)]) if m else p.encode("utf-8")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text[len(MARKER) :] if text.startswith(MARKER) else None


#: Words cached by one ``align_many`` call's segmenter (:meth:`SentencePieceSegmenter.scoped`):
#: the cache is emptied when it is full, so a call's memory is bounded (about 0.5 kB per entry)
#: and nothing outlives the call.
INFERENCE_CACHE_WORDS = 1 << 16
#: Words cached across single-pair ``Model.align`` calls by a loaded model's stored segmenters
#: (emptied when full: at most about 8 MB, whatever the queries).
MODEL_CACHE_WORDS = 1 << 14


class SentencePieceSegmenter:
    """Per-word encoding with a trained SentencePiece model (``model`` is the serialised proto).

    Results are cached per distinct word: the segmentation is a pure function of the word.
    ``cache_size`` bounds the cache (it is emptied when full); ``None`` leaves it unbounded,
    which suits a segmenter that lives for one run (training). A loaded model keeps one
    segmenter per tokeniser with a :data:`MODEL_CACHE_WORDS` bound (used by single-pair calls)
    and gives every ``align_many`` call its own :meth:`scoped` copy, so no cache grows with the
    queries a model has seen. The tokeniser is never rebuilt.
    """

    def __init__(self, model: bytes, cache_size: int | None = None, *, _processor=None):
        if _processor is None:
            import sentencepiece as spm

            _processor = spm.SentencePieceProcessor(model_proto=model)
        self.model = model
        self._sp = _processor
        self._cache: dict[bytes, list[str]] = {}
        self._cache_size = cache_size

    def scoped(self, cache_size: int | None = INFERENCE_CACHE_WORDS) -> SentencePieceSegmenter:
        """A segmenter that shares this one's tokeniser (not rebuilt) with an empty cache of its
        own, bounded by ``cache_size``."""
        return SentencePieceSegmenter(self.model, cache_size, _processor=self._sp)

    def encode_word(self, word: bytes) -> list[str]:
        cache = self._cache
        hit = cache.get(word)
        if hit is None:
            hit = self._encode(decode_word(word))
            if self._cache_size is not None and len(cache) >= self._cache_size:
                cache.clear()
            cache[word] = hit
        return hit

    def _encode(self, text: str) -> list[str]:
        whole = [MARKER + text]
        if MARKER in text or not text:
            return whole
        pieces = self._sp.encode(text, out_type=str)
        # SentencePiece may emit the marker alone as the first piece; that is fine (the word
        # still starts with it). Anything that does not give the word back stays whole.
        if not pieces or piece_text(pieces) != text or SEPARATOR in pieces:
            return whole
        if not pieces[0].startswith(MARKER) or any(p.startswith(MARKER) for p in pieces[1:]):
            return whole
        return pieces


@dataclass(frozen=True)
class SubwordConfig:
    """Tokeniser settings (reproducible: everything that SentencePiece samples is fixed).

    The defaults are the ones chosen from measurements in Step 13 (``benchmarks/SUBWORD.md``):
    BPE, one joint model of 8000 pieces, both sides segmented, no split at Unicode-script
    changes. ``num_threads=1`` is the default because multi-threaded training was measured to
    give different models on repeated runs (also for BPE); a given thread count is not
    reproducible.
    """

    model_type: str = "bpe"  # "bpe" | "unigram"
    vocab_size: int = 8000  # per model (for a joint model: of the combined model)
    joint: bool = True  # one model over both languages, else one per side
    segment: str = "both"  # "both" | "src" | "tgt": the other side stays word level
    byte_fallback: bool = True
    num_threads: int = 1
    # SentencePiece's pre-tokenisation inside a word: a forced split where the Unicode script
    # changes (letters vs. punctuation or U+FFFD) and around digits. The script split is off:
    # on a corpus whose accents became U+FFFD it cuts every accented word (SUBWORD.md).
    split_by_script: bool = False
    split_by_number: bool = True

    def __post_init__(self):
        if self.model_type not in ("unigram", "bpe"):
            raise ValueError(f"model_type must be 'unigram' or 'bpe', not {self.model_type!r}")
        if self.segment not in ("both", "src", "tgt"):
            raise ValueError("segment must be 'both', 'src' or 'tgt'")
        # with byte fallback the 256 byte pieces and the special pieces count toward the size
        floor = 260 if self.byte_fallback else 16
        if self.vocab_size < floor:
            raise ValueError(f"vocab_size must be at least {floor}")


def train_model(
    lines: Iterable[Iterable[bytes]],
    vocab_size: int,
    model_type: str,
    byte_fallback: bool,
    num_threads: int = 1,
    split_by_script: bool = True,
    split_by_number: bool = True,
) -> bytes:
    """Train one SentencePiece model on the given lines of words; returns the serialised proto.

    All sentences are used and not shuffled, normalisation is off, and the thread count is
    fixed, so two runs on the same text give byte-identical models (checked by a test).
    """
    import sentencepiece as spm

    seen = 0

    def sentences():
        nonlocal seen
        for words in lines:
            texts = [t for t in map(decode_word, words) if MARKER not in t]
            if texts:
                seen += 1
                yield " ".join(texts)

    out = io.BytesIO()
    try:
        _spm_train(
            spm,
            sentences(),
            out,
            vocab_size,
            model_type,
            byte_fallback,
            num_threads,
            split_by_script,
            split_by_number,
        )
    except RuntimeError:
        if seen == 0:
            # an empty corpus is legal for the aligner (it trains nothing); a tokeniser is not
            raise ValueError(
                "cannot train a subword tokeniser: the text has no words (an empty corpus or "
                "an empty side); train without subword"
            ) from None
        raise
    return out.getvalue()


def _spm_train(
    spm,
    sentences,
    out,
    vocab_size,
    model_type,
    byte_fallback,
    num_threads,
    split_by_script,
    split_by_number,
):
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences,
        model_writer=out,
        vocab_size=vocab_size,
        model_type=model_type,
        normalization_rule_name="identity",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        split_by_whitespace=True,
        split_by_unicode_script=split_by_script,
        split_by_number=split_by_number,
        byte_fallback=byte_fallback,
        character_coverage=1.0,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        max_sentence_length=1 << 22,
        num_threads=num_threads,
        hard_vocab_limit=False,
        minloglevel=2,
    )


def train_segmenters(
    src: Sequence[Sequence[bytes]], tgt: Sequence[Sequence[bytes]], config: SubwordConfig
) -> tuple[Segmenter, Segmenter]:
    """Segmenters for the source and target side. A side not in ``config.segment`` gets
    :class:`WholeWords` (plain words, no marker)."""
    args = (
        config.vocab_size,
        config.model_type,
        config.byte_fallback,
        config.num_threads,
        config.split_by_script,
        config.split_by_number,
    )
    do_src = config.segment in ("both", "src")
    do_tgt = config.segment in ("both", "tgt")
    if config.joint and do_src and do_tgt:
        seg = SentencePieceSegmenter(train_model(list(src) + list(tgt), *args))
        return seg, seg
    a = SentencePieceSegmenter(train_model(src, *args)) if do_src else WholeWords()
    b = SentencePieceSegmenter(train_model(tgt, *args)) if do_tgt else WholeWords()
    return a, b


# -- encoding ---------------------------------------------------------------------------------


@dataclass
class Encoded:
    """One side of a corpus cut into pieces: ``pieces[l]`` are the tokens of line ``l`` and
    ``word_of[l][k]`` the index (in the line's words) of the word piece ``k`` belongs to."""

    pieces: list[list[str]]
    word_of: list[array]
    n_words: list[int]

    def __len__(self) -> int:
        return len(self.pieces)


def encode(lines: Sequence[Sequence[bytes]], segmenter: Segmenter) -> Encoded:
    """Segment every word of every line; build the position map as it goes."""
    pieces: list[list[str]] = []
    maps: list[array] = []
    counts: list[int] = []
    for words in lines:
        line: list[str] = []
        m = array("I")
        for w, word in enumerate(words):
            ps = segmenter.encode_word(word)
            line.extend(ps)
            m.extend([w] * len(ps))
        pieces.append(line)
        maps.append(m)
        counts.append(len(words))
    return Encoded(pieces, maps, counts)


def write_pair_file(path: str, src: Encoded, tgt: Encoded) -> None:
    """Write the piece corpus as ``src pieces ||| tgt pieces`` lines for the aligner."""
    if len(src) != len(tgt):
        raise ValueError(f"line counts differ ({len(src)} and {len(tgt)})")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(
            " ".join(s) + " ||| " + " ".join(t) + "\n" for s, t in zip(src.pieces, tgt.pieces)
        )


def split_into_words(pieces: Sequence[str], marker: str = MARKER) -> list[int]:
    """Word index of every piece of an already segmented line: a new word starts at each piece
    that begins with ``marker`` (what the native tool will do without a tokeniser)."""
    out: list[int] = []
    w = -1
    for p in pieces:
        if p.startswith(marker) or w < 0:
            w += 1
        out.append(w)
    return out


# -- projection -------------------------------------------------------------------------------


def project(
    links: Iterable[Link],
    src_map: Sequence[int],
    tgt_map: Sequence[int],
    by: str = "tgt",
) -> list[Link]:
    """Piece links -> word links: ``(p, q)`` becomes ``(src_map[p], tgt_map[q])``; duplicates
    are dropped and the result is sorted by target word (``by="tgt"``, the order fast_align
    prints a forward direction in) or by source word (``by="src"``, for a reverse direction),
    ties by the other side. Links that point outside a map are an error.
    """
    seen: set[Link] = set()
    for p, q in links:
        if not (0 <= p < len(src_map) and 0 <= q < len(tgt_map)):
            raise ValueError(f"link {p}-{q} is outside the line ({len(src_map)}, {len(tgt_map)})")
        seen.add((src_map[p], tgt_map[q]))
    if by == "tgt":
        return sorted(seen, key=lambda l: (l[1], l[0]))
    if by == "src":
        return sorted(seen)
    raise ValueError("by must be 'src' or 'tgt'")


def project_many(
    links: Sequence[Iterable[Link]],
    src: Encoded,
    tgt: Encoded,
    by: str = "tgt",
) -> list[list[Link]]:
    """:func:`project` for every line of a corpus."""
    if not (len(links) == len(src) == len(tgt)):
        raise ValueError("number of alignment lines and corpus lines differ")
    return [project(l, s, t, by) for l, s, t in zip(links, src.word_of, tgt.word_of)]


# -- the product: what ``fastalign.train(subword=...)`` and a saved model use -----------------


def require_sentencepiece():
    """Import ``sentencepiece`` or fail with the way to install it."""
    try:
        import sentencepiece
    except ImportError:
        raise ImportError(
            "subword alignment needs the optional dependency sentencepiece: "
            "pip install 'fastalign-revisited[subword]' (from a checkout: "
            "pip install '.[subword]')"
        ) from None
    return sentencepiece


@dataclass(frozen=True)
class PreSegmented:
    """The corpus is already cut into pieces; only the word boundaries are needed.

    ``marker`` is the word-start marker of the pieces (a piece beginning with it begins a word:
    SentencePiece's ``▁``), or, with ``continuation=True``, the continuation marker of
    subword-nmt (a piece ending with it continues into the next piece, ``@@``). No tokeniser is
    trained or stored; text given to :meth:`fastalign.Model.align_many` must be segmented in the
    same way.
    """

    marker: str = MARKER
    continuation: bool = False

    def __post_init__(self):
        if not self.marker or any(c.isspace() for c in self.marker):
            raise ValueError("the subword marker must be non-empty and contain no whitespace")

    @property
    def kind(self) -> str:
        return "continue" if self.continuation else "start"


def resolve_config(subword) -> SubwordConfig | PreSegmented | None:
    """``None``, ``"auto"`` (the chosen defaults), a :class:`SubwordConfig` or a
    :class:`PreSegmented`."""
    if subword is None or isinstance(subword, (SubwordConfig, PreSegmented)):
        return subword
    if subword == "auto":
        return SubwordConfig()
    raise ValueError(
        f"subword must be None, 'auto', a SubwordConfig or a PreSegmented, not {subword!r}"
    )


def _wholewords_marked() -> WholeWords:
    return WholeWords(marker=True)


def make_segmenters(
    blobs: Sequence[bytes], layout: str, cache_size: int | None = None
) -> tuple[Segmenter, Segmenter]:
    """The (source, target) segmenters of stored tokeniser models. ``layout`` is ``"joint"``
    (one model for both sides) or ``"separate"`` (source, target; an empty blob is a side that
    stays word level). ``cache_size`` bounds each word cache (``None``: unbounded)."""
    require_sentencepiece()
    if layout == "joint" and len(blobs) == 1:
        seg = SentencePieceSegmenter(blobs[0], cache_size)
        return seg, seg
    if layout == "separate" and len(blobs) == 2:
        return tuple(  # type: ignore[return-value]
            SentencePieceSegmenter(b, cache_size) if b else _wholewords_marked() for b in blobs
        )
    raise ValueError("the model's tokeniser settings are inconsistent")


def settings_json(config: SubwordConfig, layout: str) -> bytes:
    return json.dumps({"format": 1, "layout": layout, **asdict(config)}, sort_keys=True).encode()


def read_settings(settings: bytes) -> dict:
    try:
        d = json.loads(settings.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ValueError("the model's subword settings are not readable") from None
    if (
        not isinstance(d, dict)
        or d.get("format") != 1
        or d.get("layout") not in ("joint", "separate")
    ):
        raise ValueError("the model's subword settings are not readable")
    return d


def _check_words(words: list[bytes], where: str) -> list[bytes]:
    if SEPARATOR.encode() in words:
        raise ValueError(f"{where}: the token '|||' cannot be aligned")
    return words


# The core's word rules (``corpus.rs``), so that segmentation sees exactly the words and sides
# of the word-level workflow: only space and tab separate words (CR and other control bytes
# are part of a word); in a joined line a tab is also a separator, and the first separator
# (tab or the word ``|||``) splits source from target; two files are the two sides.
_WORD = re.compile(rb"[^ \t]+")
_WORD_OR_TAB = re.compile(rb"\t|[^ \t]+")
_SEP = SEPARATOR.encode()


def side_words(line: bytes) -> list[bytes]:
    """The words of one line of a one-side file (no trailing newline), as the core reads it."""
    return _WORD.findall(line)


def joined_words(line: bytes) -> tuple[list[bytes], list[bytes]]:
    """The (source, target) words of one ``source ||| target`` line, as the core reads it.
    Separators after the first stay in the target as the word ``|||``."""
    toks = [_SEP if t == b"\t" else t for t in _WORD_OR_TAB.findall(line)]
    if _SEP not in toks:
        return toks, []
    i = toks.index(_SEP)
    return toks[:i], toks[i + 1 :]


def _raw_lines(source, spool_dir: str | None) -> tuple[Callable[[], Iterator[bytes]], str]:
    """A re-iterable view of the lines (without ``\n``) of a file or of standard input already
    read, and its name for messages. A path that is not a regular file (a pipe, FIFO or process
    substitution) can be read only once: it is copied to ``spool_dir`` first."""
    from fastalign._input import StdinText

    if isinstance(source, StdinText):
        data = bytes(source)
        name = "<stdin>"

        def opener():
            return io.BytesIO(data)
    else:
        name = os.fspath(source)
        path = name
        if not stat.S_ISREG(os.stat(path).st_mode):
            if spool_dir is None:
                raise ValueError(f"{path}: not a regular file and nowhere to spool it")
            fd, path = tempfile.mkstemp(dir=spool_dir, prefix="spool-")
            with open(name, "rb") as fin, os.fdopen(fd, "wb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)

        def opener():
            return open(path, "rb")

    def gen() -> Iterator[bytes]:
        with opener() as f:
            for line in f:
                yield line[:-1] if line.endswith(b"\n") else line

    return gen, name


def _lines_of(source, name: str, spool_dir: str | None) -> Callable[[], Iterator[list[bytes]]]:
    """A re-iterable view of one side of a corpus: a file (read lazily, line by line, as bytes,
    words as in the core's paired reader) or a list of sentences (strings split on whitespace,
    token lists)."""
    from fastalign._input import sentence_to_line, text_source

    src = text_source(source, name)
    if not isinstance(src, list):
        lines, where = _raw_lines(src, spool_dir)

        def from_file() -> Iterator[list[bytes]]:
            for n, line in enumerate(lines(), 1):
                yield _check_words(side_words(line), f"line {n} of {where}")

        return from_file

    words = [sentence_to_line(s, name).encode("utf-8").split() for s in src]
    return lambda: iter(words)


def _joined_sides(source, spool_dir: str | None):
    """The two sides of a ``source ||| target`` file (tab also separates, as in fast_align)."""
    lines, where = _raw_lines(source, spool_dir)

    def side(k: int) -> Callable[[], Iterator[list[bytes]]]:
        def gen() -> Iterator[list[bytes]]:
            for n, line in enumerate(lines(), 1):
                yield _check_words(joined_words(line)[k], f"line {n} of {where}")

        return gen

    return side(0), side(1)


def corpus_sides(src, tgt, spool_dir: str | None = None):
    """Re-iterable (source, target) word-line views of the arguments of ``fastalign.train``,
    with the sides and words of the word-level workflow. Inputs that can be read only once are
    copied to ``spool_dir``."""
    from fastalign._input import StdinText

    if tgt is None:
        if not isinstance(src, (str, os.PathLike, StdinText)):
            raise TypeError("a single text must be the path of a file of `source ||| target` lines")
        return _joined_sides(src, spool_dir)
    return _lines_of(src, "src", spool_dir), _lines_of(tgt, "tgt", spool_dir)


def train_stream_segmenters(
    sides, config: SubwordConfig
) -> tuple[Segmenter, Segmenter, list[bytes], str]:
    """Train the tokeniser(s) from streams of lines (no corpus is held in memory). Returns the
    segmenters, the serialised models in storage order and the layout. A side that
    ``config.segment`` leaves out stays word level, marked so that word boundaries are
    recoverable."""
    require_sentencepiece()
    src_lines, tgt_lines = sides
    args = (
        config.vocab_size,
        config.model_type,
        config.byte_fallback,
        config.num_threads,
        config.split_by_script,
        config.split_by_number,
    )
    do_src = config.segment in ("both", "src")
    do_tgt = config.segment in ("both", "tgt")
    if config.joint and do_src and do_tgt:
        from itertools import chain

        blob = train_model(chain(src_lines(), tgt_lines()), *args)
        seg = SentencePieceSegmenter(blob)
        return seg, seg, [blob], "joint"
    blobs = [
        train_model(src_lines(), *args) if do_src else b"",
        train_model(tgt_lines(), *args) if do_tgt else b"",
    ]
    a, b = make_segmenters(blobs, "separate")
    return a, b, blobs, "separate"


def scoped_segmenters(
    segs: tuple[Segmenter, Segmenter], cache_size: int | None = INFERENCE_CACHE_WORDS
) -> tuple[Segmenter, Segmenter]:
    """One call's (source, target) segmenters: :meth:`SentencePieceSegmenter.scoped` copies of a
    model's stored ones (a joint tokeniser stays one segmenter for both sides)."""
    a, b = segs
    sa = a.scoped(cache_size)  # type: ignore[attr-defined]
    return (sa, sa) if b is a else (sa, b.scoped(cache_size))  # type: ignore[attr-defined]


def write_piece_sides(sides, segs: tuple[Segmenter, Segmenter], paths: tuple[str, str]) -> None:
    """Stream each side of ``sides`` (as :func:`corpus_sides` gives them), cut into pieces, to its
    own file (one line of pieces per input line): nothing is held in memory but a line."""
    for gen, seg, path in zip(sides, segs, paths):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for words in gen():
                f.write(encode_line(words, seg))
                f.write("\n")


def encode_line(words: Sequence[bytes], seg: Segmenter) -> str:
    out: list[str] = []
    for w in words:
        out.extend(seg.encode_word(w))
    return " ".join(out)


def write_piece_corpus(path: str, sides, segs: tuple[Segmenter, Segmenter]) -> int:
    """Stream the encoded corpus to ``path`` as ``source pieces ||| target pieces`` lines.
    Returns the number of lines; the two sides must have the same number of lines."""
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        src_it, tgt_it = sides[0](), sides[1]()
        _end = object()
        while True:
            a, b = next(src_it, _end), next(tgt_it, _end)
            if a is _end and b is _end:
                break
            if a is _end or b is _end:
                raise ValueError("the source and target have different numbers of lines")
            f.write(encode_line(a, segs[0]) + " ||| " + encode_line(b, segs[1]) + "\n")
            n += 1
    return n
