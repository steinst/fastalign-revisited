"""Training, the :class:`Model` class and symmetrisation."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastalign import _core
from fastalign import subword as _sw
from fastalign._input import sentence_to_line, text_source
from fastalign.subword import PreSegmented, SubwordConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fastalign._input import Sentence, TextSource

__all__ = [
    "METHODS",
    "Alignments",
    "Link",
    "Model",
    "PreSegmented",
    "Progress",
    "SubwordConfig",
    "TableAlignment",
    "align_with_table",
    "invert",
    "parse_pharaoh",
    "symmetrize",
    "symmetrize_many",
    "train",
]

logger = logging.getLogger("fastalign")

#: An alignment link ``(source_position, target_position)``, both 0-based.
Link = tuple[int, int]

#: Sequence of alignments (one list of links per sentence pair); see :meth:`Model.alignments`.
Alignments = _core.Alignments

#: Symmetrisation heuristics, named as in ``atools -c``. ``"intersection"`` is accepted for
#: ``"intersect"``.
METHODS: tuple[str, ...] = (
    "intersect",
    "union",
    "grow-diag",
    "grow-diag-final",
    "grow-diag-final-and",
)


@dataclass(frozen=True)
class Progress:
    """One progress event of :func:`train`: an iteration of one direction has finished.

    Attributes:
        direction: ``"fwd"`` (source generates target) or ``"inv"``.
        iteration: Iteration just finished (1-based).
        iterations: Iterations planned.
        log_likelihood: Natural-log likelihood of the corpus in this iteration's E-step.
        perplexity: ``2 ** (-log2 likelihood / target tokens)``.
        p0: Posterior mass of the NULL word per target token (0 in the final iteration, which
            only decodes).
        tension: Diagonal tension after this iteration.
        seconds: Time since this direction started training.
    """

    direction: str
    iteration: int
    iterations: int
    log_likelihood: float
    perplexity: float
    p0: float
    tension: float
    seconds: float

    def __str__(self) -> str:
        return (
            f"[{self.direction}] iteration {self.iteration}/{self.iterations}: "
            f"log-likelihood {self.log_likelihood:.1f}, perplexity {self.perplexity:.4f}, "
            f"p0 {self.p0:.4f}, tension {self.tension:.4f} ({self.seconds:.1f} s)"
        )


@dataclass(frozen=True)
class TableAlignment:
    """Result of :func:`align_with_table` (fast_align's ``-f`` mode).

    Attributes:
        output: ``SRC ||| TGT ||| LINKS ||| LOGPROB`` lines, as fast_align prints them.
        log: What fast_align prints on stderr (``Loaded N translation parameters.``, error
            messages, ``TOTAL LOG PROB``).
        failed_line: 1-based number of a line with an empty side, which stops fast_align
            (``output`` then ends with that line, unfinished); ``None`` if all lines were aligned.
    """

    output: bytes
    log: str
    failed_line: int | None


def _check_threads(threads: int) -> int:
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0:
        raise ValueError(f"threads must be a non-negative integer (0 = all cores), got {threads!r}")
    return threads


def _method(name: str | None) -> str | None:
    if name == "intersection":
        return "intersect"
    return name


def _check_direction(direction: str) -> str:
    if direction not in ("fwd", "inv"):
        raise ValueError(f"direction must be 'fwd' or 'inv', not {direction!r}")
    return direction


class Model:
    """A trained word alignment model.

    Create one with :func:`train` or :meth:`Model.load`. A model holds fast_align's translation
    table for the *forward* direction (source words generate target words) and/or the *inverse*
    direction, one shared vocabulary, and the final tension and ``p0`` of each direction.

    A model that has just been trained also holds the alignments of its training corpus
    (:meth:`alignments`, :meth:`write_alignments`). A model read back with :meth:`load` does
    not; use :meth:`align` and :meth:`align_many` to align sentences with it.

    Instances are immutable and may be shared between threads.
    """

    def __init__(self, _inner: _core.Model) -> None:
        self._inner = _inner
        self._segmenters: tuple[_sw.Segmenter, _sw.Segmenter] | None = None

    # -- introspection ------------------------------------------------------------------

    @property
    def directions(self) -> tuple[str, ...]:
        """The directions this model holds: ``("fwd", "inv")``, ``("fwd",)`` or ``("inv",)``."""
        return tuple(self._inner.directions)

    @property
    def vocab_size(self) -> int:
        """Number of distinct words seen in training (one dictionary for both languages)."""
        return self._inner.vocab_size

    @property
    def has_training_alignments(self) -> bool:
        """Whether :meth:`alignments` is available (true right after :func:`train`)."""
        return self._inner.has_training_alignments

    @property
    def n_lines(self) -> int | None:
        """Number of lines of the training corpus, or ``None`` for a loaded model."""
        return self._inner.n_lines

    @property
    def subword(self) -> dict | None:
        """How this model segments words, or ``None`` for a word-level model: ``marker`` and
        ``kind`` (``"start"``: a piece beginning with the marker begins a word; ``"continue"``:
        a piece ending with it continues), ``tokenizer`` (whether a tokeniser is stored, so that
        :meth:`align_many` segments new text) and ``settings`` (the tokeniser's settings)."""
        info = self._inner.subword
        if info is None:
            return None
        kind, marker, blobs, settings = info
        return {
            "kind": kind,
            "marker": marker,
            "tokenizer": bool(blobs) and any(blobs),
            "settings": _sw.read_settings(settings) if settings else None,
        }

    def _text_segmenters(self) -> tuple[_sw.Segmenter, _sw.Segmenter] | None:
        """The stored tokenisers (built once), or ``None`` if text is not segmented here."""
        info = self._inner.subword
        if info is None or not info[2] or not any(info[2]):
            return None
        if self._segmenters is None:
            settings = _sw.read_settings(info[3])
            self._segmenters = _sw.make_segmenters(info[2], settings["layout"])
        return self._segmenters

    def _segment(self, src: TextSource, tgt: TextSource | None):
        """Cut new text into pieces with the stored tokeniser (as two lists of lines)."""
        segs = self._text_segmenters()
        assert segs is not None
        a, b = _sw.corpus_sides(src, tgt)
        return (
            [_sw.encode_line(w, segs[0]) for w in a()],
            [_sw.encode_line(w, segs[1]) for w in b()],
        )

    def parameters(self, direction: str = "fwd") -> dict[str, float | int]:
        """Final parameters of one direction: ``tension``, ``p0``, ``iterations`` and
        ``mean_srclen_multiplier`` (the target/source length ratio seen in training)."""
        t, p0, iters, mult = self._inner.parameters(_check_direction(direction))
        return {"tension": t, "p0": p0, "iterations": iters, "mean_srclen_multiplier": mult}

    def training_log(self, direction: str = "fwd") -> str:
        """fast_align's statistics for one direction of the training run: log likelihood,
        perplexity, posterior p0 and tension per iteration, and any ``Error in line`` messages
        (what fast_align prints on stderr, without the ``ARG=`` echo)."""
        raw = self._inner.training_log(_check_direction(direction))
        return raw.decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return f"<fastalign.Model directions={self.directions!r} vocab={self.vocab_size}>"

    # -- persistence --------------------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the model to ``path`` (through a temporary file, renamed when complete).
        Releases the GIL while writing.

        Raises:
            OSError: the file cannot be written.
        """
        self._inner.save(os.fspath(path))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Model:
        """Read a model written by :meth:`save`.

        Raises:
            FileNotFoundError: ``path`` does not exist.
            ValueError: the file is not a fastalign model, is truncated or corrupt, or has an
                unsupported format version.
        """
        return cls(_core.Model.load(os.fspath(path)))

    def write_table(
        self,
        path: str | os.PathLike[str],
        direction: str | None = None,
        beam_threshold: float = -4.0,
        threads: int = 0,
    ) -> None:
        """Write one direction's translation table in fast_align's ``-p`` text format
        (``source<TAB>target<TAB>log probability``), byte-identical to what fast_align writes.

        Args:
            direction: ``"fwd"`` or ``"inv"``; default the first direction of the model.
            beam_threshold: fast_align's ``-t``: only entries whose log probability is at least
                this are written.
            threads: Worker threads, 0 for all cores.
        """
        d = direction if direction is not None else self.directions[0]
        self._inner.write_table(
            os.fspath(path), _check_direction(d), float(beam_threshold), _check_threads(threads)
        )

    # -- alignments ---------------------------------------------------------------------

    def alignments(
        self,
        symmetrize: str | None = None,
        direction: str | None = None,
        pieces: bool = False,
    ) -> Alignments:
        """Alignments of the training corpus, one entry per input line.

        The result behaves like a read-only list: ``len()``, indexing, slicing and iteration
        give lists of ``(source, target)`` links (0-based positions); ``.pharaoh(i)`` gives the
        text of line ``i``. Lines without alignment (an empty side, or dropped by ``max_len``)
        give empty lists. Links of one direction come in target order, symmetrised links
        sorted.

        Args:
            symmetrize: One of :data:`METHODS`; needs a model of both directions.
            direction: ``"fwd"`` or ``"inv"``: the links of one direction without
                symmetrisation.
            pieces: For a model trained with ``subword`` and ``subword_output=True``: the
                piece-level links of one direction (positions are pieces) instead of the word
                links.

        With neither argument, a model of both directions is symmetrised with
        ``grow-diag-final-and`` and a model of one direction returns that direction.

        Raises:
            ValueError: the model was loaded from a file (no training alignments), both
                arguments were given, or the requested direction is not in the model.
        """
        return self._inner.training_alignments(_method(symmetrize), direction, bool(pieces))

    def write_alignments(
        self,
        path: str | os.PathLike[str],
        format: str = "pharaoh",
        symmetrize: str | None = None,
        direction: str | None = None,
        pieces: bool = False,
    ) -> None:
        """Write the training alignments to ``path``: one line of ``i-j`` links per input line
        (source-target orientation), the format ``atools`` reads and writes. If the model was
        trained with ``scores=True``, the lines of a single direction end with ``||| score``.

        Args:
            path: Output file.
            format: ``"pharaoh"`` (the only format).
            symmetrize, direction, pieces: As for :meth:`alignments`.
        """
        if format != "pharaoh":
            raise ValueError(f"format must be 'pharaoh', not {format!r}")
        self._inner.write_training_alignments(
            os.fspath(path), _method(symmetrize), direction, bool(pieces)
        )

    def align(
        self,
        src: Sentence,
        tgt: Sentence,
        symmetrize: str | None = None,
        direction: str | None = None,
    ) -> list[Link]:
        """Align one sentence pair (forced alignment) and return its links.

        Args:
            src, tgt: The sentences, as lists of tokens or as strings split on whitespace.
            symmetrize, direction: As for :meth:`alignments`.

        Words that were not in the training data, and word pairs the table has no entry for,
        get fast_align's probability of 1e-9 (as in its ``-f`` mode). Each call looks up the
        model's vocabulary afresh; to align many sentences use :meth:`align_many`.

        A model trained with a ``subword`` tokeniser cuts the words into pieces, aligns the
        pieces and returns links between *words* (positions in ``src`` and ``tgt``).
        """
        return self.align_many(
            [sentence_to_line(src, "src")],
            [sentence_to_line(tgt, "tgt")],
            symmetrize,
            direction,
            0,
            False,
        )[0]

    def align_many(
        self,
        src: TextSource,
        tgt: TextSource | None = None,
        symmetrize: str | None = None,
        direction: str | None = None,
        threads: int = 0,
        scores: bool = False,
    ) -> Alignments:
        """Align many sentence pairs in parallel (forced alignment).

        Args:
            src, tgt: Line-aligned text files, or lists of sentences (strings or token lists)
                of equal length. With ``tgt`` omitted, ``src`` is one file of
                ``source ||| target`` lines.
            symmetrize, direction: As for :meth:`alignments`.
            threads: Worker threads, 0 for all cores.
            scores: Append ``||| score`` (log probability of the best alignment) to every line
                of the written text; needs ``direction``. The links you read from the result
                are the same.

        With ``scores=True`` and ``result.write(path)`` the output equals that of
        ``fastalign align --scores``. The GIL is released while aligning and Ctrl-C interrupts
        the call. Lines with an empty side give empty alignments.

        With a model trained with a ``subword`` tokeniser the text is plain words: it is cut
        into pieces with the stored tokeniser (needs ``sentencepiece``) and the links are
        between words. A model trained on pre-segmented text (:class:`PreSegmented`) expects
        text segmented in the same way.
        """
        if self._text_segmenters() is not None:
            segmented = self._segment(src, tgt)
            return self._inner.align_many(
                segmented[0],
                segmented[1],
                _method(symmetrize),
                direction,
                _check_threads(threads),
                bool(scores),
            )
        return self._inner.align_many(
            text_source(src, "src"),
            None if tgt is None else text_source(tgt, "tgt"),
            _method(symmetrize),
            direction,
            _check_threads(threads),
            bool(scores),
        )


def _opt_float(name: str, v: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def train(
    src: TextSource,
    tgt: TextSource | None = None,
    threads: int = 0,
    *,
    directions: str = "both",
    iterations: int = 5,
    diagonal: bool = True,
    optimize_tension: bool = True,
    variational_bayes: bool = True,
    alpha: float = 0.01,
    tension: float = 4.0,
    p0: float = 0.08,
    null: bool = True,
    buffer_size: int = 10000,
    max_len: int | None = None,
    scores: bool = False,
    callback: Callable[[Progress], None] | None = None,
    lookup_index: bool = True,
    subword: str | SubwordConfig | PreSegmented | None = None,
    subword_output: bool = False,
) -> Model:
    """Train fast_align's reparameterised IBM Model 2 on a parallel corpus.

    The defaults are fast_align's recommended ``-d -o -v`` with five iterations. For every
    setting the alignments are byte-identical to ``OMP_NUM_THREADS=1 fast_align`` with the same
    options, whatever ``threads``.

    Args:
        src, tgt: The two sides of the corpus: paths of line-aligned text files, or lists of
            sentences (each a string split on whitespace, or a list of tokens) of equal length.
            With ``tgt`` omitted, ``src`` is the path of one file of ``source ||| target``
            lines (fast_align's format). A plain ``str`` is always a file path.
        threads: Worker threads; 0 uses all cores.
        directions: ``"both"``, ``"fwd"`` (source generates target) or ``"inv"`` (fast_align's
            ``-r``). Symmetrising needs both.
        iterations: EM iterations (fast_align ``-I``); the last one only decodes.
        diagonal: Favour the diagonal (``-d``); off gives a uniform alignment prior.
        optimize_tension: Re-estimate the diagonal tension each iteration (``-o``).
        variational_bayes: Dirichlet-prior variational Bayes (``-v``) instead of plain EM.
        alpha: Dirichlet prior parameter of variational Bayes (``-a``).
        tension: Starting tension of the diagonal prior (``-T``); needs ``diagonal``.
        p0: Probability of the NULL word with the diagonal prior (``-q``), in [0, 1).
        null: Model a NULL word (``False`` is fast_align's ``-N``).
        buffer_size: Lines per statistics buffer (``-b``). It changes only how the sums are
            grouped, exactly as in fast_align.
        max_len: Skip lines with more tokens than this on either side. Off by default:
            dropping lines changes the result relative to fast_align on the unfiltered file.
        scores: Append ``||| score`` (log probability of the best alignment, ``-s``) to the
            lines of each direction's alignments.
        callback: Called with a :class:`Progress` after every iteration of each direction
            (from a worker thread; the directions train one after the other). An exception raised by
            it stops training and propagates. Progress is also logged to the ``fastalign``
            logger at INFO level.
        lookup_index: Build a dense lookup index of the busiest table rows (64 MB per
            direction) for a faster E-step and M-step. ``False`` saves the memory; output is
            the same.
        subword: Align subword pieces and report word links. ``None`` (default): words, nothing
            changes. ``"auto"`` or a :class:`SubwordConfig`: train a SentencePiece tokeniser on
            the corpus (needs ``pip install 'fastalign[subword]'``; the defaults are BPE, one
            joint model of 8000 pieces), cut every word into pieces, train on the pieces, and
            project the piece links to words; the tokeniser is stored in the model, so
            :meth:`Model.align_many` segments new text the same way. A :class:`PreSegmented`:
            the corpus is already segmented, only the word boundaries are needed. Output is
            always word-level. ``max_len`` then counts pieces.
        subword_output: With ``subword``: keep the piece-level alignments too
            (``Model.alignments(direction=..., pieces=True)``).

    Returns:
        The trained :class:`Model`, which also holds the alignments of the training corpus.

    Raises:
        ValueError: bad arguments (mismatched corpus sizes, a token with whitespace, an empty
            corpus, ...).
        FileNotFoundError, OSError: unreadable input files.
        KeyboardInterrupt: Ctrl-C. Training stops within a fraction of a second.

    The GIL is released for the whole run, so other Python threads keep running.
    """
    if directions not in ("both", "fwd", "inv"):
        raise ValueError(f"directions must be 'both', 'fwd' or 'inv', not {directions!r}")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError(f"iterations must be a non-negative integer, got {iterations!r}")
    if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size < 0:
        raise ValueError(f"buffer_size must be a non-negative integer, got {buffer_size!r}")
    if max_len is not None and (
        isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 1
    ):
        raise ValueError(f"max_len must be a positive integer or None, got {max_len!r}")
    alpha, tension, p0 = (
        _opt_float("alpha", alpha),
        _opt_float("tension", tension),
        _opt_float("p0", p0),
    )
    if variational_bayes and alpha <= 0:
        raise ValueError("alpha must be > 0 for variational Bayes")
    if not diagonal and tension != 4.0:
        raise ValueError("tension needs diagonal=True")
    if not 0.0 <= p0 < 1.0:
        raise ValueError(f"p0 must be in [0, 1), got {p0!r}")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")

    def on_progress(
        direction: str,
        iteration: int,
        of: int,
        log_likelihood: float,
        perplexity: float,
        p0_: float,
        tension_: float,
        seconds: float,
    ) -> None:
        event = Progress(
            direction, iteration, of, log_likelihood, perplexity, p0_, tension_, seconds
        )
        logger.info("%s", event)
        if callback is not None:
            callback(event)

    sub = _sw.resolve_config(subword)
    if sub is None and subword_output:
        raise ValueError("subword_output needs subword")

    def run(src_arg, tgt_arg, subword_arg):
        return _core.train(
            src_arg,
            tgt_arg,
            _check_threads(threads),
            directions,
            iterations,
            bool(diagonal),
            bool(optimize_tension),
            bool(variational_bayes),
            alpha,
            tension,
            p0,
            bool(null),
            buffer_size,
            max_len,
            bool(scores),
            on_progress,
            bool(lookup_index),
            subword_arg,
            bool(subword_output),
        )

    if sub is None:
        inner = run(text_source(src, "src"), None if tgt is None else text_source(tgt, "tgt"), None)
    elif isinstance(sub, PreSegmented):
        inner = run(
            text_source(src, "src"),
            None if tgt is None else text_source(tgt, "tgt"),
            (sub.kind, sub.marker.encode("utf-8"), [], b""),
        )
    else:
        inner = _train_with_tokenizer(run, src, tgt, sub)
    return Model(inner)


def _train_with_tokenizer(run, src, tgt, config: SubwordConfig):
    """Train the tokeniser on the corpus (streamed), write the piece corpus to a temporary file,
    train on it and attach the tokeniser to the model."""
    import tempfile

    sides = _sw.corpus_sides(src, tgt)
    sa, sb, blobs, layout = _sw.train_stream_segmenters(sides, config)
    with tempfile.TemporaryDirectory(prefix="fastalign-subword-") as tmp:
        path = os.path.join(tmp, "pieces.txt")
        _sw.write_piece_corpus(path, sides, (sa, sb))
        return run(
            path,
            None,
            ("start", _sw.MARKER.encode("utf-8"), blobs, _sw.settings_json(config, layout)),
        )


def align_with_table(
    table: str | os.PathLike[str],
    src: TextSource,
    tgt: TextSource | None = None,
    *,
    direction: str = "fwd",
    diagonal: bool = True,
    null: bool = True,
    p0: float = 0.08,
    tension: float = 4.0,
    mean_length_multiplier: float = 1.0,
    threads: int = 0,
) -> TableAlignment:
    """Forced alignment with a table written by fast_align's ``-p`` (or
    :meth:`Model.write_table`): fast_align's ``-f`` mode, reproduced byte for byte.

    Args:
        table: The table file.
        src, tgt: As for :func:`train`.
        direction: ``"inv"`` is fast_align's ``-r`` (the table conditions on the target side).
        diagonal, null, p0, tension: As in training; ``tension`` is what ``-T`` would give.
        mean_length_multiplier: fast_align's ``-m``: expected target length per source word,
            which enters the scores only.
        threads: Worker threads for reading the corpus.

    A saved :class:`Model` needs none of these settings and is exact; use this for tables
    produced by fast_align itself.
    """
    if not diagonal and tension != 4.0:
        raise ValueError("tension needs diagonal=True")
    out, log, failed = _core.align_table(
        os.fspath(table),
        text_source(src, "src"),
        None if tgt is None else text_source(tgt, "tgt"),
        _check_direction(direction) == "inv",
        bool(diagonal),
        bool(null),
        _opt_float("p0", p0),
        _opt_float("tension", tension),
        _opt_float("mean_length_multiplier", mean_length_multiplier),
        _check_threads(threads),
    )
    return TableAlignment(out, log.decode("utf-8", errors="replace"), failed)


# -- symmetrisation -------------------------------------------------------------------------


def symmetrize(
    fwd: Sequence[Link],
    inv: Sequence[Link],
    method: str = "grow-diag-final-and",
) -> list[Link]:
    """Symmetrise the two directional alignments of one sentence pair.

    Both ``fwd`` and ``inv`` are lists of ``(source, target)`` links. The result is sorted and
    identical to what ``atools -c METHOD`` gives for the two lines (atools needs no sentence
    lengths).
    """
    return symmetrize_many([fwd], [inv], method)[0]


def symmetrize_many(
    fwd: Sequence[Sequence[Link]],
    inv: Sequence[Sequence[Link]],
    method: str = "grow-diag-final-and",
) -> list[list[Link]]:
    """Symmetrise many sentence pairs in parallel; see :func:`symmetrize`."""
    if len(fwd) != len(inv):
        raise ValueError(f"fwd and inv must have the same size ({len(fwd)} and {len(inv)})")
    return _core.symmetrize_many(
        [list(map(tuple, f)) for f in fwd],
        [list(map(tuple, i)) for i in inv],
        _method(method),
    )


def invert(links: Sequence[Link]) -> list[Link]:
    """Swap the two sides of an alignment (``atools -c invert``); the result is sorted."""
    return _core.invert_many([list(map(tuple, links))])[0]


def parse_pharaoh(line: str) -> list[Link]:
    """Parse a Pharaoh alignment line such as ``"0-0 1-2"`` into links (a ``||| score``
    suffix is ignored)."""
    return _core.parse_pharaoh(line)
