"""Command-line interface: ``fastalign train | align | symmetrize | export-table``.

Same subcommands and options as the native ``fastalign`` binary built from fastalign-core.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import fastalign
from fastalign import _core
from fastalign.model import METHODS, _method

#: ``intersection`` is accepted for ``intersect`` (the sister tool's name).
_METHOD_CHOICES = (*METHODS, "intersection")


def _add_threads(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="worker threads, 0 = all cores (the output never depends on it)",
    )


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--no-diagonal", action="store_true", help="do not favour the diagonal (fast_align -d off)"
    )
    p.add_argument(
        "-T",
        "--tension",
        type=float,
        default=4.0,
        help="starting tension of the diagonal prior (default %(default)s)",
    )
    p.add_argument(
        "--p0",
        type=float,
        default=0.08,
        help="probability of the NULL word with the diagonal prior (fast_align -q; "
        "default %(default)s)",
    )
    p.add_argument("-N", "--no-null", action="store_true", help="no NULL word")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastalign",
        description="Word alignment with fast_align's reparameterised IBM Model 2, "
        "byte-identical to fast_align.",
    )
    parser.add_argument("-V", "--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    t = sub.add_parser("train", help="train on a parallel corpus and write alignments")
    t.add_argument(
        "input",
        help="corpus: one file of `source ||| target` lines, or the source text (then give TGT)",
    )
    t.add_argument("tgt", nargs="?", help="target text, line-aligned with INPUT")
    t.add_argument(
        "-o",
        "--output",
        help="output prefix: writes PREFIX.fwd.pharaoh, PREFIX.inv.pharaoh and "
        "PREFIX.<method>.pharaoh for every --symmetrize; without it the single direction "
        "chosen by --directions goes to standard output",
    )
    _add_threads(t)
    t.add_argument("--directions", choices=["both", "fwd", "inv"], default="both")
    t.add_argument("-I", "--iterations", type=int, default=5, help="iterations (default 5)")
    _add_model_args(t)
    t.add_argument("--no-optimize-tension", action="store_true", help="do not optimise the tension")
    t.add_argument(
        "--no-variational-bayes", action="store_true", help="plain EM instead of variational Bayes"
    )
    t.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.01,
        help="Dirichlet prior of variational Bayes (default %(default)s)",
    )
    t.add_argument(
        "-b",
        "--buffer-size",
        type=int,
        default=10000,
        help="lines per statistics buffer (default %(default)s); changes only how sums are "
        "grouped, as in fast_align",
    )
    t.add_argument(
        "--max-len",
        type=int,
        help="skip lines with more tokens than this on either side (default: keep all; this "
        "changes the result relative to fast_align on the unfiltered file). Skipped lines give "
        "empty alignment lines",
    )
    t.add_argument(
        "-s",
        "--symmetrize",
        action="append",
        default=[],
        choices=_METHOD_CHOICES,
        metavar="METHOD",
        help="symmetrisation to write (repeatable): " + ", ".join(METHODS),
    )
    t.add_argument(
        "--scores",
        action="store_true",
        help="append ` ||| score` (log probability of the best alignment) to directional lines",
    )
    t.add_argument(
        "--no-lookup-index",
        action="store_true",
        help="do not build the dense lookup index of the busiest table rows (saves 64 MB per "
        "direction, slower E-step and M-step; no output changes)",
    )
    sw = t.add_argument_group(
        "subword alignment",
        "Align pieces, write word alignments. --subword trains SentencePiece (needs "
        "`pip install 'fastalign[subword]'`) and stores it in the --save model; the native "
        "binary has only --subword-marker/--subword-continuation for text that is already "
        "segmented.",
    )
    sw.add_argument(
        "--subword",
        action="store_true",
        help="segment the corpus with a SentencePiece model trained on it (BPE, one joint "
        "model of 8000 pieces unless changed below)",
    )
    sw.add_argument("--subword-vocab", type=int, default=None, help="pieces per model (8000)")
    sw.add_argument("--subword-type", choices=["bpe", "unigram"], default=None)
    sw.add_argument(
        "--subword-separate",
        action="store_true",
        help="one tokeniser per side instead of one joint model",
    )
    sw.add_argument(
        "--subword-segment",
        choices=["both", "src", "tgt"],
        default=None,
        help="which side is segmented; the other stays word level (default both)",
    )
    sw.add_argument(
        "--subword-marker",
        metavar="MARKER",
        help="the input is already segmented; a piece starting with MARKER (SentencePiece: ▁) "
        "begins a word",
    )
    sw.add_argument(
        "--subword-continuation",
        metavar="MARKER",
        help="as --subword-marker for subword-nmt text: a piece ending with MARKER (@@) "
        "continues into the next piece",
    )
    sw.add_argument(
        "--subword-output",
        action="store_true",
        help="also write the piece-level alignments of each direction "
        "(PREFIX.<dir>.subword.pharaoh); needs --output",
    )
    t.add_argument("--save", help="save the trained model here")
    t.add_argument(
        "--table",
        action="store_true",
        help="also write fast_align's -p tables (PREFIX.<dir>.table); needs --output",
    )
    t.add_argument(
        "--beam-threshold",
        type=float,
        default=-4.0,
        help="beam threshold of the tables (fast_align -t; default %(default)s)",
    )
    t.add_argument(
        "--log",
        action="store_true",
        help="also write fast_align's statistics (PREFIX.<dir>.log); needs --output",
    )
    t.add_argument("-q", "--quiet", action="store_true", help="no progress output")

    a = sub.add_parser(
        "align",
        help="align new sentence pairs with a saved model, or with a fast_align -p table",
    )
    a.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="MODEL SRC [TGT], or with --table: SRC [TGT]; SRC alone is a `source ||| target` file",
    )
    a.add_argument("-o", "--output", help="output file (default: standard output)")
    a.add_argument(
        "-s",
        "--symmetrize",
        choices=_METHOD_CHOICES,
        metavar="METHOD",
        help="symmetrisation (needs a model of both directions); otherwise one direction",
    )
    a.add_argument(
        "--direction",
        choices=["fwd", "inv"],
        default="fwd",
        help="direction written without --symmetrize (with --table: which one the table is "
        "for); default %(default)s",
    )
    a.add_argument("--scores", action="store_true", help="append ` ||| score` to each line")
    _add_threads(a)
    a.add_argument(
        "--table",
        help="align with a table written by fast_align's -p (or `export-table`) instead of a "
        "model, and print fast_align's -f format (`src ||| tgt ||| links ||| score`)",
    )
    _add_model_args(a)
    a.add_argument(
        "-m",
        "--mean-length-multiplier",
        type=float,
        default=1.0,
        help="mean target length per source word for the score (fast_align -m; --table only)",
    )

    s = sub.add_parser(
        "symmetrize", help="symmetrise two directional alignment files (as atools -c METHOD)"
    )
    s.add_argument("fwd", help="forward alignments (Pharaoh), one line per sentence pair")
    s.add_argument(
        "inv", nargs="?", help="inverse alignments (source-target orientation); not for invert"
    )
    s.add_argument(
        "-m", "--method", choices=[*_METHOD_CHOICES, "invert"], default="grow-diag-final-and"
    )
    s.add_argument("-o", "--output", help="output file (default: standard output)")

    e = sub.add_parser(
        "export-table", help="write a saved model's table in fast_align's -p text format"
    )
    e.add_argument("model")
    e.add_argument("-o", "--output", help="output file (default: standard output)")
    e.add_argument("--direction", choices=["fwd", "inv"], default="fwd")
    e.add_argument("--beam-threshold", type=float, default=-4.0)
    _add_threads(e)
    return parser


def _write_bytes(path: str | None, data: bytes) -> None:
    if path:
        with open(path, "wb") as f:
            f.write(data)
    else:
        sys.stdout.flush()
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


def _read_bytes(path: str) -> bytes:
    if path == "-":
        return sys.stdin.buffer.read()
    with open(path, "rb") as f:
        return f.read()


def _subword_arg(args: argparse.Namespace):
    """The ``subword=`` argument of :func:`fastalign.train` for the command line options."""
    given = [
        o
        for o in ("subword_vocab", "subword_type", "subword_segment")
        if getattr(args, o) is not None
    ] + (["subword_separate"] if args.subword_separate else [])
    if args.subword_marker is not None and args.subword_continuation is not None:
        raise ValueError("--subword-marker and --subword-continuation exclude each other")
    pre = args.subword_marker is not None or args.subword_continuation is not None
    if pre:
        if args.subword or given:
            raise ValueError(
                "--subword-marker/--subword-continuation (input already segmented) cannot be "
                "combined with the options that train a tokeniser"
            )
        return fastalign.PreSegmented(
            marker=args.subword_marker or args.subword_continuation,
            continuation=args.subword_continuation is not None,
        )
    if not args.subword:
        if given:
            raise ValueError("--subword-vocab, -type, -separate and -segment need --subword")
        return None
    kwargs = {}
    if args.subword_vocab is not None:
        kwargs["vocab_size"] = args.subword_vocab
    if args.subword_type is not None:
        kwargs["model_type"] = args.subword_type
    if args.subword_segment is not None:
        kwargs["segment"] = args.subword_segment
    if args.subword_separate:
        kwargs["joint"] = False
    return fastalign.SubwordConfig(**kwargs)


def _cmd_train(args: argparse.Namespace) -> None:
    both = args.directions == "both"
    if args.symmetrize:
        if not both:
            raise ValueError("--symmetrize needs --directions both")
        if not args.output:
            raise ValueError("--symmetrize needs --output PREFIX")
    if not args.output and both:
        raise ValueError("both directions need --output PREFIX (or choose --directions fwd|inv)")
    if (args.table or args.log) and not args.output:
        raise ValueError("--table and --log need --output PREFIX")
    subword = _subword_arg(args)
    if args.subword_output and subword is None:
        raise ValueError(
            "--subword-output needs --subword, --subword-marker or --subword-continuation"
        )
    if args.subword_output and not args.output:
        raise ValueError("--subword-output needs --output PREFIX")
    t0 = time.perf_counter()
    model = fastalign.train(
        args.input,
        args.tgt,
        threads=args.threads,
        directions=args.directions,
        iterations=args.iterations,
        diagonal=not args.no_diagonal,
        optimize_tension=not args.no_optimize_tension,
        variational_bayes=not args.no_variational_bayes,
        alpha=args.alpha,
        tension=args.tension,
        p0=args.p0,
        null=not args.no_null,
        buffer_size=args.buffer_size,
        max_len=args.max_len,
        scores=args.scores,
        lookup_index=not args.no_lookup_index,
        subword=subword,
        subword_output=args.subword_output,
    )
    prefix = args.output
    for d in model.directions:
        if prefix is None:
            _write_bytes(None, model.alignments(direction=d).to_bytes())
            continue
        model.write_alignments(f"{prefix}.{d}.pharaoh", direction=d)
        if args.subword_output:
            model.write_alignments(f"{prefix}.{d}.subword.pharaoh", direction=d, pieces=True)
        if args.log:
            log = model._inner.training_log(d)
            if args.table:
                log += f"conditional probabilities: {prefix}.{d}.table\n".encode()
            _write_bytes(f"{prefix}.{d}.log", log)
        if args.table:
            model.write_table(
                f"{prefix}.{d}.table", d, beam_threshold=args.beam_threshold, threads=args.threads
            )
    for m in args.symmetrize:
        model.write_alignments(f"{prefix}.{_method(m)}.pharaoh", symmetrize=m)
    if args.save:
        model.save(args.save)
    logging.getLogger("fastalign").info("done in %.1f s", time.perf_counter() - t0)


def _cmd_align(args: argparse.Namespace) -> int:
    files = args.files
    if args.table:
        if len(files) not in (1, 2):
            raise ValueError("usage: fastalign align --table TABLE SRC [TGT]")
        if args.symmetrize:
            raise ValueError("--symmetrize is not available with --table")
        res = fastalign.align_with_table(
            args.table,
            files[0],
            files[1] if len(files) == 2 else None,
            direction=args.direction,
            diagonal=not args.no_diagonal,
            null=not args.no_null,
            p0=args.p0,
            tension=args.tension,
            mean_length_multiplier=args.mean_length_multiplier,
            threads=args.threads,
        )
        _write_bytes(args.output, res.output)
        sys.stderr.write(res.log)
        return 1 if res.failed_line is not None else 0
    if len(files) not in (2, 3):
        raise ValueError("usage: fastalign align MODEL SRC [TGT]")
    if args.symmetrize and args.scores:
        raise ValueError("--scores belongs to one direction; drop --symmetrize")
    model = fastalign.Model.load(files[0])
    result = model.align_many(
        files[1],
        files[2] if len(files) == 3 else None,
        symmetrize=args.symmetrize,
        direction=None if args.symmetrize else args.direction,
        threads=args.threads,
        scores=args.scores,
    )
    _write_bytes(args.output, result.to_bytes())
    return 0


def _cmd_symmetrize(args: argparse.Namespace) -> None:
    method = _method(args.method)
    if method != "invert" and args.inv is None:
        raise ValueError(f"{method} needs two alignment files")
    fwd = _read_bytes(args.fwd)
    inv = _read_bytes(args.inv) if (args.inv is not None and method != "invert") else None
    _write_bytes(args.output, _core.symmetrize_text(fwd, inv, method))


def _cmd_export(args: argparse.Namespace) -> None:
    model = fastalign.Model.load(args.model)
    if args.output:
        model.write_table(args.output, args.direction, args.beam_threshold, args.threads)
        return
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "table")
        model.write_table(path, args.direction, args.beam_threshold, args.threads)
        _write_bytes(None, _read_bytes(path))


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.version:
        print(f"fastalign {fastalign.__version__}")
        return 0
    if args.command is None:
        parser.print_usage(sys.stderr)
        print(
            "fastalign: error: a command is required (train, align, symmetrize, export-table)",
            file=sys.stderr,
        )
        return 2
    quiet = getattr(args, "quiet", False)
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    handlers = {
        "train": _cmd_train,
        "align": _cmd_align,
        "symmetrize": _cmd_symmetrize,
        "export-table": _cmd_export,
    }
    try:
        return handlers[args.command](args) or 0
    except KeyboardInterrupt:
        print("fastalign: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # Output closed early (`fastalign align ... | head`): not an error worth a message.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    except (ValueError, TypeError, OSError) as e:
        print(f"fastalign: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
