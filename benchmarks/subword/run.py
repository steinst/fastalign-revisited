#!/usr/bin/env python3
"""Subword alignment harness: tokenise, align, project to words, symmetrise, score.

    run.py CORPUS [--type unigram|bpe|word] [--vocab N] [--joint|--separate]
                  [--segment both|src|tgt] [--order proj-sym|sym-proj] [--threads N]

CORPUS is an id such as ``enfr-tiny`` or ``roen-medium`` (files under
``reference/data/corpora``; the gold sentences are the last lines of the corpus).
``--type word`` is the word-level baseline (no tokeniser). One row per system (fwd, rev and
every symmetrisation method) is appended to ``reference/runs/subword/results.tsv``; the encoded
corpus and alignments stay in ``reference/runs/subword/<run id>/``.

Times: ``t_tok`` tokeniser training, ``t_encode`` encoding + writing the piece corpus,
``t_align`` the aligner (``fastalign train``, both directions), ``t_project`` parsing,
projection and symmetrisation. ``rss_align_mb`` is the aligner's peak RSS (via ``/usr/bin/time``), ``rss_self_mb`` this
process's (tokeniser and maps).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "benchmarks"))

import aer
import fastalign
from fastalign import subword as sw

CORPORA = REPO / "reference" / "data" / "corpora"
OUT = REPO / "reference" / "runs" / "subword"
NATIVE = REPO / "target" / "release" / "fastalign"
COLUMNS = [
    "run", "corpus", "type", "vocab", "joint", "segment", "order", "threads", "system",
    "precision", "recall", "aer", "t_tok", "t_encode", "t_align", "t_project",
    "rss_align_mb", "rss_self_mb", "src_pieces_per_word", "tgt_pieces_per_word",
    "src_words_split", "tgt_words_split", "src_word_types", "tgt_word_types",
    "src_piece_types", "tgt_piece_types", "joint_piece_types",
]  # fmt: skip


def stats(enc: sw.Encoded) -> tuple[float, int]:
    """Pieces per word and number of distinct pieces."""
    n_words = sum(enc.n_words)
    n_pieces = sum(len(p) for p in enc.pieces)
    return n_pieces / max(n_words, 1), len({t for line in enc.pieces for t in line})


def word_types(lines) -> int:
    return len({w for line in lines for w in line})


def read_pharaoh(path: Path) -> list[list[tuple[int, int]]]:
    with open(path, encoding="utf-8") as f:
        return [fastalign.parse_pharaoh(line) for line in f]


def score(systems: dict[str, list], gold, offset: int) -> dict[str, dict[str, float]]:
    sure, poss = gold
    out = {}
    for name, links in systems.items():
        hyp = [set(l) for l in links[offset : offset + len(sure)]]
        out[name] = aer.score(hyp, sure, poss)
    return out


def run(args) -> list[dict]:
    pair, size = args.corpus.split("-")
    meta = json.loads((CORPORA / pair / "meta.json").read_text())
    nt = meta["test_sentences"]
    src_lines = sw.read_lines(str(CORPORA / pair / f"{size}.src"))
    tgt_lines = sw.read_lines(str(CORPORA / pair / f"{size}.tgt"))
    n = len(src_lines)
    with open(CORPORA / pair / "gold.txt", encoding="utf-8") as f:
        gold = aer.read_gold(f)

    joint = "joint" if args.joint else "separate"
    run_id = f"{args.corpus}-{args.type}{args.vocab}-{joint}-{args.segment}-{args.order}"
    if args.type == "word":
        run_id = f"{args.corpus}-word"
    rundir = OUT / run_id
    rundir.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    if args.type == "word":
        segs: tuple[sw.Segmenter, sw.Segmenter] = (sw.WholeWords(), sw.WholeWords())
    else:
        cfg = sw.SubwordConfig(
            model_type=args.type, vocab_size=args.vocab, joint=args.joint, segment=args.segment
        )
        segs = sw.train_segmenters(src_lines, tgt_lines, cfg)
    t_tok = time.perf_counter() - t

    t = time.perf_counter()
    src, tgt = sw.encode(src_lines, segs[0]), sw.encode(tgt_lines, segs[1])
    corpus_file = rundir / "pieces.txt"
    sw.write_pair_file(str(corpus_file), src, tgt)
    t_encode = time.perf_counter() - t

    t = time.perf_counter()
    prefix = rundir / "align"
    # Under /usr/bin/time: a child forked from this process inherits this process's RSS into
    # its ru_maxrss across exec, so RUSAGE_CHILDREN would overstate the aligner's peak.
    stat = rundir / "align.time"
    subprocess.run(
        ["/usr/bin/time", "-f", "%M", "-o", str(stat), str(NATIVE), "train", str(corpus_file)]
        + ["-o", str(prefix), "-t", str(args.threads), "-q"],
        check=True,
    )
    t_align = time.perf_counter() - t
    rss_align = int(stat.read_text().split()[-1])  # kB

    t = time.perf_counter()
    fwd = read_pharaoh(prefix.with_suffix(".fwd.pharaoh"))
    inv = read_pharaoh(prefix.with_suffix(".inv.pharaoh"))
    systems: dict[str, list] = {}
    if args.order == "proj-sym":
        pf = sw.project_many(fwd, src, tgt, by="tgt")
        pi = sw.project_many(inv, src, tgt, by="src")
        systems["fwd"], systems["rev"] = pf, pi
        for m in fastalign.METHODS:
            systems[m] = fastalign.symmetrize_many(pf, pi, m)
    else:
        systems["fwd"] = sw.project_many(fwd, src, tgt, by="tgt")
        systems["rev"] = sw.project_many(inv, src, tgt, by="src")
        for m in fastalign.METHODS:
            systems[m] = sw.project_many(fastalign.symmetrize_many(fwd, inv, m), src, tgt, "src")
    t_project = time.perf_counter() - t

    for name, links in systems.items():
        with open(rundir / f"{name}.words.pharaoh", "w") as f:
            f.writelines(" ".join(f"{i}-{j}" for i, j in l) + "\n" for l in links)

    results = score(systems, gold, n - nt)
    sp, sv = stats(src)
    tp, tv = stats(tgt)
    joint_types = len({p for e in (src, tgt) for line in e.pieces for p in line})
    rows = []
    for name, r in results.items():
        rows.append(
            dict(  # noqa: C408
                run=run_id,
                corpus=args.corpus,
                type=args.type,
                vocab=args.vocab if args.type != "word" else 0,
                joint=joint,
                segment=args.segment,
                order=args.order,
                threads=args.threads,
                system=name,
                precision=f"{100 * r['precision']:.2f}",
                recall=f"{100 * r['recall']:.2f}",
                aer=f"{100 * r['aer']:.2f}",
                t_tok=f"{t_tok:.2f}",
                t_encode=f"{t_encode:.2f}",
                t_align=f"{t_align:.2f}",
                t_project=f"{t_project:.2f}",
                rss_align_mb=f"{rss_align / 1024:.0f}",
                rss_self_mb=f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}",
                src_pieces_per_word=f"{sp:.3f}",
                tgt_pieces_per_word=f"{tp:.3f}",
                src_words_split=f"{split_fraction(src):.4f}",
                tgt_words_split=f"{split_fraction(tgt):.4f}",
                src_word_types=word_types(src_lines),
                tgt_word_types=word_types(tgt_lines),
                src_piece_types=sv,
                tgt_piece_types=tv,
                joint_piece_types=joint_types,
            )
        )
    return rows


def split_fraction(enc: sw.Encoded) -> float:
    """Fraction of word tokens that were cut into more than one piece."""
    words = split = 0
    for m in enc.word_of:
        words_in_line = 0
        prev = -1
        run_len = 0
        for w in m:
            if w != prev:
                if run_len > 1:
                    split += 1
                words_in_line += 1
                prev, run_len = w, 0
            run_len += 1
        if run_len > 1:
            split += 1
        words += words_in_line
    return split / max(words, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("corpus", help="e.g. enfr-tiny, roen-medium")
    ap.add_argument("--type", choices=["unigram", "bpe", "word"], default="bpe")
    ap.add_argument("--vocab", type=int, default=8000)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--joint", dest="joint", action="store_true", default=True)
    g.add_argument("--separate", dest="joint", action="store_false")
    ap.add_argument("--segment", choices=["both", "src", "tgt"], default="both")
    ap.add_argument("--order", choices=["proj-sym", "sym-proj"], default="proj-sym")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tsv", default=str(OUT / "results.tsv"))
    args = ap.parse_args(argv)
    if not NATIVE.exists():
        print(f"error: build the native binary first ({NATIVE})", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run(args)
    new = not os.path.exists(args.tsv)
    with open(args.tsv, "a") as f:
        if new:
            f.write("\t".join(COLUMNS) + "\n")
        f.writelines("\t".join(str(r[c]) for c in COLUMNS) + "\n" for r in rows)
    for r in rows:
        print(
            f"{r['run']:50s} {r['system']:20s} P {r['precision']:>6} R {r['recall']:>6} AER {r['aer']:>6}"
        )
    r = rows[0]
    print(
        f"tok {r['t_tok']}s encode {r['t_encode']}s align {r['t_align']}s project {r['t_project']}s; "
        f"rss align {r['rss_align_mb']} MB; pieces/word src {r['src_pieces_per_word']} "
        f"tgt {r['tgt_pieces_per_word']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
