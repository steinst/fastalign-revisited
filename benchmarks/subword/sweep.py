#!/usr/bin/env python3
"""Subword sweep (Step 13): many tokeniser settings on one corpus, both projection orders.

    sweep.py CORPUS [--types unigram,bpe] [--vocabs 1000,2000,...] [--modes joint,separate,src,tgt]
             [--threads 4] [--jobs 3] [--word] [--keep] [--tsv PATH]

For every (type, vocab, mode) it trains (or reuses) the SentencePiece model(s), encodes the
corpus per word, runs ``fastalign train`` (both directions, ``-d -o -v``), projects the piece
links to words in **both** orders (project-then-symmetrise and symmetrise-then-project, from the
same alignment run) and scores every system against the gold set. ``--word`` adds the word-level
baseline. Modes: ``joint`` = one model over both sides, ``separate`` = one model per side,
``src``/``tgt`` = only that side segmented (its own model), the other side stays word level.

Outputs under ``reference/runs/subword/sweep/``:

* ``sweep.tsv``: one row per (run, order, system) with P/R/AER, segmentation statistics and cost
  (wall and CPU seconds of each stage, the aligner's own peak RSS from ``wait4``);
* ``<run>/links.json``: the word-level links of the gold sentences for every system, which
  ``bootstrap.py`` resamples;
* ``models/``: the tokeniser models, reused by later runs with the same settings.

The piece corpus and full alignment files are deleted after scoring unless ``--keep``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "python")]
import aer
import fastalign
import run as harness
from fastalign import subword as sw

OUT = harness.OUT / "sweep"
MODELS = OUT / "models"
COLUMNS = [
    "run", "corpus", "lines", "type", "vocab", "mode", "order", "system",
    "precision", "recall", "aer",
    "t_tok", "t_encode", "t_align", "cpu_align", "t_project", "rss_align_mb",
    "src_ppw", "tgt_ppw", "src_split", "tgt_split", "src_word_types", "tgt_word_types",
    "src_piece_types", "tgt_piece_types", "src_model_size", "tgt_model_size",
    "src_tokens", "tgt_tokens", "pretok",
]  # fmt: skip

_CACHE: dict = {}


def load(corpus: str):
    """Lines and gold of a corpus id (cached per worker process)."""
    if corpus not in _CACHE:
        pair, size = corpus.split("-")
        d = harness.CORPORA / pair
        meta = json.loads((d / "meta.json").read_text())
        src = sw.read_lines(str(d / f"{size}.src"))
        tgt = sw.read_lines(str(d / f"{size}.tgt"))
        with open(d / "gold.txt", encoding="utf-8") as f:
            gold = aer.read_gold(f)
        _CACHE[corpus] = (src, tgt, gold, meta["test_sentences"])
    return _CACHE[corpus]


PRETOK = {"": {}, "ns": {"split_by_script": False}, "nn": {"split_by_number": False}}
PRETOK["nsn"] = {**PRETOK["ns"], **PRETOK["nn"]}


def model(corpus: str, typ: str, vocab: int, which: str, lines, pretok: str = "") -> bytes:
    """SentencePiece model for ``which`` in joint|src|tgt, cached on disk. ``pretok``: ``""``
    (SentencePiece's default pre-tokenisation), ``ns`` (no split at script changes), ``nn`` (no
    split around digits), ``nsn`` (neither)."""
    MODELS.mkdir(parents=True, exist_ok=True)
    tag = f"-{pretok}" if pretok else ""
    path = MODELS / f"{corpus}-{typ}{vocab}{tag}-{which}.model"
    if path.exists():
        return path.read_bytes()
    proto = sw.train_model(lines, vocab, typ, byte_fallback=True, **PRETOK[pretok])
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(proto)
    tmp.rename(path)
    return proto


def segmenters(corpus, typ, vocab, mode, src, tgt, pretok=""):
    if typ == "word":
        return sw.WholeWords(), sw.WholeWords(), 0, 0
    if mode == "joint":
        m = model(corpus, typ, vocab, "joint", list(src) + list(tgt), pretok)
        s = sw.SentencePieceSegmenter(m)
        return s, s, len(s._sp), len(s._sp)
    a = b = None
    na = nb = 0
    if mode in ("separate", "src"):
        a = sw.SentencePieceSegmenter(model(corpus, typ, vocab, "src", src, pretok))
        na = len(a._sp)
    if mode in ("separate", "tgt"):
        b = sw.SentencePieceSegmenter(model(corpus, typ, vocab, "tgt", tgt, pretok))
        nb = len(b._sp)
    return a or sw.WholeWords(), b or sw.WholeWords(), na, nb


def align(corpus_file: Path, prefix: Path, threads: int) -> tuple[float, float, float]:
    """Run the aligner; returns (wall s, CPU s, peak RSS MB) of the aligner alone.

    The aligner runs under ``/usr/bin/time``: a child forked straight from this (large) Python
    process inherits the parent's RSS into its ``ru_maxrss`` across ``exec``, which would make
    the peak meaningless."""
    stat = prefix.with_suffix(".time")
    cmd = ["/usr/bin/time", "-f", "%e %U %S %M", "-o", str(stat), str(harness.NATIVE), "train"]
    cmd += [str(corpus_file), "-o", str(prefix), "-t", str(threads), "-q"]
    subprocess.run(cmd, check=True)
    wall, user, sys_, rss_kb = stat.read_text().split()[-4:]
    return float(wall), float(user) + float(sys_), int(rss_kb) / 1024


def one(job: dict) -> list[dict]:
    corpus, typ, vocab, mode = job["corpus"], job["type"], job["vocab"], job["mode"]
    src_lines, tgt_lines, gold, nt = load(corpus)
    n = len(src_lines)
    pretok = job.get("pretok", "")
    run_id = f"{corpus}-word" if typ == "word" else f"{corpus}-{typ}{vocab}-{mode}"
    if pretok and typ != "word":
        run_id += f"-{pretok}"
    rundir = OUT / run_id
    if (rundir / "done").exists() and not job["force"]:
        return []
    rundir.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    seg_s, seg_t, ms, mt = segmenters(corpus, typ, vocab, mode, src_lines, tgt_lines, pretok)
    t_tok = time.perf_counter() - t

    t = time.perf_counter()
    src, tgt = sw.encode(src_lines, seg_s), sw.encode(tgt_lines, seg_t)
    pieces = rundir / "pieces.txt"
    sw.write_pair_file(str(pieces), src, tgt)
    t_encode = time.perf_counter() - t

    prefix = rundir / "align"
    t_align, cpu_align, rss = align(pieces, prefix, job["threads"])

    t = time.perf_counter()
    fwd = harness.read_pharaoh(prefix.with_suffix(".fwd.pharaoh"))
    inv = harness.read_pharaoh(prefix.with_suffix(".inv.pharaoh"))
    pf = sw.project_many(fwd, src, tgt, by="tgt")
    pi = sw.project_many(inv, src, tgt, by="src")
    systems: dict[str, dict[str, list]] = {"proj-sym": {"fwd": pf, "rev": pi}, "sym-proj": {}}
    for m in fastalign.METHODS:
        systems["proj-sym"][m] = fastalign.symmetrize_many(pf, pi, m)
    t_project = time.perf_counter() - t  # one order: what a user pays
    for m in fastalign.METHODS:
        systems["sym-proj"][m] = sw.project_many(fastalign.symmetrize_many(fwd, inv, m), src, tgt)
    systems["sym-proj"]["fwd"], systems["sym-proj"]["rev"] = pf, pi

    off = n - nt
    links = {
        f"{o}/{name}": [sorted(x) for x in ls[off:]]
        for o, d in systems.items()
        for name, ls in d.items()
    }
    (rundir / "links.json").write_text(json.dumps(links))

    sp, sv = harness.stats(src)
    tp, tv = harness.stats(tgt)
    base = dict(  # noqa: C408
        run=run_id, corpus=corpus, lines=n, type=typ, vocab=vocab if typ != "word" else 0,
        mode=mode if typ != "word" else "word",
        t_tok=f"{t_tok:.2f}", t_encode=f"{t_encode:.2f}", t_align=f"{t_align:.2f}",
        cpu_align=f"{cpu_align:.2f}", t_project=f"{t_project:.2f}", rss_align_mb=f"{rss:.0f}",
        src_ppw=f"{sp:.4f}", tgt_ppw=f"{tp:.4f}",
        src_split=f"{harness.split_fraction(src):.4f}",
        tgt_split=f"{harness.split_fraction(tgt):.4f}",
        src_word_types=harness.word_types(src_lines), tgt_word_types=harness.word_types(tgt_lines),
        src_piece_types=sv, tgt_piece_types=tv, src_model_size=ms, tgt_model_size=mt,
        src_tokens=sum(len(p) for p in src.pieces), tgt_tokens=sum(len(p) for p in tgt.pieces),
        pretok=pretok or "default",
    )  # fmt: skip
    rows = []
    for o, d in systems.items():
        res = harness.score(d, gold, off)
        for name, r in res.items():
            row = dict(base, order=o, system=name)
            row.update(
                precision=f"{100 * r['precision']:.2f}",
                recall=f"{100 * r['recall']:.2f}",
                aer=f"{100 * r['aer']:.2f}",
            )
            rows.append(row)
    if not job["keep"]:
        for f in rundir.iterdir():
            if f.name != "links.json":
                f.unlink()
    with open(job["tsv"], "a") as f:
        f.writelines("\t".join(str(r[c]) for c in COLUMNS) + "\n" for r in rows)
    (rundir / "done").write_text("")
    gd = next(r for r in rows if r["order"] == "proj-sym" and r["system"] == "grow-diag-final-and")
    print(
        f"{run_id:42s} gdfa {gd['aer']:>6} ppw {sp:.3f}/{tp:.3f} "
        f"tok {t_tok:.1f}s align {t_align:.1f}s cpu {cpu_align:.1f}s rss {rss:.0f}MB",
        flush=True,
    )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--types", default="unigram,bpe")
    ap.add_argument("--vocabs", default="1000,2000,4000,8000,16000,32000,64000")
    ap.add_argument("--modes", default="joint,separate,src,tgt")
    ap.add_argument("--word", action="store_true", help="also run the word-level baseline")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pretok", default="", choices=sorted(PRETOK), help="see model()")
    ap.add_argument("--tsv", default=str(OUT / "sweep.tsv"))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(a.tsv):
        Path(a.tsv).write_text("\t".join(COLUMNS) + "\n")
    common = {"corpus": a.corpus, "threads": a.threads, "keep": a.keep, "force": a.force}
    common["tsv"] = a.tsv
    common["pretok"] = a.pretok
    jobs = [dict(common, type="word", vocab=0, mode="word")] if a.word else []
    jobs += [
        dict(common, type=t, vocab=int(v), mode=m)
        for t in a.types.split(",")
        if t
        for v in a.vocabs.split(",")
        if v
        for m in a.modes.split(",")
        if m
    ]
    if a.jobs <= 1:
        for j in jobs:
            one(j)
    else:
        with ProcessPoolExecutor(a.jobs) as ex:
            for _ in ex.map(one, jobs):
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
