#!/usr/bin/env python3
"""Subword alignment of a big corpus with streaming encode and projection (Step 13).

    large.py prepare CORPUS --type unigram --vocab N --mode joint|separate|src|tgt [--pretok ns]
    large.py score CORPUS RUN_DIR

``prepare`` trains (or reuses) the tokeniser model(s) on the whole corpus, then encodes it per
word line by line into ``RUN_DIR/pieces.txt`` and prints the run directory. The aligner is run
separately (``fastalign train RUN_DIR/pieces.txt -o RUN_DIR/align -s grow-diag-final-and``), so
that it can be timed on a quiet machine with ``quiet_run.py``. ``score`` then projects the piece
alignments to words line by line (word boundaries recovered from the word-start marker, as the
native CLI will do in Step 14), writes ``words.fwd.pharaoh``/``words.inv.pharaoh``, symmetrises
them with the native ``fastalign symmetrize`` (project-then-symmetrise), projects the aligner's
own ``align.grow-diag-final-and.pharaoh`` (symmetrise-then-project), and scores fwd, rev and
both grow-diag-final-and variants on the gold sentences (the last lines of the corpus).

The positional piece -> word map of ``subword.encode`` and the marker rule give the same words
for every line; ``prepare`` checks that on the first 20,000 lines.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "python")]
import aer
import fastalign
import run as harness
import sweep
from fastalign import subword as sw

OUT = harness.OUT / "large"


def side_lines(path: Path):
    with open(path, "rb") as f:
        for line in f:
            yield line.split()


def cmd_prepare(a) -> int:
    pair, size = a.corpus.split("-")
    d = harness.CORPORA / pair
    run_id = f"{a.corpus}-{a.type}{a.vocab}-{a.mode}" + (f"-{a.pretok}" if a.pretok else "")
    rundir = OUT / run_id
    rundir.mkdir(parents=True, exist_ok=True)
    info: dict = {"run": run_id, "type": a.type, "vocab": a.vocab, "mode": a.mode}
    info["pretok"] = a.pretok or "default"

    t = time.perf_counter()
    if a.mode in ("joint", "separate", "src"):
        src_all = list(side_lines(d / f"{size}.src"))
    if a.mode in ("joint", "separate", "tgt"):
        tgt_all = list(side_lines(d / f"{size}.tgt"))
    t_read = time.perf_counter() - t
    t = time.perf_counter()
    if a.mode == "joint":
        m = sweep.model(a.corpus, a.type, a.vocab, "joint", src_all + tgt_all, a.pretok)
        seg_s = seg_t = sw.SentencePieceSegmenter(m)
    else:
        seg_s = seg_t = sw.WholeWords()
        if a.mode in ("separate", "src"):
            seg_s = sw.SentencePieceSegmenter(
                sweep.model(a.corpus, a.type, a.vocab, "src", src_all, a.pretok)
            )
        if a.mode in ("separate", "tgt"):
            seg_t = sw.SentencePieceSegmenter(
                sweep.model(a.corpus, a.type, a.vocab, "tgt", tgt_all, a.pretok)
            )
    info["t_read"] = round(t_read, 2)
    info["t_tok"] = round(time.perf_counter() - t, 2)
    info["rss_tok_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    src_all = tgt_all = None

    t = time.perf_counter()
    n = n_src_w = n_tgt_w = n_src_p = n_tgt_p = 0
    seg_flags = (a.mode in ("joint", "separate", "src"), a.mode in ("joint", "separate", "tgt"))
    with (
        open(d / f"{size}.src", "rb") as fs,
        open(d / f"{size}.tgt", "rb") as ft,
        open(rundir / "pieces.txt", "w", encoding="utf-8", newline="\n") as out,
    ):
        for ls, lt in zip(fs, ft):
            ws, wt = ls.split(), lt.split()
            ps = [p for w in ws for p in seg_s.encode_word(w)]
            pt = [p for w in wt for p in seg_t.encode_word(w)]
            if n < 20000:  # the marker rule gives the positional map
                for words, pieces, seg, on in (
                    (ws, ps, seg_s, seg_flags[0]),
                    (wt, pt, seg_t, seg_flags[1]),
                ):
                    pos = [k for k, w in enumerate(words) for _ in seg.encode_word(w)]
                    rec = sw.split_into_words(pieces) if on else list(range(len(pieces)))
                    assert pos == rec, (n, words)
            out.write(" ".join(ps) + " ||| " + " ".join(pt) + "\n")
            n += 1
            n_src_w += len(ws)
            n_tgt_w += len(wt)
            n_src_p += len(ps)
            n_tgt_p += len(pt)
    info.update(
        t_encode=round(time.perf_counter() - t, 2),
        lines=n,
        src_ppw=round(n_src_p / n_src_w, 4),
        tgt_ppw=round(n_tgt_p / n_tgt_w, 4),
        src_tokens=n_src_p,
        tgt_tokens=n_tgt_p,
        seg_src=seg_flags[0],
        seg_tgt=seg_flags[1],
        rss_self_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    )
    (rundir / "prepare.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    print(rundir)
    return 0


def word_map(pieces: list[str], segmented: bool) -> list[int]:
    return sw.split_into_words(pieces) if segmented else list(range(len(pieces)))


def cmd_score(a) -> int:
    rundir = Path(a.run_dir)
    info = json.loads((rundir / "prepare.json").read_text())
    pair = a.corpus.split("-")[0]
    meta = json.loads((harness.CORPORA / pair / "meta.json").read_text())
    with open(harness.CORPORA / pair / "gold.txt", encoding="utf-8") as f:
        sure, poss = aer.read_gold(f)
    off = info["lines"] - meta["test_sentences"]
    prefix = rundir / "align"
    t = time.perf_counter()
    sym_proj = []
    with (
        open(rundir / "pieces.txt", encoding="utf-8") as fp,
        open(prefix.with_suffix(".fwd.pharaoh")) as ff,
        open(prefix.with_suffix(".inv.pharaoh")) as fi,
        open(rundir / "words.fwd.pharaoh", "w") as of,
        open(rundir / "words.inv.pharaoh", "w") as oi,
    ):
        gd = open(prefix.with_suffix(".grow-diag-final-and.pharaoh"))  # noqa: SIM115
        for k, (lp, lf, li, lg) in enumerate(zip(fp, ff, fi, gd)):
            s, _, tg = lp.partition(" ||| ")
            sm = word_map(s.split(), info["seg_src"])
            tm = word_map(tg.split(), info["seg_tgt"])
            wf = sw.project(fastalign.parse_pharaoh(lf), sm, tm, by="tgt")
            wi = sw.project(fastalign.parse_pharaoh(li), sm, tm, by="src")
            of.write(" ".join(f"{i}-{j}" for i, j in wf) + "\n")
            oi.write(" ".join(f"{i}-{j}" for i, j in wi) + "\n")
            if k >= off:
                sym_proj.append(set(sw.project(fastalign.parse_pharaoh(lg), sm, tm)))
        gd.close()
    t_project = time.perf_counter() - t
    t = time.perf_counter()
    subprocess.run(
        [str(harness.NATIVE), "symmetrize", str(rundir / "words.fwd.pharaoh")]
        + [str(rundir / "words.inv.pharaoh"), "-m", "grow-diag-final-and"]
        + ["-o", str(rundir / "words.grow-diag-final-and.pharaoh")],
        check=True,
    )
    t_sym = time.perf_counter() - t

    def tail(path: Path) -> list[set]:
        with open(path) as f:
            return [set(fastalign.parse_pharaoh(x)) for x in f.readlines()[off:]]

    systems = {
        "fwd": tail(rundir / "words.fwd.pharaoh"),
        "rev": tail(rundir / "words.inv.pharaoh"),
        "proj-sym/grow-diag-final-and": tail(rundir / "words.grow-diag-final-and.pharaoh"),
        "sym-proj/grow-diag-final-and": sym_proj,
    }
    res = {}
    for name, hyp in systems.items():
        r = aer.score(hyp, sure, poss)
        res[name] = {k: round(100 * v, 2) for k, v in r.items()}
        print(f"{name:30s} P {res[name]['precision']:6.2f} R {res[name]['recall']:6.2f} "
              f"AER {res[name]['aer']:6.2f}")  # fmt: skip
    links = {k: [sorted(x) for x in v] for k, v in systems.items()}
    (rundir / "links.json").write_text(json.dumps(links))
    out = {"t_project": round(t_project, 2), "t_symmetrize": round(t_sym, 2), "scores": res}
    (rundir / "score.json").write_text(json.dumps(out, indent=1))
    print(f"project {t_project:.1f}s symmetrize {t_sym:.1f}s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("corpus")
    p.add_argument("--type", default="unigram", choices=["unigram", "bpe"])
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--mode", default="joint", choices=["joint", "separate", "src", "tgt"])
    p.add_argument("--pretok", default="", choices=sorted(sweep.PRETOK))
    s = sub.add_parser("score")
    s.add_argument("corpus")
    s.add_argument("run_dir")
    a = ap.parse_args(argv)
    return cmd_prepare(a) if a.cmd == "prepare" else cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())
