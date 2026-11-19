#!/usr/bin/env python3
"""Paired bootstrap over gold sentences for AER differences (Step 13).

    bootstrap.py table CORPUS [--systems fwd,rev,grow-diag-final-and] [--order proj-sym]
                              [--baseline RUN] [--md]
    bootstrap.py compare CORPUS RUN_A RUN_B [--system grow-diag-final-and] [--order-a proj-sym]
                         [--order-b proj-sym]

AER is a corpus-level ratio, so a resample draws gold sentences with replacement (the same
draw for both systems: a *paired* bootstrap), sums the four counts (|A|, |S|, |A&S|, |A&P|) of
each system over the drawn sentences and takes the AER difference. 2000 resamples, fixed seed.
A difference counts as real when its 95 % percentile interval excludes 0.

Inputs: ``reference/runs/subword/sweep/<run>/links.json`` (written by ``sweep.py``) and the
corpus's gold file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "python")]
import aer
import run as harness

SWEEP = harness.OUT / "sweep"
RESAMPLES = 2000
SEED = 13


def gold(corpus: str):
    pair = corpus.split("-")[0]
    with open(harness.CORPORA / pair / "gold.txt", encoding="utf-8") as f:
        return aer.read_gold(f)


def counts(links: list, sure, poss) -> np.ndarray:
    """Per-sentence (|A|, |S|, |A&S|, |A&P|)."""
    out = np.zeros((len(sure), 4), dtype=np.int64)
    for k, (a, s, p) in enumerate(zip(links, sure, poss)):
        a = {tuple(x) for x in a}
        p = p | s
        out[k] = (len(a), len(s), len(a & s), len(a & p))
    return out


def aer_of(c: np.ndarray) -> np.ndarray:
    """AER (in %) from summed counts, last axis = the four counts."""
    return 100 * (1 - (c[..., 2] + c[..., 3]) / (c[..., 0] + c[..., 1]))


_LINKS: dict = {}


def run_counts(corpus: str, run: str, key: str, g) -> np.ndarray:
    if run not in _LINKS:
        _LINKS[run] = json.loads((SWEEP / run / "links.json").read_text())
    return counts(_LINKS[run][key], *g)


def draws(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))


def paired(ca: np.ndarray, cb: np.ndarray, idx: np.ndarray) -> tuple[float, float, float, float]:
    """(AER a, AER b, CI low, CI high) of b - a."""
    d = aer_of(cb[idx].sum(axis=1)) - aer_of(ca[idx].sum(axis=1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(aer_of(ca.sum(0))), float(aer_of(cb.sum(0))), float(lo), float(hi)


def runs_of(corpus: str) -> list[dict]:
    with open(SWEEP / "sweep.tsv") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["corpus"] == corpus]
    seen, out = set(), []
    for r in rows:
        if r["run"] not in seen:
            seen.add(r["run"])
            out.append(r)
    return out


def cmd_table(a) -> int:
    g = gold(a.corpus)
    idx = draws(len(g[0]))
    base = a.baseline or f"{a.corpus}-word"
    systems = a.systems.split(",")
    head = ["run", "ppw src/tgt"] + [f"{s} AER (Δ, 95% CI)" for s in systems]
    lines = []
    for r in runs_of(a.corpus):
        cells = [r["run"], f"{float(r['src_ppw']):.2f}/{float(r['tgt_ppw']):.2f}"]
        for s in systems:
            key_b = f"{a.order}/{s}"
            cb = run_counts(a.corpus, r["run"], key_b, g)
            ca = run_counts(a.corpus, base, f"proj-sym/{s}", g)
            va, vb, lo, hi = paired(ca, cb, idx)
            mark = "*" if lo > 0 or hi < 0 else ""
            cells.append(f"{vb:.2f} ({vb - va:+.2f}{mark}, {lo:+.2f}..{hi:+.2f})")
        lines.append(cells)
    if a.md:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for c in lines:
            print("| " + " | ".join(c) + " |")
    else:
        for c in lines:
            print("  ".join(c))
    return 0


def cmd_compare(a) -> int:
    g = gold(a.corpus)
    idx = draws(len(g[0]))
    ca = run_counts(a.corpus, a.run_a, f"{a.order_a}/{a.system}", g)
    cb = run_counts(a.corpus, a.run_b, f"{a.order_b}/{a.system}", g)
    va, vb, lo, hi = paired(ca, cb, idx)
    print(f"{a.system}: A {va:.2f}  B {vb:.2f}  B-A {vb - va:+.2f}  95% CI {lo:+.2f}..{hi:+.2f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("corpus")
    t.add_argument("--systems", default="fwd,rev,grow-diag-final-and")
    t.add_argument("--order", default="proj-sym")
    t.add_argument("--baseline")
    t.add_argument("--md", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("corpus")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--system", default="grow-diag-final-and")
    c.add_argument("--order-a", default="proj-sym")
    c.add_argument("--order-b", default="proj-sym")
    a = ap.parse_args(argv)
    return cmd_table(a) if a.cmd == "table" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
