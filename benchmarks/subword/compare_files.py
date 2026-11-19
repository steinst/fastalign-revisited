#!/usr/bin/env python3
"""Paired-bootstrap AER difference of two Pharaoh files over a corpus's gold sentences.

    compare_files.py PAIR A.pharaoh B.pharaoh    (PAIR: enfr | roen; gold = the last lines)

Prints AER of A and B and the 95 % interval of B - A (``bootstrap.py``'s method).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "python")]
import bootstrap as bs
import fastalign
import run as harness


def tail(path: str, n: int) -> list:
    with open(path) as f:
        return [fastalign.parse_pharaoh(x) for x in f.readlines()[-n:]]


def main(argv: list[str]) -> int:
    pair, a, b = argv
    meta = json.loads((harness.CORPORA / pair / "meta.json").read_text())
    n = meta["test_sentences"]
    g = bs.gold(f"{pair}-x")
    ca, cb = bs.counts(tail(a, n), *g), bs.counts(tail(b, n), *g)
    va, vb, lo, hi = bs.paired(ca, cb, bs.draws(n))
    print(f"A {va:.2f}  B {vb:.2f}  B-A {vb - va:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
