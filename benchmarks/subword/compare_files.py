#!/usr/bin/env python3
"""Paired-bootstrap AER difference of two Pharaoh files over a corpus's gold sentences.

    compare_files.py PAIR A.pharaoh B.pharaoh    (PAIR: enfr | roen; gold = the last lines)

Prints AER of A and B and the 95 % interval of B - A (``bootstrap.py``'s method).
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "python")]
import bootstrap as bs
import fastalign
import run as harness


def tail(path: str, n: int) -> list:
    """The last ``n`` lines of a Pharaoh file, streamed; fewer than ``n`` lines is an error."""
    with open(path) as f:
        last = [*deque(f, maxlen=n)]
    if len(last) < n:
        raise ValueError(f"{path} has {len(last)} lines, {n} gold sentences need at least {n}")
    return [fastalign.parse_pharaoh(x) for x in last]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    pair, a, b = argv
    meta = json.loads((harness.CORPORA / pair / "meta.json").read_text())
    n = meta["test_sentences"]
    g = bs.gold(f"{pair}-x")
    if len(g[0]) != n:
        raise ValueError(f"meta.json says {n} test sentences, the gold file has {len(g[0])}")
    ca, cb = bs.counts(tail(a, n), *g), bs.counts(tail(b, n), *g)
    va, vb, lo, hi = bs.paired(ca, cb, bs.draws(n))
    print(f"A {va:.2f}  B {vb:.2f}  B-A {vb - va:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
