#!/usr/bin/env python3
"""Compare our output with the golden fast_align output; report the first difference.

    parity_compare.py KIND GOLDEN OURS

KIND is ``out`` (alignment lines, byte-exact), ``err`` (stderr statistics, byte-exact after
progress dots and ``[N]`` counters are removed) or ``table`` (the ``-p`` table, byte-exact).
Exit status 0 if identical, 1 if not (with a message on stdout).
"""

from __future__ import annotations

import re
import sys

PROGRESS = re.compile(rb"^\.*( \[\d+\])?$")
TABLE_LINE = re.compile(rb"^(conditional probabilities: ).*$")


def read(path: str, kind: str) -> list[bytes]:
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    if kind == "err":
        lines = [ln for ln in lines if not PROGRESS.match(ln)]
        # The -p path is echoed; it differs between golden and test runs.
        lines = [TABLE_LINE.sub(rb"\1<table>", ln) for ln in lines]
    return lines


def main() -> int:
    kind, gold_path, our_path = sys.argv[1:4]
    gold, ours = read(gold_path, kind), read(our_path, kind)
    if gold == ours:
        return 0
    header = ""
    for i in range(max(len(gold), len(ours))):
        g = gold[i] if i < len(gold) else None
        o = ours[i] if i < len(ours) else None
        if kind == "err" and g is not None and g.startswith(b"ITERATION"):
            header = g.decode(errors="replace")
        if g != o:
            where = f"line {i + 1}" + (f" (after '{header}')" if header else "")
            print(f"  first difference at {where}")
            print(f"    fast_align: {g.decode(errors='replace') if g is not None else '<missing>'}")
            print(f"    ours:       {o.decode(errors='replace') if o is not None else '<missing>'}")
            ndiff = sum(1 for a, b in zip(gold, ours) if a != b) + abs(len(gold) - len(ours))
            print(f"  {ndiff} differing line(s); {len(gold)} vs {len(ours)} lines")
            return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
