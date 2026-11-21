#!/usr/bin/env python3
"""Alignment error rate (Och and Ney 2003) against sure/possible gold links.

Gold format (HLT-NAACL 2003 shared task): ``sent_no src_pos tgt_pos S|P [confidence]``
with 1-based positions. Hypothesis format: Pharaoh, one line per sentence, ``i-j`` pairs with
0-based positions. Gold sentence numbers are mapped to hypothesis lines by their rank, so
the k-th distinct gold sentence number is compared with the k-th evaluated hypothesis line
(``--offset`` selects the first hypothesis line, for corpora with the gold sentences at the end).

    precision = |A & P| / |A|
    recall    = |A & S| / |S|
    AER       = 1 - (|A & S| + |A & P|) / (|A| + |S|)        (S is a subset of P)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from collections.abc import Iterable
from itertools import islice

Link = tuple[int, int]


def read_gold(lines: Iterable[str]) -> tuple[list[set[Link]], list[set[Link]]]:
    """Parse gold lines into per-sentence (sure, possible) sets of 0-based links.

    Links with position 0 on either side (NULL links) are ignored. Possible sets include
    the sure links. Sentences are returned in order of first appearance of their number.
    """
    sure: dict[str, set[Link]] = defaultdict(set)
    poss: dict[str, set[Link]] = defaultdict(set)
    order: list[str] = []
    for raw in lines:
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < 4 or parts[3] not in ("S", "P"):
            raise ValueError(f"bad gold line: {raw.rstrip()!r}")
        sid = parts[0]
        if sid not in sure:
            order.append(sid)
            sure[sid] = set()
            poss[sid] = set()
        i, j = int(parts[1]), int(parts[2])
        if i == 0 or j == 0:
            continue
        link = (i - 1, j - 1)
        poss[sid].add(link)
        if parts[3] == "S":
            sure[sid].add(link)
    return [sure[s] for s in order], [poss[s] for s in order]


def parse_pharaoh(line: str) -> set[Link]:
    """Parse one Pharaoh alignment line (``0-0 1-2``) into a set of links."""
    links = set()
    for tok in line.split():
        a, b = tok.split("-")
        links.add((int(a), int(b)))
    return links


def read_window(lines: Iterable[str], offset: int, n: int) -> list[set[Link]]:
    """Parse exactly ``n`` hypothesis lines starting at ``offset`` (negative: from the end).

    Streams the input: only the window is parsed and, for a negative offset, only the last
    ``-offset`` lines are kept. A file that is too short or has surplus lines after the window
    is an error (a truncated or mismatched result file must not score).
    """
    if offset < 0:
        window = [*deque(lines, maxlen=-offset)]
        if len(window) < -offset:
            raise ValueError(f"the hypothesis has {len(window)} lines, offset {offset} needs more")
        if -offset != n:
            raise ValueError(f"the last {-offset} hypothesis lines were requested, gold has {n}")
    else:
        it = iter(lines)
        for _ in islice(it, offset):
            pass
        window = list(it)
        if len(window) != n:
            raise ValueError(
                f"the hypothesis has {len(window)} lines after offset {offset}, gold has {n}"
            )
    return [parse_pharaoh(line) for line in window]


def score(hyp: list[set[Link]], sure: list[set[Link]], poss: list[set[Link]]) -> dict[str, float]:
    """Return precision, recall and AER (all as fractions in [0, 1]) over all sentences."""
    if not (len(hyp) == len(sure) == len(poss)):
        raise ValueError(
            f"sentence count mismatch: {len(hyp)} hypotheses, {len(sure)} gold sentences"
        )
    a_total = s_total = a_and_s = a_and_p = 0
    for a, s, p in zip(hyp, sure, poss):
        p = p | s
        a_total += len(a)
        s_total += len(s)
        a_and_s += len(a & s)
        a_and_p += len(a & p)
    precision = a_and_p / a_total if a_total else 0.0
    recall = a_and_s / s_total if s_total else 0.0
    denom = a_total + s_total
    aer = 1.0 - (a_and_s + a_and_p) / denom if denom else 0.0
    return {"precision": precision, "recall": recall, "aer": aer}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("hyp", help="Pharaoh-format alignment file (0-based, i-j)")
    ap.add_argument("gold", help="gold file (1-based, 'sent src tgt S|P')")
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="index of the first hypothesis line to evaluate (default 0); a negative value "
        "counts from the end, e.g. -447 evaluates the last 447 lines",
    )
    ap.add_argument("--format", choices=["text", "tsv"], default="text")
    args = ap.parse_args(argv)

    try:
        with open(args.gold, encoding="utf-8") as f:
            sure, poss = read_gold(f)
        if not sure:
            raise ValueError("the gold file has no sentences")
        with open(args.hyp, encoding="utf-8") as f:
            hyp = read_window(f, args.offset, len(sure))
        r = score(hyp, sure, poss)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.format == "tsv":
        print(f"{100 * r['precision']:.2f}\t{100 * r['recall']:.2f}\t{100 * r['aer']:.2f}")
    else:
        print(
            f"precision {100 * r['precision']:.2f}  recall {100 * r['recall']:.2f}  "
            f"AER {100 * r['aer']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
