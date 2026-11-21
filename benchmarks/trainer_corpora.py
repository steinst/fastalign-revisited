#!/usr/bin/env python3
"""Step 27: synthetic corpora for the trainer experiments (review performance items 6, 7).

    trainer_corpora.py OUTDIR [--seed 27]

Writes two `src ||| tgt` files to OUTDIR:

- `short-hv.txt`: 1,000,000 short pairs (1-6 tokens per side) over a large vocabulary (Zipf
  s=1.05 over 1M types per side; each target token is the source token's "translation" with
  probability 0.7, else drawn independently). Many phase-A tasks with little work each, and
  vocabulary-sized scratch that is large against that work.
- `long-rep.txt`: 4,000 long pairs (150-300 tokens per side) over a small vocabulary (Zipf
  s=1.2 over 3,000 types per side, same translation scheme): long lines with many repeated
  words. Train with a small `-b` (e.g. `-b 100`), or phase A has one buffer and runs serially.

Deterministic for a seed. Needs numpy.
"""

import argparse
from pathlib import Path

import numpy as np


def zipf_sampler(rng: np.random.Generator, types: int, s: float):
    w = 1.0 / np.arange(1, types + 1) ** s
    cdf = np.cumsum(w / w.sum())
    return lambda k: np.minimum(np.searchsorted(cdf, rng.random(k)), types - 1)


def write(path: Path, rng, lines: int, lo: int, hi: int, types: int, s: float) -> None:
    draw = zipf_sampler(rng, types, s)
    perm = rng.permutation(types)  # source type -> its "translation"
    with open(path, "w", encoding="ascii") as f:
        for _ in range(lines):
            n = int(rng.integers(lo, hi + 1))
            m = max(lo, min(hi, n + int(rng.integers(-1, 2))))
            src = draw(n)
            base = np.resize(src, m)
            trg = np.where(rng.random(m) < 0.7, perm[base], draw(m))
            f.write(" ".join(f"s{x}" for x in src))
            f.write(" ||| ")
            f.write(" ".join(f"t{x}" for x in trg))
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--seed", type=int, default=27)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    write(a.outdir / "short-hv.txt", rng, 1_000_000, 1, 6, 1_000_000, 1.05)
    write(a.outdir / "long-rep.txt", rng, 4_000, 150, 300, 3_000, 1.2)


if __name__ == "__main__":
    main()
