#!/usr/bin/env python3
"""Generate the libstdc++ unordered_map test vectors used by Step 4's stlhash.rs tests.

    stlhash_vectors.py PROBE OUTDIR

Writes OUTDIR/cases.txt (insertion sequences, deterministic), OUTDIR/expected.txt (the probe's
output for them: bucket count after each insert, iteration order, order after copy-assignment and
after vector swap) and OUTDIR/primes.txt (libstdc++'s bucket-count prime table).
See benchmarks/stlhash_probe.cc for the formats.
"""

import random
import subprocess
import sys
from pathlib import Path


def cases():
    rng = random.Random(20261118)
    yield "u32", "empty", []
    yield "u32", "one", [7]
    yield "u32", "seq_1_20", list(range(1, 21))
    yield "u32", "seq_0_1000", list(range(1000))
    yield "u32", "rev_1000_0", list(range(1000, -1, -1))
    yield "u32", "dups_small", [rng.randrange(40) for _ in range(300)]
    yield "u32", "mult13", [13 * k for k in range(60)]
    yield "u32", "mult29", [29 * k for k in range(1, 120)]
    yield "u32", "same_bucket_13", [5 + 13 * k for k in range(13)]
    yield "u32", "rehash_edges", list(range(13)) + [100] + list(range(14, 29)) + [200, 300]
    yield "u32", "big_keys", [rng.randrange(1 << 32) for _ in range(500)]
    yield "u32", "max_keys", [0xFFFFFFFF - k for k in range(50)]
    for n in (14, 30, 60, 98, 200, 400, 800, 1600, 3300, 6800):
        yield "u32", f"random_{n}", [rng.randrange(3, 4 * n) for _ in range(n)]
    # Zipf-like target ids with repeats, as a fast_align row receives them (several rehashes).
    for n, v in ((2000, 3000), (20000, 30000)):
        yield "u32", f"zipf_{n}", [min(int(rng.paretovariate(0.8)) + 2, v) for _ in range(n)]
    yield "u32", "dense_25000", rng.sample(range(3, 60000), 25000)
    # size_counts keys (target length, source length) as shorts.
    yield "pair", "pairs_small", [(rng.randrange(1, 15), rng.randrange(1, 15)) for _ in range(200)]
    yield (
        "pair",
        "pairs_corpus",
        [(rng.randrange(0, 90), rng.randrange(0, 90)) for _ in range(5000)],
    )
    yield (
        "pair",
        "pairs_signed",
        [(-3, 4), (4, -3), (32767, 1), (-32768, -1), (0, 0), (1, 0), (0, 1)],
    )


def main():
    probe, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for kind, name, keys in cases():
        toks = [f"{a},{b}" for a, b in keys] if kind == "pair" else [str(k) for k in keys]
        lines.append(" ".join([kind, name] + toks))
    text = "\n".join(lines) + "\n"
    (out / "cases.txt").write_text(text)
    res = subprocess.run([probe], input=text, capture_output=True, text=True, check=True)
    (out / "expected.txt").write_text(res.stdout)
    res = subprocess.run([probe, "primes"], capture_output=True, text=True, check=True)
    (out / "primes.txt").write_text(res.stdout)
    print(f"{len(lines)} cases written to {out}")


if __name__ == "__main__":
    main()
