#!/usr/bin/env python3
"""Step 26: per-line probability scratch on long sentences (review performance item 8).

    long_lines.py PHASES_BIN WORKDIR [--threads 20 1] [--repeat 3] [--cells 1048576 0]
                  [--lines 40] [--len 3000] [--idle-cpus 1.5]

Writes two corpora of LINES pairs with LEN tokens per side: `distinct` (every line is a random
permutation of the same LEN words per side: a line's matrix is LEN x LEN cells, and the whole
table has only LEN x LEN cells too, so the scratch is not hidden behind the table) and `repeated`
(words drawn from 40 per side: 40 x 40 cells). Each is trained with the `phases` example (`seq`,
`-d -o -v -b 1`, `FA_GROUP_LINES=1`: one line per phase-A task, so up to THREADS long lines are
in flight at once) at every thread count and every `FA_LINE_CELLS` value (0 = no cap, the matrix
as before Step 26), under `quiet_run.py` and `/usr/bin/time`.
Outputs of every run must equal the first run's (same corpus). Rows are appended to
WORKDIR/long_lines.tsv: corpus, threads, cells, wall_s, peak_mb, other_cpus.
"""

import argparse
import filecmp
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NO_CAP = str(2**62)


def corpus(path: Path, lines: int, length: int, kind: str, seed: int) -> None:
    rng = random.Random(seed)

    def side(p: str) -> str:
        if kind == "distinct":
            w = list(range(length))
            rng.shuffle(w)
        else:
            w = [rng.randrange(40) for _ in range(length)]
        return " ".join(f"{p}{x}" for x in w)

    with path.open("w") as f:
        for _ in range(lines):
            f.write(f"{side('s')} ||| {side('t')}\n")


def run(phases, inp, prefix, threads, cells, idle):
    with tempfile.NamedTemporaryFile("r", dir=prefix.parent, suffix=".time") as tf:
        cmd = [
            sys.executable,
            str(HERE / "quiet_run.py"),
            "--idle-cpus",
            str(idle),
            "--",
            "/usr/bin/time",
            "-f",
            "%e %M",
            "-o",
            tf.name,
            phases,
            str(inp),
            str(prefix),
            str(threads),
            "seq",
            "-d",
            "-o",
            "-v",
            "-b",
            "1",
        ]
        env = {
            "FA_LINE_CELLS": NO_CAP if cells == 0 else str(cells),
            "FA_GROUP_LINES": "1",
            "PATH": "/usr/bin:/bin",
        }
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            sys.exit(f"failed: {' '.join(cmd)}\n{r.stderr}")
        wall, rss = tf.read().split()
    other = re.search(r"other_cpus=([0-9.]+)", r.stderr).group(1)
    return float(wall), int(rss) / 1024, float(other)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phases")
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--threads", type=int, nargs="+", default=[20, 1])
    ap.add_argument("--cells", type=int, nargs="+", default=[1048576, 0])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--lines", type=int, default=40)
    ap.add_argument("--len", type=int, default=3000)
    ap.add_argument("--idle-cpus", type=float, default=1.5)
    ap.add_argument("--only", nargs="+", default=["distinct", "repeated"])
    a = ap.parse_args()
    a.workdir.mkdir(parents=True, exist_ok=True)
    tsv = a.workdir / "long_lines.tsv"
    for name in a.only:
        inp = a.workdir / f"{name}.txt"
        if not inp.exists():
            corpus(inp, a.lines, a.len, name, seed=26)
        first = None
        for threads in a.threads:
            for cells in a.cells:
                for rep in range(a.repeat):
                    prefix = a.workdir / f"{name}.t{threads}.c{cells}.r{rep}"
                    wall, peak, other = run(a.phases, inp, prefix, threads, cells, a.idle_cpus)
                    outs = [Path(f"{prefix}.{x}") for x in ("fwd", "rev", "gdfa")]
                    if first is None:
                        first = outs
                    else:
                        for x, y in zip(first, outs):
                            if not filecmp.cmp(x, y, shallow=False):
                                sys.exit(f"output differs: {x} {y}")
                        for x in outs:
                            x.unlink()
                    row = f"{name}\t{threads}\t{cells}\t{wall:.2f}\t{peak:.0f}\t{other:.2f}"
                    print(row, flush=True)
                    with tsv.open("a") as f:
                        f.write(row + "\n")


if __name__ == "__main__":
    main()
