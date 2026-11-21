#!/usr/bin/env python3
"""Step 26: wall time of native `symmetrize` on the en-fr large golden alignments.

    symmetrize_time.py WORKDIR TAG=BIN [TAG=BIN ...] [--threads 1 20] [--repeat 3]
                       [--methods grow-diag grow-diag-final grow-diag-final-and]

Runs `BIN symmetrize -m METHOD -t THREADS fwd rev -o OUT` on
reference/golden/enfr-large/{fwd,rev}.default/out under `quiet_run.py` and `/usr/bin/time`,
builds interleaved (one repetition of every build before the next), checks every output against
atools' golden, and appends rows to WORKDIR/symmetrize_time.tsv: tag, method, threads, wall_s,
peak_mb, other_cpus.
"""

import argparse
import filecmp
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLD = HERE.parent / "reference/golden/enfr-large"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", type=Path)
    ap.add_argument("builds", nargs="+")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 20])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument(
        "--methods", nargs="+", default=["grow-diag", "grow-diag-final", "grow-diag-final-and"]
    )
    ap.add_argument("--idle-cpus", type=float, default=1.5)
    a = ap.parse_args()
    a.workdir.mkdir(parents=True, exist_ok=True)
    builds = [b.split("=", 1) for b in a.builds]
    fwd, rev = GOLD / "fwd.default/out", GOLD / "rev.default/out"
    for method in a.methods:
        for threads in a.threads:
            for _ in range(a.repeat):
                for tag, binary in builds:
                    out = a.workdir / f"{tag}.{method}.out"
                    with tempfile.NamedTemporaryFile("r", dir=a.workdir, suffix=".time") as tf:
                        cmd = [
                            sys.executable, str(HERE / "quiet_run.py"),
                            "--idle-cpus", str(a.idle_cpus), "--",
                            "/usr/bin/time", "-f", "%e %M", "-o", tf.name,
                            binary, "symmetrize", "-m", method, "-t", str(threads),
                            str(fwd), str(rev), "-o", str(out),
                        ]  # fmt: skip
                        r = subprocess.run(cmd, capture_output=True, text=True)
                        if r.returncode != 0:
                            sys.exit(f"failed: {' '.join(cmd)}\n{r.stderr}")
                        wall, rss = tf.read().split()
                    other = re.search(r"other_cpus=([0-9.]+)", r.stderr).group(1)
                    gold = GOLD / f"atools.{method}/out"
                    if not filecmp.cmp(out, gold, shallow=False):
                        sys.exit(f"{tag} {method}: output differs from {gold}")
                    out.unlink()
                    row = f"{tag}\t{method}\t{threads}\t{wall}\t{int(rss) / 1024:.0f}\t{other}"
                    print(row, flush=True)
                    with (a.workdir / "symmetrize_time.tsv").open("a") as f:
                        f.write(row + "\n")


if __name__ == "__main__":
    main()
