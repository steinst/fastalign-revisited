#!/usr/bin/env python3
"""Step 25: wall time and peak RSS of file-producing commands on a large corpus (native and Python).

    step25_memory.py --tag before --native BIN --python PYBIN INPUT WORKDIR [--threads 20]
                     [--only NAME ...]

Every scenario is one process under `/usr/bin/time` and `quiet_run.py` (which waits for a quiet
machine and reports `other_cpus`); rows are appended to WORKDIR/step25.tsv:
tag, scenario, wall_s, peak_mb, other_cpus. `--python` is the interpreter of the build to measure
(`fastalign` imported from its venv). Scenarios that need a saved model use the one the
`*-save` scenario of the same build wrote.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, workdir, tag, name, idle):
    with tempfile.NamedTemporaryFile("r", dir=workdir, suffix=".time") as tf:
        full = [
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
            *cmd,
        ]
        r = subprocess.run(full, check=False, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAILED {name}: {r.stderr[-600:]}", file=sys.stderr)
            return
        wall, kb = tf.read().split()[-2:]
    m = re.search(r"other_cpus=([0-9.]+)", r.stderr)
    row = [tag, name, f"{float(wall):.2f}", f"{int(kb) / 1024:.0f}", m.group(1) if m else "?"]
    with open(Path(workdir) / "step25.tsv", "a") as f:
        f.write("\t".join(row) + "\n")
    print("\t".join(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--threads", default="20")
    ap.add_argument("--idle", default="1.5")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("input")
    ap.add_argument("workdir")
    a = ap.parse_args()
    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    nat, py, inp, t = a.native, a.python, a.input, a.threads
    pre = lambda n: str(wd / f"{a.tag}-{n}")
    fa = str(Path(py).parent / "fastalign")
    api = (
        "import sys, fastalign;"
        "m = fastalign.train(sys.argv[1], threads=int(sys.argv[2]));"
        "[m.write_alignments(sys.argv[3] + s, **k) for s, k in ("
        "('.fwd', dict(direction='fwd')), ('.inv', dict(direction='inv')),"
        "('.gdfa', dict(symmetrize='grow-diag-final-and')))]"
    )
    model = pre("n-save") + ".model"
    scenarios = [
        ("native-train-align", [nat, "train", inp, "-o", pre("n-align"), "-q", "-t", t]),
        (
            "native-train-table",
            [nat, "train", inp, "-o", pre("n-table"), "-q", "-t", t, "--table"],
        ),
        (
            "native-train-save",
            [nat, "train", inp, "-o", pre("n-save"), "-q", "-t", t, "--save", model],
        ),
        (
            "native-train-table-save",
            [
                nat,
                "train",
                inp,
                "-o",
                pre("n-ts"),
                "-q",
                "-t",
                t,
                "--table",
                "--save",
                pre("n-ts.m"),
            ],
        ),
        (
            "native-train-gdfa",
            [nat, "train", inp, "-o", pre("n-gdfa"), "-q", "-t", t, "-s", "grow-diag-final-and"],
        ),
        ("native-export", [nat, "export-table", model, "-o", pre("n-export"), "-t", t]),
        (
            "native-symmetrize",
            [
                nat,
                "symmetrize",
                pre("n-gdfa") + ".fwd.pharaoh",
                pre("n-gdfa") + ".inv.pharaoh",
                "-m",
                "grow-diag-final-and",
                "-o",
                pre("n-sym"),
            ],
        ),
        ("python-api-train-write", [py, "-c", api, inp, t, pre("p-api")]),
        (
            "python-cli-train-align",
            [fa, "train", inp, "-o", pre("p-align"), "-q", "-t", t],
        ),
        (
            "python-cli-train-table",
            [fa, "train", inp, "-o", pre("p-table"), "-q", "-t", t, "--table"],
        ),
        (
            "python-cli-train-save",
            [
                fa,
                "train",
                inp,
                "-o",
                pre("p-save"),
                "-q",
                "-t",
                t,
                "--save",
                pre("p-save") + ".model",
            ],
        ),
        (
            "python-cli-export",
            [fa, "export-table", pre("p-save") + ".model", "-o", pre("p-export"), "-t", t],
        ),
        (
            "python-cli-symmetrize",
            [
                fa,
                "symmetrize",
                pre("n-gdfa") + ".fwd.pharaoh",
                pre("n-gdfa") + ".inv.pharaoh",
                "-m",
                "grow-diag-final-and",
                "-o",
                pre("p-sym"),
            ],
        ),
    ]
    for name, cmd in scenarios:
        if a.only and name not in a.only:
            continue
        run(cmd, a.workdir, a.tag, name, a.idle)
        # outputs of large scenarios are only needed by later scenarios of the same build
    for p in wd.glob(f"{a.tag}-*"):
        if p.suffix == ".table" and p.stat().st_size > 1 << 28:
            p.unlink()


if __name__ == "__main__":
    sys.exit(main())
