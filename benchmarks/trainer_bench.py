#!/usr/bin/env python3
"""Step 27: interleaved `phases seq` runs of several trainer builds/settings on one corpus.

    trainer_bench.py WORKDIR CORPUS_FILE --variant NAME=PHASES_BIN[,ENV=VAL...] ...
                     [--threads 20 1] [--repeat 3] [--flags "-d -o -v"] [--pin CPU]
                     [--golden DIR] [--idle-cpus 1.5] [--max-wait 14400]

Every repetition runs each variant once, in turn (interleaved, so slow drift of the machine
hits every variant alike), at each thread count, under `quiet_run.py` and `timed_run.py`.
Outputs (`fwd`, `rev`, `gdfa`, and the two logs) must be byte-identical across all runs of the
corpus: they are compared with the first run's copies kept in WORKDIR/ref (and, with
`--golden DIR`, the three alignment files with DIR/{fwd.default,rev.default,
atools.grow-diag-final-and}/out). Any difference stops the script.

`--pin CPU` pins one-thread runs to that CPU (a P-core).

Appends one row per run to WORKDIR/runs.tsv: corpus, threads, variant, rep, other_cpus,
peak_mb, then every `phase<TAB>dir<TAB>value` line `phases` printed as `phase.dir=value`.
"""

import argparse
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["fwd", "rev", "gdfa", "fwd.err", "rev.err"]
GOLD = {"fwd": "fwd.default", "rev": "rev.default", "gdfa": "atools.grow-diag-final-and"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", type=Path)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[20, 1])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--flags", default="")
    ap.add_argument("--pin", default=None)
    ap.add_argument("--golden", type=Path, default=None)
    ap.add_argument("--idle-cpus", default="1.5")
    ap.add_argument("--max-wait", default="14400")
    a = ap.parse_args()
    a.workdir.mkdir(parents=True, exist_ok=True)
    ref = a.workdir / "ref"
    ref.mkdir(exist_ok=True)
    variants = []
    for v in a.variant:
        name, spec = v.split("=", 1)
        parts = spec.split(",")
        env = dict(p.split("=", 1) for p in parts[1:])
        variants.append((name, parts[0], env))
    tsv = a.workdir / "runs.tsv"
    cname = a.corpus.stem
    for rep in range(a.repeat):
        for t in a.threads:
            for name, binary, env in variants:
                p = a.workdir / f"{cname}-{name}-t{t}-r{rep}"
                cmd = [binary, str(a.corpus), str(p), str(t), "seq", *a.flags.split()]
                if t == 1 and a.pin is not None:
                    cmd = ["taskset", "-c", a.pin, *cmd]
                full = [
                    sys.executable,
                    str(HERE / "quiet_run.py"),
                    "--idle-cpus",
                    a.idle_cpus,
                    "--max-wait",
                    a.max_wait,
                    "--",
                    sys.executable,
                    str(HERE / "timed_run.py"),
                    str(p),
                    "--",
                    *cmd,
                ]
                r = subprocess.run(
                    full, check=False, capture_output=True, text=True, env={**os.environ, **env}
                )
                if r.returncode != 0:
                    print(r.stdout, r.stderr, file=sys.stderr)
                    return 1
                m = re.search(r"other_cpus=([0-9.]+)", r.stderr)
                other = m.group(1) if m else "?"
                summary = json.loads(r.stdout.strip().splitlines()[-1])
                peak = summary.get("peak_rss_mb", "?")
                # `phases` prints its table on stdout, which timed_run passes through.
                phases = [ln.split("\t") for ln in r.stdout.splitlines() if ln.count("\t") == 2]
                for f in FILES:
                    out = Path(f"{p}.{f}")
                    keep = ref / f"{cname}.{f}"
                    if not keep.exists():
                        shutil.copy(out, keep)
                        if a.golden is not None and f in GOLD:
                            g = a.golden / GOLD[f] / "out"
                            if not filecmp.cmp(out, g, shallow=False):
                                print(f"DIFFERS from golden: {out}", file=sys.stderr)
                                return 1
                    elif not filecmp.cmp(out, keep, shallow=False):
                        print(f"DIFFERS: {out} vs {keep}", file=sys.stderr)
                        return 1
                    out.unlink()
                for x in [".log", ".rss", ".json"]:
                    Path(f"{p}{x}").unlink(missing_ok=True)
                cols = [f"{k}.{d}={v}" for k, d, v in phases]
                row = [cname, str(t), name, str(rep), other, str(peak), *cols]
                with open(tsv, "a") as f:
                    f.write("\t".join(row) + "\n")
                key = {k + "." + d: v for k, d, v in phases}
                print(
                    f"{cname} t{t} {name} r{rep}: total {key.get('total.-')} "
                    f"e {key.get('e_step.fwd')}/{key.get('e_step.rev')} "
                    f"m {key.get('m_step.fwd')}/{key.get('m_step.rev')} "
                    f"dec {key.get('decode.fwd')}/{key.get('decode.rev')} "
                    f"prior {key.get('prior.fwd')} peak {peak} other {other}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
