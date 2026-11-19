"""Markdown tables for RESULTS.md from `run_scaling.sh` runs (Step 9a).

Usage: python benchmarks/collect_phases.py RUN_DIR [--t1-suffix -pin]

RUN_DIR holds `phases-tN.out` / `.json` and `cli-tN.json` (see run_scaling.sh). With
`--t1-suffix -pin`, the 1-thread numbers come from `phases-t1-pin.*` / `cli-t1-pin.*` (runs pinned
to a P-core) when they exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PHASES = ["initial_pass", "setup", "prior", "e_step", "m_step", "tension", "decode", "train_total"]
GLOBAL = ["load", "train_both", "symmetrize", "write", "total"]


def read_phases(path: Path) -> dict[tuple[str, str], float]:
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            try:
                out[(parts[0], parts[1])] = float(parts[2])
            except ValueError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--t1-suffix", default="")
    ap.add_argument("--threads", default="1 2 4 8 16 20")
    a = ap.parse_args()
    threads = [int(t) for t in a.threads.split()]

    def name(kind: str, t: int, ext: str) -> Path:
        if t == 1 and a.t1_suffix:
            p = a.run_dir / f"{kind}-t1{a.t1_suffix}.{ext}"
            if p.exists():
                return p
        return a.run_dir / f"{kind}-t{t}.{ext}"

    ph = {t: read_phases(name("phases", t, "out")) for t in threads}
    head = "| phase | " + " | ".join(f"{t} thr" for t in threads) + " | 1→20 |"
    sep = "|---" * (len(threads) + 2) + "|"
    print("### Per phase, seconds (fwd / rev, each direction trained alone)\n")
    print(head)
    print(sep)
    for p in PHASES:
        cells = []
        for t in threads:
            f, r = ph[t].get((p, "fwd")), ph[t].get((p, "rev"))
            cells.append(f"{f:.2f} / {r:.2f}" if f is not None else "–")
        f1 = ph[threads[0]][(p, "fwd")] + ph[threads[0]][(p, "rev")]
        fl = ph[threads[-1]][(p, "fwd")] + ph[threads[-1]][(p, "rev")]
        sp = f"{f1 / fl:.1f}×" if fl > 0 else "–"
        print(f"| {p} | " + " | ".join(cells) + f" | {sp} |")
    for p in GLOBAL:
        cells = [f"{ph[t].get((p, '-'), float('nan')):.2f}" for t in threads]
        v1, vl = ph[threads[0]][(p, "-")], ph[threads[-1]][(p, "-")]
        print(f"| {p} | " + " | ".join(cells) + f" | {v1 / vl:.1f}× |")
    rss = [json.loads(name("phases", t, "json").read_text())["peak_rss_mb"] for t in threads]
    print("| peak RSS (MB) | " + " | ".join(f"{r:.0f}" for r in rss) + " | |")

    print("\n### End to end: `fastalign train INPUT -o P -t N -s grow-diag-final-and`\n")
    print("| | " + " | ".join(f"{t} thr" for t in threads) + " |")
    print("|---" * (len(threads) + 1) + "|")
    cli = [json.loads(name("cli", t, "json").read_text()) for t in threads]
    print("| wall (s) | " + " | ".join(f"{c['wall_s']:.1f}" for c in cli) + " |")
    print("| speed-up | " + " | ".join(f"{cli[0]['wall_s'] / c['wall_s']:.1f}×" for c in cli) + " |")
    print("| peak RSS (MB) | " + " | ".join(f"{c['peak_rss_mb']:.0f}" for c in cli) + " |")


if __name__ == "__main__":
    main()
