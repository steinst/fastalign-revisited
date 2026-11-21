#!/usr/bin/env python3
"""Step 27: medians of `trainer_bench.py` runs.

    trainer_summary.py RUNS_TSV [--max-other 1.5]

Prints, per corpus, thread count and variant (in first-seen order), the median over repetitions
of each phase summed over both directions (seconds), the total, peak RSS (MB), the number of
runs, and the offset-cache share/size when present. Runs whose other_cpus exceed --max-other
are left out where at least one run of the same cell stays.
"""

import argparse
import statistics
from collections import OrderedDict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--max-other", type=float, default=1.5)
    a = ap.parse_args()
    cells: OrderedDict[tuple, list] = OrderedDict()
    with open(a.tsv) as fh:
        lines = fh.read().splitlines()
    for line in lines:
        f = line.rstrip("\n").split("\t")
        corpus, t, var, _rep, other, peak = f[:6]
        kv = dict(x.split("=", 1) for x in f[6:])
        cells.setdefault((corpus, int(t), var), []).append((float(other), float(peak), kv))
    cols = ["setup", "prior", "e_step", "m_step", "decode"]
    print("corpus\tthreads\tvariant\tn\t" + "\t".join(cols) + "\ttotal\tpeak_mb\tcache")
    for (corpus, t, var), runs in cells.items():
        ok = [r for r in runs if r[0] <= a.max_other] or runs

        def med(fn, ok=ok):
            return statistics.median(fn(r) for r in ok)

        vals = [med(lambda r, c=c: float(r[2][f"{c}.fwd"]) + float(r[2][f"{c}.rev"])) for c in cols]
        cache = ""
        kv = ok[0][2]
        if "offset_cache_hit.fwd" in kv:
            cache = (
                f"hit {kv['offset_cache_hit.fwd']}/{kv['offset_cache_hit.rev']} "
                f"{kv['offset_cache_mib.fwd']}/{kv['offset_cache_mib.rev']} MiB"
            )
        print(
            f"{corpus}\t{t}\t{var}\t{len(ok)}\t"
            + "\t".join(f"{v:.3f}" for v in vals)
            + f"\t{med(lambda r: float(r[2]['total.-'])):.3f}\t{med(lambda r: r[1]):.0f}\t{cache}"
        )


if __name__ == "__main__":
    main()
