#!/usr/bin/env python3
"""Step 14: reproduce the en-fr large subword run (BPE joint 8k) through the product.

    check_product.py native OUT_DIR   # native CLI on the pre-segmented pieces.txt of Step 13
    check_product.py api OUT_DIR      # Python API: fastalign.train(..., subword="auto") from the
                                      #   raw corpus (trains the tokeniser itself)

Writes fwd/inv/grow-diag-final-and word alignments (and the piece-level ones) under OUT_DIR,
prints AER/P/R on the gold sentences (the last lines of the corpus) and compares the files with
Step 13's ``words.*.pharaoh``, which came from the Python prototype's projection.
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "python")]
import aer
import fastalign

ROOT = HERE.parents[1]
CORPUS = ROOT / "reference" / "data" / "corpora" / "enfr"
STEP13 = ROOT / "reference" / "runs" / "subword" / "large" / "enfr-large-bpe8000-joint-ns"
NATIVE = ROOT / "target" / "release" / "fastalign"


def score(out: Path) -> dict:
    meta = json.loads((CORPUS / "meta.json").read_text())
    with open(CORPUS / "gold.txt", encoding="utf-8") as f:
        sure, poss = aer.read_gold(f)
    res = {}
    for name, fn in (
        ("fwd", "fwd"),
        ("rev", "inv"),
        ("gdfa", "grow-diag-final-and"),
    ):
        lines = (out / f"out.{fn}.pharaoh").read_text().splitlines()
        hyp = [set(fastalign.parse_pharaoh(x)) for x in lines[-meta["test_sentences"] :]]
        r = aer.score(hyp, sure, poss)
        res[name] = {k: round(100 * v, 2) for k, v in r.items()}
        print(f"{name:5s} P {res[name]['precision']:6.2f} R {res[name]['recall']:6.2f} "
              f"AER {res[name]['aer']:6.2f}")  # fmt: skip
    same = {}
    for fn in ("fwd", "inv", "grow-diag-final-and"):
        a = (out / f"out.{fn}.pharaoh").read_bytes()
        b = (STEP13 / f"words.{fn}.pharaoh").read_bytes()
        same[fn] = a == b
    print("identical to Step 13's word files:", same)
    return {"scores": res, "identical_to_step13": same}


def main() -> int:
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    t = time.perf_counter()
    if mode == "native":
        subprocess.run(
            [str(NATIVE), "train", str(STEP13 / "pieces.txt"), "-o", str(out / "out")]
            + ["-s", "grow-diag-final-and", "-t", str(threads), "--subword-marker", "▁", "-q"],
            check=True,
        )
    elif mode == "api":
        model = fastalign.train(
            str(CORPUS / "large.src"), str(CORPUS / "large.tgt"), threads=threads, subword="auto"
        )
        for d in ("fwd", "inv"):
            model.write_alignments(out / f"out.{d}.pharaoh", direction=d)
        model.write_alignments(out / "out.grow-diag-final-and.pharaoh", symmetrize="grow-diag-final-and")
        model.save(out / "model.fastalign")
    else:
        raise SystemExit(__doc__)
    wall = time.perf_counter() - t
    peak = resource.getrusage(resource.RUSAGE_CHILDREN if mode == "native" else resource.RUSAGE_SELF)
    print(f"{mode}: {wall:.1f} s wall, peak RSS {peak.ru_maxrss / 1024:.0f} MB")
    res = score(out)
    res.update(mode=mode, threads=threads, wall_s=round(wall, 1), peak_rss_mb=round(peak.ru_maxrss / 1024))
    (out / "check.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
