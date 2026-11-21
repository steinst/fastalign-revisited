#!/usr/bin/env python3
"""Memory of repeated high-cardinality subword inference with one loaded model.

    subword_memory.py [--batches 10] [--batch-words 8000] [--input lists|files]
                      [--api align_many|align] [--workdir DIR]

Trains (once, cached in --workdir) a both-direction model with a joint SentencePiece tokeniser,
loads it, then aligns ``--batches`` batches of distinct, never-seen words (every word new: the
worst case for a per-word cache) with ``align_many``, as in-memory lists or as two files. After
each batch (input and result released) it prints the segmenter cache entries, the live Python
allocations traced since the first batch started (tracemalloc) and the process RSS. Memory has
stabilised when the last two columns stop growing. ``--api align`` aligns each line pair of a
batch with its own ``Model.align`` call instead (lists only). Every batch's output is compared with a fresh
model's (cold cache), so the bound never changes a result.

Uses only the public API (the cache is read through private attributes if they exist).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import tracemalloc
import warnings

import fastalign

STEMS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]


def words(n: int, offset: int, tag: str) -> list[str]:
    return [f"{STEMS[i % 8]}{STEMS[(i // 8) % 8]}{tag}{i}" for i in range(offset, offset + n)]


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def cache_entries(model) -> int | str:
    segs = getattr(model, "_segmenters", None)
    if not isinstance(segs, tuple):
        return "-"
    total = 0
    for s in {id(x): x for x in segs}.values():
        c = getattr(s, "_cache", None)
        if c is None:
            return "-"
        total += len(c)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--batch-words", type=int, default=8000)
    ap.add_argument("--input", choices=["lists", "files"], default="lists")
    ap.add_argument("--api", choices=["align_many", "align"], default="align_many")
    ap.add_argument("--workdir")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="subword-memory-") as tmp:
        workdir = a.workdir or tmp
        os.makedirs(workdir, exist_ok=True)
        path = os.path.join(workdir, "subword-memory.model")
        if not os.path.exists(path):
            src, tgt = words(20000, 0, "s"), words(20000, 0, "t")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cfg = fastalign.SubwordConfig(vocab_size=600)
                fastalign.train(src, tgt, iterations=2, subword=cfg).save(path)
        model = fastalign.Model.load(path)
        print("batch\tcache_entries\ttraced_kb\trss_mb")
        tracemalloc.start()
        base = None
        for b in range(a.batches):
            # ten words per line, every word new
            n = a.batch_words // 10
            src = words(a.batch_words, 10**6 + b * a.batch_words, "q")
            tgt = words(a.batch_words, 10**6 + b * a.batch_words, "r")
            src_l = [" ".join(src[k * 10 : k * 10 + 10]) for k in range(n)]
            tgt_l = [" ".join(tgt[k * 10 : k * 10 + 10]) for k in range(n)]
            if a.input == "files":
                sp, tp = os.path.join(tmp, "q.src"), os.path.join(tmp, "q.tgt")
                with open(sp, "w") as f:
                    f.write("\n".join(src_l) + "\n")
                with open(tp, "w") as f:
                    f.write("\n".join(tgt_l) + "\n")
                args = (sp, tp)
            else:
                args = (src_l, tgt_l)
            if base is None:
                gc.collect()
                base = tracemalloc.get_traced_memory()[0]
            if a.api == "align":
                out = [model.align(x, y) for x, y in zip(src_l, tgt_l)]
            else:
                out = model.align_many(*args).to_bytes()
            if b in (0, a.batches - 1):
                fresh = fastalign.Model.load(path).align_many(*args)
                want = fresh.to_list() if a.api == "align" else fresh.to_bytes()
                assert out == want, "a warm cache changed the output"
            del out, args, src, tgt, src_l, tgt_l
            gc.collect()
            traced = (tracemalloc.get_traced_memory()[0] - base) / 1024
            print(f"{b}\t{cache_entries(model)}\t{traced:.0f}\t{rss_mb():.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
