#!/usr/bin/env python3
"""Saved-model inference latency: N single-pair ``Model.align`` calls vs one ``align_many``.

    infer_latency.py [--words 200,40000] [--queries 128] [--trials 3] [--threads 0]
                     [--kinds word-fwd,word-both,subword] [--distinct] [--workdir DIR]

The review's measurement (an external code review, "Single-sentence inference"): a forward-only
two-iteration model trained on distinct one-word pairs, saved and loaded again, then 128
identical queries, once as 128 ``align`` calls and once as one ``align_many`` call of 128 lines.
Each row is the median of ``--trials`` trials (each trial: one warm-up call of each kind, then
the timed calls). The Step 24 target: ``align`` per query within 2× of ``align_many``'s.

Kinds:
  word-fwd   the review's model (forward only), direction fwd
  word-both  both directions, symmetrised with grow-diag-final-and
  subword    both directions with a joint SentencePiece tokeniser (needs sentencepiece),
             symmetrised; the query is a plain word the tokeniser cuts into pieces

``--distinct`` uses 128 different training pairs instead of one pair 128 times (no repeated
word for a cache to reuse, in either call).

Uses only the public API, so it runs against any build.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
import warnings

import fastalign


def corpus(n_words: int) -> tuple[list[str], list[str]]:
    """Distinct one-word pairs: n_words distinct words in all (half source, half target)."""
    pairs = max(1, n_words // 2)
    return [f"s{i}" for i in range(pairs)], [f"t{i}" for i in range(pairs)]


def subword_corpus(n_words: int) -> tuple[list[str], list[str]]:
    # words that share pieces, so that the tokeniser has something to learn
    pairs = max(1, n_words // 2)
    stems = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]
    src = [f"{stems[i % 8]}{stems[(i // 8) % 8]}{i}" for i in range(pairs)]
    tgt = [f"{stems[(i + 3) % 8]}{stems[(i // 8 + 5) % 8]}x{i}" for i in range(pairs)]
    return src, tgt


def build(kind: str, n_words: int, workdir: str) -> fastalign.Model:
    path = os.path.join(workdir, f"{kind}-{n_words}.model")
    if not os.path.exists(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if kind == "subword":
                src, tgt = subword_corpus(n_words)
                cfg = fastalign.SubwordConfig(vocab_size=600)
                m = fastalign.train(src, tgt, iterations=2, subword=cfg)
            else:
                src, tgt = corpus(n_words)
                dirs = "fwd" if kind == "word-fwd" else "both"
                m = fastalign.train(src, tgt, iterations=2, directions=dirs)
        m.save(path)
    return fastalign.Model.load(path)


def measure(model, kind, n_words, queries, trials, threads, distinct):
    s, t = subword_corpus(n_words) if kind == "subword" else corpus(n_words)
    if distinct:
        srcs, tgts = (
            [s[k % len(s)] for k in range(queries)],
            [t[k % len(t)] for k in range(queries)],
        )
    else:
        srcs, tgts = [s[1 % len(s)]] * queries, [t[1 % len(t)]] * queries
    sel = {"direction": "fwd"} if kind == "word-fwd" else {"symmetrize": "grow-diag-final-and"}
    pairs = list(zip(srcs, tgts))
    one, many = [], []
    for _ in range(trials):
        ref = [model.align(a, b, **sel) for a, b in pairs]
        batch = model.align_many(srcs, tgts, threads=threads, **sel).to_list()
        assert batch == ref, "align and align_many disagree"
        t0 = time.perf_counter()
        for a, b in pairs:
            model.align(a, b, **sel)
        one.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        model.align_many(srcs, tgts, threads=threads, **sel).to_list()
        many.append(time.perf_counter() - t0)
    return statistics.median(one), statistics.median(many)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--words", default="200,40000")
    ap.add_argument("--queries", type=int, default=128)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--kinds", default="word-fwd,word-both,subword")
    ap.add_argument("--distinct", action="store_true")
    ap.add_argument("--workdir")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="infer-latency-") as tmp:
        workdir = a.workdir or tmp
        os.makedirs(workdir, exist_ok=True)
        print("kind\twords\tqueries\talign_ms\talign_many_ms\tper_query_ratio")
        for kind in a.kinds.split(","):
            for n in (int(x) for x in a.words.split(",")):
                model = build(kind, n, workdir)
                one, many = measure(model, kind, n, a.queries, a.trials, a.threads, a.distinct)
                print(
                    f"{kind}\t{n}\t{a.queries}\t{one * 1e3:.3f}\t{many * 1e3:.3f}\t"
                    f"{one / many:.2f}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
