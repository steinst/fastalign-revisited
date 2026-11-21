#!/usr/bin/env python3
"""Ctrl-C latency of file-based Python calls, per phase (Step 18).

For each phase, the call is first run to completion to time the phase, then run again with a
SIGINT sent half-way through the phase; the latency is the time from the signal to the
`KeyboardInterrupt` in the calling thread. Phases:

  parse       train(corpus, max_len=1): reading dominates (nearly every line is dropped)
  setup       train(corpus, iterations=0): reading, then the initial pass and trainer setup
  iteration   train(corpus, iterations=2): an EM iteration (checked before Step 18 too)
  projection  train(pre-segmented corpus, subword=PreSegmented()): the signal is sent from
              the final progress event, half-way through the projection that follows
  table       align_with_table(table, corpus): fast_align's -f loop (text table read first)

Usage: cancel_latency.py CORPUS WORKDIR [--threads N] [--reps N] [--table-corpus FILE]
CORPUS is a `source ||| target` file (en-fr large), WORKDIR receives the segmented copy and
the table. Linux/macOS only (SIGINT via os.kill).
"""

from __future__ import annotations

import argparse
import os
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

import fastalign

MARKER = "▁"


def kill_after(delay: float, sent: list[float]) -> threading.Timer:
    def fire():
        sent.append(time.perf_counter())
        os.kill(os.getpid(), signal.SIGINT)

    t = threading.Timer(delay, fire)
    t.start()
    return t


def interrupted(run, delay: float | None, *, from_callback=None) -> float:
    """Run `run(callback)` with a SIGINT `delay` seconds after the start (or after the event
    `from_callback` selects). Returns the latency in seconds."""
    sent: list[float] = []
    timers: list[threading.Timer] = []

    def cb(p):
        if from_callback is not None and from_callback(p) is not None:
            timers.append(kill_after(from_callback(p), sent))

    if delay is not None:
        timers.append(kill_after(delay, sent))
    try:
        run(cb)
    except KeyboardInterrupt:
        got = time.perf_counter()
    else:
        raise SystemExit("the call finished before the signal: phase timing is off")
    finally:
        for t in timers:
            t.cancel()
    if not sent:
        raise SystemExit("KeyboardInterrupt without a signal")
    return got - sent[0]


def timed(run) -> float:
    t0 = time.perf_counter()
    run(None)
    return time.perf_counter() - t0


def segmented_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    tmp = dst.with_suffix(".tmp")
    with open(src, "rb") as f, open(tmp, "wb") as g:
        m = MARKER.encode()
        g.writelines(
            b" ".join(m + w if w != b"|||" else w for w in line.split()) + b"\n" for line in f
        )
    tmp.rename(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--table-corpus", type=Path, help="corpus to train the -p table on")
    ap.add_argument("--phases", default="parse,setup,iteration,projection,table")
    a = ap.parse_args()
    a.workdir.mkdir(parents=True, exist_ok=True)
    corpus, thr = str(a.corpus), a.threads
    phases = a.phases.split(",")
    rows = []

    def report(name, duration, lat):
        rows.append((name, duration, lat))
        print(
            f"{name:10s} phase {duration:7.2f} s  latency median {statistics.median(lat) * 1e3:7.1f} ms"
            f"  max {max(lat) * 1e3:7.1f} ms  ({len(lat)} runs)",
            flush=True,
        )

    def parse_run(cb):
        try:
            fastalign.train(corpus, threads=thr, directions="fwd", iterations=1, max_len=1)
        except ValueError:
            pass  # every line dropped: reading was the whole job

    t_parse = timed(parse_run)
    if "parse" in phases:
        report("parse", t_parse, [interrupted(parse_run, t_parse / 2) for _ in range(a.reps)])

    def setup_run(cb):
        fastalign.train(corpus, threads=thr, directions="fwd", iterations=0)

    t_setup = timed(setup_run) - t_parse
    if "setup" in phases:
        lat = [interrupted(setup_run, t_parse + t_setup / 2) for _ in range(a.reps)]
        report("setup", t_setup, lat)

    if "iteration" in phases:
        events: list[float] = []

        def iter_run(cb):
            fastalign.train(
                corpus,
                threads=thr,
                directions="fwd",
                iterations=2,
                callback=lambda p: events.append(time.perf_counter()),
            )

        t0 = time.perf_counter()
        iter_run(None)
        t_iter = events[0] - t0 - t_parse - t_setup
        lat = [interrupted(iter_run, t_parse + t_setup + t_iter / 2) for _ in range(a.reps)]
        report("iteration", t_iter, lat)

    if "projection" in phases:
        seg = a.workdir / "segmented.txt"
        segmented_copy(a.corpus, seg)
        marks: list[float] = []

        def proj_run(cb):
            def on(p):
                if p.iteration == p.iterations:
                    marks.append(time.perf_counter())
                if cb:
                    cb(p)

            fastalign.train(
                str(seg),
                threads=thr,
                directions="fwd",
                iterations=1,
                subword=fastalign.PreSegmented(),
                callback=on,
            )

        proj_run(None)
        t_proj = time.perf_counter() - marks[-1]  # projection + model assembly
        lat = [
            interrupted(
                proj_run,
                None,
                from_callback=lambda p: t_proj / 2 if p.iteration == p.iterations else None,
            )
            for _ in range(a.reps)
        ]
        report("projection", t_proj, lat)

    if "table" in phases:
        table = a.workdir / "fwd.table"
        if not table.exists():
            src = str(a.table_corpus or a.corpus)
            m = fastalign.train(src, threads=0, directions="fwd")
            m.write_table(table, direction="fwd")

        def table_run(cb):
            fastalign.align_with_table(table, corpus, threads=thr)

        one = a.workdir / "one.txt"
        with open(a.corpus, "rb") as f:
            one.write_bytes(f.readline())
        t0 = time.perf_counter()
        fastalign.align_with_table(table, str(one), threads=thr)
        t_read_table = time.perf_counter() - t0
        t_total = timed(table_run)
        t_loop = t_total - t_parse - t_read_table
        lat = [interrupted(table_run, t_parse + t_read_table + t_loop / 2) for _ in range(a.reps)]
        report("table", t_loop, lat)
        print(f"           (table read {t_read_table:.2f} s, corpus read {t_parse:.2f} s)")

    print("\nphase\tphase_s\tlatency_median_ms\tlatency_max_ms")
    for name, d, lat in rows:
        print(f"{name}\t{d:.2f}\t{statistics.median(lat) * 1e3:.1f}\t{max(lat) * 1e3:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
