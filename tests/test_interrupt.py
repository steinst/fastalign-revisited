"""Ctrl-C handling: training stops promptly, cleanly, and the library stays usable."""

from __future__ import annotations

import _thread
import os
import signal
import sys
import threading
import time

import fastalign
import pytest
from conftest import make_corpus

pytestmark = pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(), reason="needs the main thread"
)


def _interrupt_after(delay: float) -> threading.Timer:
    # On Windows os.kill(pid, SIGINT) terminates the process; interrupt_main raises the same
    # KeyboardInterrupt in the main thread.
    send = (
        _thread.interrupt_main
        if sys.platform == "win32"
        else (lambda: os.kill(os.getpid(), signal.SIGINT))
    )
    t = threading.Timer(delay, send)
    t.start()
    return t


def test_sigint_interrupts_train_quickly():
    src, tgt, _ = make_corpus(n=40000, vocab=2000, max_len=30, min_len=20, seed=5)
    timer = _interrupt_after(0.5)
    t0 = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            fastalign.train(src, tgt, threads=2, iterations=400)
    finally:
        timer.cancel()
    assert time.perf_counter() - t0 < 10.0
    # Not poisoned: a following call works and is not cancelled.
    m = fastalign.train(src[:50], tgt[:50], threads=1)
    assert len(m.alignments()) == 50


def test_sigint_interrupts_align_many():
    src, tgt, _ = make_corpus(n=200, vocab=40, seed=9)
    m = fastalign.train(src, tgt, threads=2)
    big_s, big_t, _ = make_corpus(n=300_000, vocab=40, max_len=40, min_len=30, seed=11)
    timer = _interrupt_after(0.3)
    t0 = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            m.align_many(big_s, big_t, threads=1)
    finally:
        timer.cancel()
    assert time.perf_counter() - t0 < 10.0
    assert len(m.align_many(src[:5], tgt[:5])) == 5


def test_gil_is_released_during_train():
    """Another Python thread keeps running while training."""
    src, tgt, _ = make_corpus(n=20000, vocab=1000, max_len=30, min_len=20, seed=3)
    ticks = []
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.005)

    th = threading.Thread(target=tick)
    th.start()
    try:
        t0 = time.perf_counter()
        fastalign.train(src, tgt, threads=1, iterations=3)
        elapsed = time.perf_counter() - t0
    finally:
        stop.set()
        th.join()
    assert elapsed > 0.3, "corpus too small to prove anything"
    assert len(ticks) > elapsed / 0.05
