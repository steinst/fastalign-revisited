"""Cancellation is per operation: one call's failure or interrupt never stops another call."""

from __future__ import annotations

import threading
import time

import fastalign
import pytest
from conftest import make_corpus
from textio import write_text

MARKER = "▁"


def _segmented_file(path, n: int, seed: int) -> None:
    """A `source ||| target` file of SentencePiece-style pieces: every word is cut into a
    marked first piece and an unmarked second piece."""
    src, tgt, _ = make_corpus(n=n, vocab=300, min_len=8, max_len=14, seed=seed)

    def pieces(words):
        return " ".join(f"{MARKER}{w[:2]} {w[2:]}" if len(w) > 2 else MARKER + w for w in words)

    write_text(path, "".join(f"{pieces(s)} ||| {pieces(t)}\n" for s, t in zip(src, tgt)))


def _run(results: dict, key: str, fn) -> None:
    try:
        results[key] = fn()
    except BaseException as e:  # noqa: BLE001 - the test inspects what each job raised
        results[key] = e


@pytest.mark.parametrize("attempt", range(5))
def test_callback_failure_stops_only_its_own_job(tmp_path, attempt):
    """The review's reproduction (#2): job A fails on its final progress event while its subword
    projection is still to do; job B, released only after A has failed, enters its next
    iteration. Only A fails, with its own exception; B's result is that of a solo run."""
    a_path = tmp_path / "a.txt"
    _segmented_file(a_path, n=400_000, seed=attempt)
    b_src, b_tgt, _ = make_corpus(n=300, vocab=40, seed=100 + attempt)
    expected = fastalign.train(b_src, b_tgt, threads=1, iterations=3).alignments().to_bytes()

    barrier = threading.Barrier(2, timeout=120)
    a_failed = threading.Event()

    def a_callback(p):
        if p.iteration == p.iterations:
            barrier.wait()
            a_failed.set()
            raise ValueError("stop A")

    def b_callback(p):
        if p.direction == "fwd" and p.iteration == 1:
            barrier.wait()
            a_failed.wait(timeout=120)
            # A's exception is recorded (and its cancellation requested) as soon as the
            # callback returns; give that time to happen while A still has projection to do.
            time.sleep(0.03)

    results: dict = {}
    ta = threading.Thread(
        target=_run,
        args=(
            results,
            "a",
            lambda: fastalign.train(
                str(a_path),
                threads=1,
                directions="fwd",
                iterations=1,
                subword=fastalign.PreSegmented(),
                callback=a_callback,
            ),
        ),
    )
    tb = threading.Thread(
        target=_run,
        args=(
            results,
            "b",
            lambda: fastalign.train(b_src, b_tgt, threads=1, iterations=3, callback=b_callback),
        ),
    )
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    assert isinstance(results["a"], ValueError), repr(results["a"])
    assert str(results["a"]) == "stop A"
    assert isinstance(results["b"], fastalign.Model), repr(results["b"])
    assert results["b"].alignments().to_bytes() == expected


def test_interrupted_job_does_not_cancel_a_concurrent_one():
    """A callback failure in one job while another job trains on its own pool: the other job
    completes with the same output as alone."""
    src, tgt, _ = make_corpus(n=3000, vocab=200, min_len=10, max_len=20, seed=21)
    expected = fastalign.train(src, tgt, threads=2, iterations=6).alignments().to_bytes()
    started = threading.Event()

    def failing(p):
        started.wait(timeout=120)
        raise RuntimeError("A gave up")

    def b_callback(p):
        started.set()

    results: dict = {}
    threads = [
        threading.Thread(
            target=_run,
            args=(
                results,
                f"a{i}",
                lambda: fastalign.train(src[:500], tgt[:500], threads=1, callback=failing),
            ),
        )
        for i in range(3)
    ]
    threads.append(
        threading.Thread(
            target=_run,
            args=(
                results,
                "b",
                lambda: fastalign.train(src, tgt, threads=2, iterations=6, callback=b_callback),
            ),
        )
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in range(3):
        assert isinstance(results[f"a{i}"], RuntimeError), repr(results[f"a{i}"])
    assert isinstance(results["b"], fastalign.Model), repr(results["b"])
    assert results["b"].alignments().to_bytes() == expected
