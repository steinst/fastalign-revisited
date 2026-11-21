"""Grow-diag that scales with the number of links (Step 26, review performance item 4)."""

from __future__ import annotations

import random

import fastalign
import pytest
from test_pharaoh import _atools, _sym

GROW = ["grow-diag", "grow-diag-final", "grow-diag-final-and"]


def _random_lines(seed: int, n: int) -> tuple[bytes, bytes]:
    rng = random.Random(seed)

    def line(w: int, h: int) -> str:
        k = rng.randrange(w * h // 2 + 2)
        return " ".join(f"{rng.randrange(w)}-{rng.randrange(h)}" for _ in range(k))

    f, i = [], []
    for _ in range(n):
        w, h = rng.randint(1, 15), rng.randint(1, 15)
        f.append(line(w, h))
        i.append(line(w, h))
    return ("\n".join(f) + "\n").encode(), ("\n".join(i) + "\n").encode()


@pytest.mark.parametrize("method", fastalign.METHODS)
def test_random_lines_equal_atools(atools_binary, tmp_path, method):
    a, b = _random_lines(26, 3000)
    assert _sym(a, b, method) == _atools(atools_binary, method, a, b, tmp_path)


# Far links: the old dense grid refused (20000, 20000) ("alignment indices are too large").
FAR_F = b"0-0 1-1\n20000-20000\n5-5 20000-3\n3000-3000 2999-2999\n"
FAR_I = b"0-0 1-2\n20000-20000\n5-5 20000-4\n3000-3000 3000-2999\n"


@pytest.mark.parametrize("method", GROW)
def test_far_links_equal_atools(atools_binary, tmp_path, method):
    assert _sym(FAR_F, FAR_I, method) == _atools(atools_binary, method, FAR_F, FAR_I, tmp_path)


def test_far_link_through_the_api():
    for m in fastalign.METHODS:
        if m == "invert":
            continue
        assert fastalign.symmetrize([(20000, 20000)], [(20000, 20000)], m) == [(20000, 20000)]
    big = 4_000_000_000
    fwd = [[(0, 0), (big, big)], [(big, 1)]]
    inv = [[(0, 0), (big, big), (big, big + 1)], [(big, 1)]]
    got = fastalign.symmetrize_many(fwd, inv, "grow-diag-final-and")
    assert got == [[(0, 0), (big, big), (big, big + 1)], [(big, 1)]]
