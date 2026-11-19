"""Shared fixtures: a small synthetic parallel corpus (no real text is committed)."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "target" / "release" / "fastalign"
FAST_ALIGN = ROOT / "reference" / "fast_align-src" / "build" / "fast_align"
ATOOLS = ROOT / "reference" / "fast_align-src" / "build" / "atools"


def make_corpus(
    n: int = 400, vocab: int = 40, seed: int = 7, min_len: int = 3, max_len: int = 9
) -> tuple[list[list[str]], list[list[str]], list[list[int]]]:
    """Source sentences, target sentences and the gold permutation of each pair.

    Target word ``t{k}`` translates source word ``s{k}``; each target sentence is the source
    sentence with a few adjacent words swapped.
    """
    rng = random.Random(seed)
    src, tgt, perms = [], [], []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        words = [rng.randrange(vocab) for _ in range(length)]
        order = list(range(length))
        for _ in range(rng.randint(0, 2)):
            i = rng.randrange(length - 1)
            order[i], order[i + 1] = order[i + 1], order[i]
        src.append([f"s{w}" for w in words])
        tgt.append([f"t{words[o]}" for o in order])
        perms.append(order)
    return src, tgt, perms


@pytest.fixture(scope="session")
def corpus():
    return make_corpus()


@pytest.fixture()
def corpus_files(tmp_path: Path, corpus):
    src, tgt, _ = corpus
    a, b = tmp_path / "c.src", tmp_path / "c.tgt"
    a.write_text("".join(" ".join(s) + "\n" for s in src), encoding="utf-8")
    b.write_text("".join(" ".join(t) + "\n" for t in tgt), encoding="utf-8")
    return a, b


@pytest.fixture()
def joined_file(tmp_path: Path, corpus):
    src, tgt, _ = corpus
    p = tmp_path / "c.joined"
    p.write_text(
        "".join(f"{' '.join(s)} ||| {' '.join(t)}\n" for s, t in zip(src, tgt)), encoding="utf-8"
    )
    return p


@pytest.fixture(scope="session")
def model(corpus):
    import fastalign

    src, tgt, _ = corpus
    return fastalign.train(src, tgt, threads=2)


@pytest.fixture(scope="session")
def native_binary():
    """The native `fastalign` binary, if it has been built (`cargo build --release`)."""
    if not NATIVE.exists():
        pytest.skip("native binary not built (cargo build --release)")
    return NATIVE


@pytest.fixture(scope="session")
def fast_align_binary():
    """The reference fast_align, if the local reference build exists."""
    if not FAST_ALIGN.exists():
        pytest.skip("reference fast_align not built")
    return FAST_ALIGN


@pytest.fixture(scope="session")
def atools_binary():
    if not ATOOLS.exists() and not shutil.which("atools"):
        pytest.skip("reference atools not built")
    return ATOOLS
