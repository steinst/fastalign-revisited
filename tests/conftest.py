"""Shared fixtures: a small synthetic parallel corpus (no real text is committed)."""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

import pytest
from textio import write_text

ROOT = Path(__file__).resolve().parent.parent
EXE = ".exe" if sys.platform == "win32" else ""
NATIVE = ROOT / "target" / "release" / f"fastalign{EXE}"
FAST_ALIGN = ROOT / "reference" / "fast_align-src" / "build" / "fast_align"
ATOOLS = ROOT / "reference" / "fast_align-src" / "build" / "atools"

# In the CI job that is meant to run the cross-frontend comparisons, FASTALIGN_REQUIRE lists the
# executables that must exist (`native`, `reference`); a missing one is an error, not a skip.
REQUIRED = set(filter(None, os.environ.get("FASTALIGN_REQUIRE", "").replace(",", " ").split()))


def _missing(kind: str, message: str):
    if kind in REQUIRED:
        pytest.fail(f"{message} (required by FASTALIGN_REQUIRE={os.environ['FASTALIGN_REQUIRE']})")
    pytest.skip(message)


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
    write_text(a, "".join(" ".join(s) + "\n" for s in src), encoding="utf-8")
    write_text(b, "".join(" ".join(t) + "\n" for t in tgt), encoding="utf-8")
    return a, b


@pytest.fixture()
def joined_file(tmp_path: Path, corpus):
    src, tgt, _ = corpus
    p = tmp_path / "c.joined"
    write_text(
        p, "".join(f"{' '.join(s)} ||| {' '.join(t)}\n" for s, t in zip(src, tgt)), encoding="utf-8"
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
        _missing("native", f"native binary not built ({NATIVE}; cargo build --release)")
    return NATIVE


@pytest.fixture(scope="session")
def fast_align_binary():
    """The reference fast_align, if the local reference build exists."""
    if not FAST_ALIGN.exists():
        _missing("reference", "reference fast_align not built")
    return FAST_ALIGN


@pytest.fixture(scope="session")
def atools_binary():
    if not ATOOLS.exists() and not shutil.which("atools"):
        _missing("reference", "reference atools not built")
    return ATOOLS
