"""The README quick start runs as written (on a synthetic corpus)."""

from __future__ import annotations

import re
from pathlib import Path

import fastalign
import pytest
from conftest import ROOT, make_corpus
from textio import read_text, write_text


def test_quick_start_runs_as_written(tmp_path, monkeypatch):
    block = re.search(
        r"```python\n(.*?)```", read_text(ROOT / "README.md").replace("\r\n", "\n"), re.DOTALL
    ).group(1)
    src, tgt, _ = make_corpus(n=100)
    for stem, n in (("corpus", 100), ("new", 10)):
        write_text(Path(tmp_path / f"{stem}.en"), "".join(" ".join(s) + "\n" for s in src[:n]))
        write_text(Path(tmp_path / f"{stem}.fr"), "".join(" ".join(t) + "\n" for t in tgt[:n]))
    monkeypatch.chdir(tmp_path)
    # The README's own example sentence pair must be in the vocabulary for its comment to hold.
    write_text(tmp_path / "corpus.en", "the house\n" * 20 + read_text(tmp_path / "corpus.en"))
    write_text(tmp_path / "corpus.fr", "la maison\n" * 20 + read_text(tmp_path / "corpus.fr"))
    namespace: dict = {}
    exec(compile(block, "README.md", "exec"), namespace)  # noqa: S102
    assert (tmp_path / "en-fr.fastalign").exists() and (tmp_path / "out.align").exists()
    assert len(read_text(tmp_path / "new.align").splitlines()) == 10
    assert namespace["model"].align(["the", "house"], ["la", "maison"]) == [(0, 0), (1, 1)]
    assert fastalign.Model.load(tmp_path / "en-fr.fastalign").directions == ("fwd", "inv")


def test_subword_example_runs_as_written(tmp_path, monkeypatch):
    pytest.importorskip("sentencepiece")
    blocks = re.findall(
        r"```python\n(.*?)```", read_text(ROOT / "README.md").replace("\r\n", "\n"), re.DOTALL
    )
    block = next(b for b in blocks if "subword" in b)
    src, tgt, _ = make_corpus(n=300)
    write_text(tmp_path / "corpus.en", "".join(" ".join(s) + "\n" for s in src))
    write_text(tmp_path / "corpus.fr", "".join(" ".join(t) + "\n" for t in tgt))
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    exec(compile(block, "README.md", "exec"), namespace)  # noqa: S102
    assert len(read_text(tmp_path / "subword.align").splitlines()) == 300
    assert namespace["model"].subword["tokenizer"]
    # unseen words in, word links out (one link list per word position, no piece positions)
    links = namespace["model"].align(["the", "houses"], ["les", "maisons"])
    assert all(0 <= i < 2 and 0 <= j < 2 for i, j in links)
