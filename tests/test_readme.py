"""The README quick start runs as written (on a synthetic corpus)."""

from __future__ import annotations

import re
from pathlib import Path

import fastalign
import pytest
from conftest import ROOT, make_corpus


def test_quick_start_runs_as_written(tmp_path, monkeypatch):
    block = re.search(r"```python\n(.*?)```", (ROOT / "README.md").read_text().replace("\r\n", "\n"), re.DOTALL).group(1)
    src, tgt, _ = make_corpus(n=100)
    for stem, n in (("corpus", 100), ("new", 10)):
        Path(tmp_path / f"{stem}.en").write_text("".join(" ".join(s) + "\n" for s in src[:n]))
        Path(tmp_path / f"{stem}.fr").write_text("".join(" ".join(t) + "\n" for t in tgt[:n]))
    monkeypatch.chdir(tmp_path)
    # The README's own example sentence pair must be in the vocabulary for its comment to hold.
    (tmp_path / "corpus.en").write_text("the house\n" * 20 + (tmp_path / "corpus.en").read_text())
    (tmp_path / "corpus.fr").write_text("la maison\n" * 20 + (tmp_path / "corpus.fr").read_text())
    namespace: dict = {}
    exec(compile(block, "README.md", "exec"), namespace)  # noqa: S102
    assert (tmp_path / "en-fr.fastalign").exists() and (tmp_path / "out.align").exists()
    assert len((tmp_path / "new.align").read_text().splitlines()) == 10
    assert namespace["model"].align(["the", "house"], ["la", "maison"]) == [(0, 0), (1, 1)]
    assert fastalign.Model.load(tmp_path / "en-fr.fastalign").directions == ("fwd", "inv")


def test_subword_example_runs_as_written(tmp_path, monkeypatch):
    pytest.importorskip("sentencepiece")
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text().replace("\r\n", "\n"), re.DOTALL)
    block = next(b for b in blocks if "subword" in b)
    src, tgt, _ = make_corpus(n=300)
    (tmp_path / "corpus.en").write_text("".join(" ".join(s) + "\n" for s in src))
    (tmp_path / "corpus.fr").write_text("".join(" ".join(t) + "\n" for t in tgt))
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    exec(compile(block, "README.md", "exec"), namespace)  # noqa: S102
    assert len((tmp_path / "subword.align").read_text().splitlines()) == 300
    assert namespace["model"].subword["tokenizer"]
    # unseen words in, word links out (one link list per word position, no piece positions)
    links = namespace["model"].align(["the", "houses"], ["les", "maisons"])
    assert all(0 <= i < 2 and 0 <= j < 2 for i, j in links)
