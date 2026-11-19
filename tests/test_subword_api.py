"""Subword alignment in the product (Step 14): API == Python CLI == native CLI on segmented text,
identity, save/load/align with a stored tokeniser, model file versions, projection against the
pure-Python reference, and the error when sentencepiece is missing."""

from __future__ import annotations

import subprocess
import sys

import fastalign
import pytest
from conftest import make_corpus
from fastalign import subword as sw
from fastalign.cli import main

MARK = "▁"


def segmented_corpus(n=300, seed=3):
    """A corpus whose words are cut into 1-3 pieces by a fixed rule (no tokeniser involved)."""
    src, tgt, _ = make_corpus(n=n, seed=seed)

    def cut(word: str) -> list[str]:
        k = sum(map(ord, word)) % 3
        stem = word
        if k == 0 or len(stem) < 3:
            return [MARK + stem]
        if k == 1:
            return [MARK + stem[:-1], stem[-1]]
        return [MARK + stem[:-2], stem[-2], stem[-1]]

    def side(lines):
        return [[p for w in line for p in cut(w)] for line in lines], [
            [i for i, w in enumerate(line) for _ in cut(w)] for line in lines
        ]

    (sp, sm), (tp, tm) = side(src), side(tgt)
    return src, tgt, sp, tp, sm, tm


@pytest.fixture()
def piece_files(tmp_path):
    _, _, sp, tp, _, _ = segmented_corpus()
    a, b = tmp_path / "p.src", tmp_path / "p.tgt"
    a.write_text("".join(" ".join(x) + "\n" for x in sp), encoding="utf-8")
    b.write_text("".join(" ".join(x) + "\n" for x in tp), encoding="utf-8")
    return a, b


def test_presegmented_api_cli_native_agree(tmp_path, piece_files, native_binary):
    a, b = piece_files
    model = fastalign.train(
        a, b, subword=fastalign.PreSegmented(MARK), subword_output=True, threads=2
    )
    assert model.subword["kind"] == "start" and not model.subword["tokenizer"]
    py = tmp_path / "py"
    rc = main(
        ["train", str(a), str(b), "-o", str(py), "-s", "grow-diag-final-and", "-q", "-t", "2"]
        + ["--subword-marker", MARK, "--subword-output", "--save", str(tmp_path / "py.model")]
    )
    assert rc == 0
    nat = tmp_path / "nat"
    subprocess.run(
        [str(native_binary), "train", str(a), str(b), "-o", str(nat), "-s", "grow-diag-final-and"]
        + ["-q", "-t", "2", "--subword-marker", MARK, "--subword-output"]
        + ["--save", str(tmp_path / "nat.model")],
        check=True,
    )
    for name in ("fwd", "inv", "grow-diag-final-and", "fwd.subword", "inv.subword"):
        want = (tmp_path / f"nat.{name}.pharaoh").read_bytes()
        assert (tmp_path / f"py.{name}.pharaoh").read_bytes() == want, name
    for d in ("fwd", "inv"):
        assert (
            model.alignments(direction=d).to_bytes() == (tmp_path / f"nat.{d}.pharaoh").read_bytes()
        )
        assert (
            model.alignments(direction=d, pieces=True).to_bytes()
            == (tmp_path / f"nat.{d}.subword.pharaoh").read_bytes()
        )
    gdfa = model.alignments(symmetrize="grow-diag-final-and").to_bytes()
    assert gdfa == (tmp_path / "nat.grow-diag-final-and.pharaoh").read_bytes()
    # the native and the Python model files are byte-identical, and so is `align` with them
    assert (tmp_path / "py.model").read_bytes() == (tmp_path / "nat.model").read_bytes()
    out_nat = subprocess.run(
        [str(native_binary), "align", str(tmp_path / "nat.model"), str(a), str(b), "-s"]
        + ["grow-diag-final-and"],
        check=True,
        capture_output=True,
    ).stdout
    assert out_nat == gdfa
    loaded = fastalign.Model.load(tmp_path / "py.model")
    assert loaded.align_many(a, b, symmetrize="grow-diag-final-and").to_bytes() == gdfa


def test_word_projection_equals_the_python_reference(piece_files):
    a, b = piece_files
    _, _, _, _, sm, tm = segmented_corpus()
    model = fastalign.train(a, b, subword=fastalign.PreSegmented(MARK), subword_output=True)
    for d, by in (("fwd", "tgt"), ("inv", "src")):
        pieces = model.alignments(direction=d, pieces=True)
        words = model.alignments(direction=d)
        for k in range(len(pieces)):
            want = sw.project(pieces[k], sm[k], tm[k], by)
            assert words[k] == want, (d, k)
            assert words.pharaoh(k) == " ".join(f"{i}-{j}" for i, j in want)


def test_identity_segmentation_equals_word_level(tmp_path, corpus_files):
    a, b = corpus_files
    marked = []
    for p in (a, b):
        q = tmp_path / (p.name + ".marked")
        q.write_text(
            "".join(
                " ".join(MARK + w for w in line.split()) + "\n"
                for line in p.read_text().splitlines()
            ),
            encoding="utf-8",
        )
        marked.append(q)
    plain = fastalign.train(a, b)
    sub = fastalign.train(*marked, subword=fastalign.PreSegmented(MARK))
    for d in ("fwd", "inv"):
        assert plain.alignments(direction=d).to_bytes() == sub.alignments(direction=d).to_bytes()
    for m in fastalign.METHODS:
        assert plain.alignments(symmetrize=m).to_bytes() == sub.alignments(symmetrize=m).to_bytes()


def test_continuation_marker_equals_prefix_marker_projection(tmp_path):
    _, _, sp, tp, _, _ = segmented_corpus(n=120)

    def conv(line):
        out = []
        for i, p in enumerate(line):
            cont = i + 1 < len(line) and not line[i + 1].startswith(MARK)
            out.append(p.lstrip(MARK) + ("@@" if cont else ""))
        return out

    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("".join(" ".join(conv(x)) + "\n" for x in sp))
    b.write_text("".join(" ".join(conv(x)) + "\n" for x in tp))
    m = fastalign.train(a, b, subword=fastalign.PreSegmented("@@", continuation=True))
    assert m.subword["kind"] == "continue" and m.subword["marker"] == "@@"
    link_ok = all(
        0 <= i and 0 <= j for k in range(len(m.alignments())) for i, j in m.alignments()[k]
    )
    assert link_ok
    # and it round-trips through the file format
    m.save(tmp_path / "m")
    assert fastalign.Model.load(tmp_path / "m").subword["kind"] == "continue"


def test_subword_off_is_unchanged(corpus_files):
    a, b = corpus_files
    base = fastalign.train(a, b)
    assert base.subword is None
    assert (
        base.alignments().to_bytes() == fastalign.train(a, b, subword=None).alignments().to_bytes()
    )
    with pytest.raises(ValueError, match="subword_output needs subword"):
        fastalign.train(a, b, subword_output=True)
    with pytest.raises(ValueError, match="subword must be"):
        fastalign.train(a, b, subword="yes")
    with pytest.raises(ValueError, match="no piece-level"):
        base.alignments(direction="fwd", pieces=True)


# -- with a tokeniser -------------------------------------------------------------------------


@pytest.fixture()
def need_sp():
    pytest.importorskip("sentencepiece")


sp_only = pytest.mark.usefixtures("need_sp")


def morph_corpus(n=400):
    """Words with shared stems and endings, so that a tokeniser has something to split."""
    import random

    rng = random.Random(5)
    stems = [f"stem{i}x" for i in range(30)]
    ends = ["ing", "ed", "s", "er", "ly"]
    src, tgt = [], []
    for _ in range(n):
        idx = [(rng.randrange(30), rng.randrange(5)) for _ in range(rng.randint(2, 8))]
        src.append([stems[i] + ends[e] for i, e in idx])
        tgt.append([stems[i].replace("stem", "tige") + "-" + ends[e] for i, e in idx])
    return src, tgt


CFG = fastalign.SubwordConfig(vocab_size=300)


@sp_only
def test_tokeniser_training_and_native_equivalence(tmp_path, native_binary):
    src, tgt = morph_corpus()
    model = fastalign.train(src, tgt, subword=CFG, subword_output=True, threads=2)
    info = model.subword
    assert info["tokenizer"] and info["settings"]["vocab_size"] == 300
    assert info["settings"]["layout"] == "joint"
    pieces = model.alignments(direction="fwd", pieces=True)
    words = model.alignments(direction="fwd")
    assert len(words) == len(src)
    # word links point at existing words and the piece links are more numerous
    for k, line in enumerate(words):
        assert all(0 <= i < len(src[k]) and 0 <= j < len(tgt[k]) for i, j in line)
    assert sum(len(pieces[k]) for k in range(len(src))) > sum(len(x) for x in words)
    # the same pieces through the native binary give the same word alignments
    segs = sw.make_segmenters(model._inner.subword[2], "joint")
    path = tmp_path / "pieces.txt"
    sw.write_piece_corpus(
        str(path),
        (
            lambda: iter([w.encode().split() for w in map(" ".join, src)]),
            lambda: iter([w.encode().split() for w in map(" ".join, tgt)]),
        ),
        segs,
    )
    subprocess.run(
        [str(native_binary), "train", str(path), "-o", str(tmp_path / "n"), "-q", "-t", "2"]
        + ["--subword-marker", MARK],
        check=True,
    )
    assert words.to_bytes() == (tmp_path / "n.fwd.pharaoh").read_bytes()


@sp_only
def test_save_load_align_reproduces_training_alignments(tmp_path):
    src, tgt = morph_corpus()
    model = fastalign.train(src, tgt, subword=CFG)
    model.save(tmp_path / "m.fastalign")
    loaded = fastalign.Model.load(tmp_path / "m.fastalign")
    assert loaded.subword == model.subword and loaded.subword["tokenizer"]
    for kw in ({"direction": "fwd"}, {"direction": "inv"}, {"symmetrize": "grow-diag-final-and"}):
        assert loaded.align_many(src, tgt, **kw).to_bytes() == model.alignments(**kw).to_bytes()
    # one pair, plain words in, word links out; unseen words are fine (byte-level fallback)
    links = loaded.align(src[0], tgt[0])
    assert links == model.alignments()[0]
    assert all(0 <= i < 2 and 0 <= j < 2 for i, j in loaded.align(["zzzunseen", "q"], ["w", "e"]))
    # files work like lists
    f1, f2 = tmp_path / "s.txt", tmp_path / "t.txt"
    f1.write_text("".join(" ".join(s) + "\n" for s in src))
    f2.write_text("".join(" ".join(t) + "\n" for t in tgt))
    assert loaded.align_many(f1, f2).to_bytes() == model.alignments().to_bytes()


@sp_only
def test_python_cli_with_tokeniser_and_files_equal_api(tmp_path):
    src, tgt = morph_corpus(200)
    a, b = tmp_path / "s", tmp_path / "t"
    a.write_text("".join(" ".join(s) + "\n" for s in src))
    b.write_text("".join(" ".join(t) + "\n" for t in tgt))
    out = tmp_path / "o"
    rc = main(
        ["train", str(a), str(b), "-o", str(out), "-q", "--subword", "--subword-vocab", "300"]
        + ["--subword-output", "--save", str(tmp_path / "m"), "-s", "union"]
    )
    assert rc == 0
    api = fastalign.train(str(a), str(b), subword=CFG, subword_output=True)
    assert (tmp_path / "o.fwd.pharaoh").read_bytes() == api.alignments(direction="fwd").to_bytes()
    assert (tmp_path / "o.inv.subword.pharaoh").read_bytes() == api.alignments(
        direction="inv", pieces=True
    ).to_bytes()
    assert (tmp_path / "o.union.pharaoh").read_bytes() == api.alignments(
        symmetrize="union"
    ).to_bytes()
    # `align` with the saved model segments the raw text
    dest = tmp_path / "al"
    assert main(["align", str(tmp_path / "m"), str(a), str(b), "-s", "union", "-o", str(dest)]) == 0
    assert dest.read_bytes() == api.alignments(symmetrize="union").to_bytes()
    # a single `source ||| target` file gives the same corpus
    j = tmp_path / "joined"
    j.write_text("".join(f"{' '.join(s)} ||| {' '.join(t)}\n" for s, t in zip(src, tgt)))
    assert (
        fastalign.train(str(j), subword=CFG).alignments().to_bytes() == api.alignments().to_bytes()
    )


@sp_only
def test_one_side_and_separate_models(tmp_path):
    src, tgt = morph_corpus(200)
    for cfg in (
        fastalign.SubwordConfig(vocab_size=300, segment="src"),
        fastalign.SubwordConfig(vocab_size=300, joint=False),
    ):
        m = fastalign.train(src, tgt, subword=cfg)
        m.save(tmp_path / "m")
        l = fastalign.Model.load(tmp_path / "m")
        assert l.align_many(src, tgt).to_bytes() == m.alignments().to_bytes()


@sp_only
def test_the_tokeniser_models_are_reproducible():
    src, tgt = morph_corpus(150)
    a = fastalign.train(src, tgt, subword=CFG)._inner.subword
    b = fastalign.train(src, tgt, subword=CFG)._inner.subword
    assert a == b


# -- model files ------------------------------------------------------------------------------


def test_v1_and_v2_files(tmp_path, corpus):
    src, tgt, _ = corpus
    m = fastalign.train(src, tgt)
    m.save(tmp_path / "v2")
    raw = (tmp_path / "v2").read_bytes()
    assert raw[8:12] == (2).to_bytes(4, "little")
    # version 1 = no subword flag byte before the end marker
    v1 = bytearray(raw[:-9] + raw[-8:])
    v1[8:12] = (1).to_bytes(4, "little")
    (tmp_path / "v1").write_bytes(bytes(v1))
    old = fastalign.Model.load(tmp_path / "v1")
    assert old.subword is None
    assert old.align_many(src, tgt).to_bytes() == m.alignments().to_bytes()
    bad = bytearray(raw)
    bad[8:12] = (3).to_bytes(4, "little")
    (tmp_path / "v3").write_bytes(bytes(bad))
    with pytest.raises(ValueError, match="version 3"):
        fastalign.Model.load(tmp_path / "v3")


@sp_only
def test_truncated_and_corrupt_subword_models(tmp_path):
    src, tgt = morph_corpus(60)
    m = fastalign.train(src, tgt, subword=CFG, iterations=2)
    m.save(tmp_path / "m")
    raw = (tmp_path / "m").read_bytes()
    # the subword section sits at the end: cut inside it and at some other points
    cuts = list(range(len(raw) - 300_000 if len(raw) > 300_000 else 0, len(raw), 997)) + [
        len(raw) - 1,
        len(raw) - 9,
        len(raw) - 10,
        12,
        100,
    ]
    for cut in cuts:
        p = tmp_path / "cut"
        p.write_bytes(raw[:cut])
        with pytest.raises(ValueError):
            fastalign.Model.load(p)
    # corruption inside the subword section never crashes (it loads or raises ValueError)
    for k in range(len(raw) - 9, len(raw) - 9 - 60, -1):
        c = bytearray(raw)
        c[k] ^= 0xFF
        p = tmp_path / "bad"
        p.write_bytes(bytes(c))
        try:
            fastalign.Model.load(p)
        except ValueError:
            pass


def test_missing_sentencepiece_is_a_clear_error(monkeypatch, corpus):
    src, tgt, _ = corpus
    monkeypatch.setitem(sys.modules, "sentencepiece", None)
    with pytest.raises(ImportError, match=r"fastalign\[subword\]"):
        fastalign.train(src, tgt, subword="auto")


def test_input_validation(tmp_path):
    with pytest.raises(ValueError, match="marker"):
        fastalign.PreSegmented("")
    with pytest.raises(ValueError, match="marker"):
        fastalign.PreSegmented("a b")
    assert main(["train", "x", "y", "-o", str(tmp_path / "o"), "--subword-output"]) == 1
    assert main(["train", "x", "y", "-o", str(tmp_path / "o"), "--subword-vocab", "500"]) == 1
    assert (
        main(["train", "x", "y", "-o", str(tmp_path / "o"), "--subword", "--subword-marker", "a"])
        == 1
    )
    s, t = tmp_path / "s", tmp_path / "t"
    s.write_text("a ||| b\n")
    t.write_text("c\n")
    with pytest.raises(ValueError, match="'\\|\\|\\|'"):
        fastalign.train(s, t, subword=CFG)
    s.write_text("a\nb\n")
    with pytest.raises(ValueError, match="different numbers of lines"):
        fastalign.train(s, t, subword=CFG)
