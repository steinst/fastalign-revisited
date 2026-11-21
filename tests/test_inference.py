"""Saved-model inference (Step 24): the direct one-pair path equals ``align_many``, the word
index is built once and shared by concurrent readers, and subword inference memory is bounded
(tokeniser fetched once, per-call caches, files streamed through piece files)."""

from __future__ import annotations

import random
import threading

import fastalign
import pytest
from conftest import make_corpus
from fastalign import subword as sw
from textio import write_text

METHODS = ["intersect", "union", "grow-diag", "grow-diag-final", "grow-diag-final-and"]
SELECTIONS = [{"direction": "fwd"}, {"direction": "inv"}, {}] + [{"symmetrize": m} for m in METHODS]


def queries(src, tgt, seed=11):
    """Training pairs, pairs with unknown words, empty sides, token lists and odd words."""
    rng = random.Random(seed)
    qs = [(s, t) for s, t in zip(src[:40], tgt[:40])]
    for s, t in zip(src[40:60], tgt[40:60]):
        s, t = list(s), list(t)
        s[rng.randrange(len(s))] = "unknownword"
        t.insert(rng.randrange(len(t) + 1), "zzz")
        qs.append((s, t))
    qs += [
        ([], ["t1", "t2"]),
        (["s1"], []),
        ("", ""),
        ("s1 s2\ts3", "t3 t2  t1\n"),
        (["<eps>", "s4"], ["t4", "<eps>"]),
        (["s5"] * 30, ["t5"] * 2),
    ]
    return qs


def check_same(model, qs, sel):
    batch = model.align_many([q[0] for q in qs], [q[1] for q in qs], **sel)
    for k, (s, t) in enumerate(qs):
        assert model.align(s, t, **sel) == batch[k], (sel, k, s, t)


@pytest.fixture(scope="module")
def word_model(tmp_path_factory):
    src, tgt, _ = make_corpus(n=300, seed=4)
    m = fastalign.train(src, tgt, iterations=3)
    path = tmp_path_factory.mktemp("inf") / "both.model"
    m.save(path)
    return m, fastalign.Model.load(path), src, tgt


@pytest.mark.parametrize("sel", SELECTIONS, ids=lambda s: str(s))
def test_one_pair_equals_align_many(word_model, sel):
    trained, loaded, src, tgt = word_model
    qs = queries(src, tgt)
    for model in (trained, loaded):
        check_same(model, qs, sel)


def test_one_pair_reproduces_training(word_model):
    trained, loaded, src, tgt = word_model
    for sel in ({"direction": "fwd"}, {"direction": "inv"}, {"symmetrize": "grow-diag-final-and"}):
        want = trained.alignments(**sel)
        for k in range(len(src)):
            assert loaded.align(src[k], tgt[k], **sel) == want[k]


def test_one_direction_and_iteration_variants(tmp_path):
    src, tgt, _ = make_corpus(n=120, seed=8)
    qs = queries(src, tgt)
    for kw in (
        {"directions": "fwd"},
        {"directions": "inv"},
        {"iterations": 1},
        {"null": False},
        {"diagonal": False},
    ):
        m = fastalign.train(src, tgt, **kw)
        m.save(tmp_path / "m")
        loaded = fastalign.Model.load(tmp_path / "m")
        check_same(loaded, qs, {})


def test_one_pair_errors_match(word_model):
    _, loaded, _, _ = word_model
    with pytest.raises(ValueError, match="either symmetrize or direction"):
        loaded.align("a", "b", symmetrize="union", direction="fwd")
    with pytest.raises(ValueError):
        loaded.align("a", "b", symmetrize="invert")
    with pytest.raises(ValueError, match="'|||'"):
        loaded.align(["a", "|||"], "b")
    with pytest.raises(ValueError, match="whitespace"):
        loaded.align(["a b"], "b")


def test_simultaneous_readers(tmp_path):
    # many threads use one freshly loaded model at once (the first ones race to build the word
    # index); every result equals the sequential one
    src, tgt, _ = make_corpus(n=300, seed=4)
    fastalign.train(src, tgt).save(tmp_path / "m")
    qs = queries(src, tgt)
    sel = {"symmetrize": "grow-diag-final-and"}
    want = (
        fastalign.Model.load(tmp_path / "m")
        .align_many([q[0] for q in qs], [q[1] for q in qs], **sel)
        .to_list()
    )
    for _ in range(3):
        model = fastalign.Model.load(tmp_path / "m")
        errors, barrier = [], threading.Barrier(8)

        def reader(k, model=model, barrier=barrier, errors=errors):
            try:
                barrier.wait()
                for _ in range(3):
                    if k % 2:
                        got = [model.align(s, t, **sel) for s, t in qs]
                    else:
                        got = model.align_many(
                            [q[0] for q in qs], [q[1] for q in qs], threads=2, **sel
                        ).to_list()
                    assert got == want
            except BaseException as e:  # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors[0]


# -- subword models -----------------------------------------------------------------------------


def morph(n=300, seed=5):
    rng = random.Random(seed)
    stems = [f"stem{i}x" for i in range(25)]
    ends = ["", "ing", "ed", "s", "ation"]
    src, tgt = [], []
    for _ in range(n):
        ws = [rng.choice(stems) + rng.choice(ends) for _ in range(rng.randint(2, 7))]
        src.append(ws)
        tgt.append(["T" + w for w in reversed(ws)] if rng.random() < 0.3 else ["T" + w for w in ws])
    return src, tgt


@pytest.fixture(scope="module", params=["joint", "separate", "one-sided"])
def subword_model(request, tmp_path_factory):
    pytest.importorskip("sentencepiece")
    cfg = {
        "joint": fastalign.SubwordConfig(vocab_size=300),
        "separate": fastalign.SubwordConfig(vocab_size=300, joint=False),
        "one-sided": fastalign.SubwordConfig(vocab_size=300, segment="tgt"),
    }[request.param]
    src, tgt = morph()
    m = fastalign.train(src, tgt, subword=cfg)
    path = tmp_path_factory.mktemp("sw") / "m.model"
    m.save(path)
    return m, path, src, tgt


@pytest.mark.parametrize("sel", SELECTIONS, ids=lambda s: str(s))
def test_subword_one_pair_equals_align_many(subword_model, sel):
    trained, path, src, tgt = subword_model
    loaded = fastalign.Model.load(path)
    qs = queries(src, tgt) + [(["stem3xation", "newwordment"], ["Tstem3xation", "Tnewword"])]
    check_same(loaded, qs, sel)
    if sel:
        want = trained.alignments(**sel)
        for k in range(len(src)):
            assert loaded.align(src[k], tgt[k], **sel) == want[k]


def test_presegmented_one_pair_equals_align_many():
    src, tgt, _ = make_corpus(n=200, seed=6)

    def cut(line):
        return [
            p
            for w in line
            for p in ((sw.MARKER + w[:-1], w[-1]) if len(w) > 2 else (sw.MARKER + w,))
        ]

    psrc, ptgt = [cut(s) for s in src], [cut(t) for t in tgt]
    m = fastalign.train(psrc, ptgt, subword=fastalign.PreSegmented(sw.MARKER))
    for sel in SELECTIONS:
        check_same(m, list(zip(psrc[:50], ptgt[:50])) + [([], ["▁t1"])], sel)


def test_subword_files_stream_through_piece_files(subword_model, tmp_path):
    _, path, src, tgt = subword_model
    loaded = fastalign.Model.load(path)
    a, b, j = tmp_path / "a", tmp_path / "b", tmp_path / "j"
    write_text(a, "".join(" ".join(s) + "\n" for s in src))
    write_text(b, "".join(" ".join(t) + "\n" for t in tgt))
    write_text(j, "".join(" ".join(s) + " ||| " + " ".join(t) + "\n" for s, t in zip(src, tgt)))
    for sel in ({"direction": "fwd"}, {"direction": "inv"}, {}):
        want = loaded.align_many(src, tgt, **sel).to_bytes()
        assert loaded.align_many(a, b, **sel).to_bytes() == want
        assert loaded.align_many(j, **sel).to_bytes() == want
        assert loaded.align_many(a, tgt, **sel).to_bytes() == want
    scored = loaded.align_many(a, b, direction="fwd", scores=True).to_bytes()
    assert scored == loaded.align_many(src, tgt, direction="fwd", scores=True).to_bytes()


class _CountingInner:
    """Wraps the core model and counts how often the tokeniser blobs are fetched."""

    def __init__(self, inner):
        self._inner = inner
        self.fetches = 0

    @property
    def subword(self):
        self.fetches += 1
        return self._inner.subword

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_tokeniser_fetched_once(subword_model):
    _, path, src, tgt = subword_model
    loaded = fastalign.Model.load(path)
    counting = _CountingInner(loaded._inner)
    loaded._inner = counting
    for _ in range(3):
        loaded.align(src[0], tgt[0])
        loaded.align_many(src[:5], tgt[:5])
    assert counting.fetches == 1
    # a word-level model never fetches anything
    m = fastalign.train(*make_corpus(n=30)[:2])
    counting = _CountingInner(m._inner)
    m._inner = counting
    m.align("s1", "t1")
    m.align_many([["s1"]], [["t1"]])
    assert counting.fetches == 0


def test_repeated_high_cardinality_batches_stay_bounded(subword_model, monkeypatch):
    _, path, _, _ = subword_model
    loaded = fastalign.Model.load(path)
    fresh = fastalign.Model.load(path)
    monkeypatch.setattr(sw, "INFERENCE_CACHE_WORDS", 50)
    monkeypatch.setattr(sw, "MODEL_CACHE_WORDS", 40)
    # patching after loading applies: the stored segmenters are built on first use
    for b in range(6):
        ws = [f"novel{b}w{i}ing" for i in range(400)]
        qs = [ws[i : i + 4] for i in range(0, 400, 4)]
        ts = [["T" + w for w in q] for q in qs]
        out = loaded.align_many(qs, ts).to_list()
        singles = [loaded.align(q, t) for q, t in zip(qs, ts)]
        assert out == singles
        stored = loaded._segmenters
        sizes = {id(s): len(s._cache) for s in stored if hasattr(s, "_cache")}
        assert sizes and all(n <= 40 for n in sizes.values()), sizes
    # bounded caches never change a result: compare with a model whose caches are unbounded
    monkeypatch.setattr(sw, "INFERENCE_CACHE_WORDS", None)
    monkeypatch.setattr(sw, "MODEL_CACHE_WORDS", None)
    qs = [[f"novel9w{i}ing", "stem1xed"] for i in range(200)]
    ts = [["Tstem1xed", f"Tnovel9w{i}ing"] for i in range(200)]
    assert loaded.align_many(qs, ts).to_list() == fresh.align_many(qs, ts).to_list()
    assert [loaded.align(q, t) for q, t in zip(qs, ts)] == fresh.align_many(qs, ts).to_list()


def test_bounded_segmenter_cache(subword_model):
    _, path, _, _ = subword_model
    model = fastalign.Model.load(path)
    seg = model._text_segmenters()[1]
    if not isinstance(seg, sw.SentencePieceSegmenter):
        pytest.skip("this side is not segmented")
    bounded, unbounded = seg.scoped(25), seg.scoped(None)
    assert bounded._sp is seg._sp  # the tokeniser is shared, not rebuilt
    words = [f"word{i}ation".encode() for i in range(300)] * 2
    for w in words:
        assert bounded.encode_word(w) == unbounded.encode_word(w)
        assert len(bounded._cache) <= 25
    assert len(unbounded._cache) == 300
    assert len(seg._cache) <= sw.MODEL_CACHE_WORDS


def test_subword_arguments_checked_before_encoding(subword_model, monkeypatch):
    _, path, src, tgt = subword_model
    loaded = fastalign.Model.load(path)
    calls = []
    monkeypatch.setattr(sw, "write_piece_sides", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="threads"):
        loaded.align_many(src, tgt, threads=-1)
    with pytest.raises(ValueError, match="threads"):
        loaded.align_many(__file__, __file__, threads=-1)
    assert calls == []
