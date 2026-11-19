from __future__ import annotations

import logging
import subprocess

import fastalign
import pytest


def lines(path):
    return path.read_text().split("\n")[:-1]


# -- training: inputs -----------------------------------------------------------------------


def test_train_sources_agree(model, corpus, corpus_files, joined_file):
    """Token lists, strings, two files and one `|||` file give the same alignments."""
    src, tgt, _ = corpus
    ref = model.alignments().to_bytes()
    strs = fastalign.train([" ".join(s) for s in src], [" ".join(t) + "\n" for t in tgt], threads=1)
    assert strs.alignments().to_bytes() == ref
    a, b = corpus_files
    assert fastalign.train(a, b, threads=3).alignments().to_bytes() == ref
    assert fastalign.train(str(a), str(b)).alignments().to_bytes() == ref
    assert fastalign.train(joined_file).alignments().to_bytes() == ref
    # a file and a list mix
    assert fastalign.train(a, tgt).alignments().to_bytes() == ref


def test_thread_count_does_not_change_output(corpus):
    src, tgt, _ = corpus
    outs = {
        fastalign.train(src, tgt, threads=t).alignments(direction="fwd").to_bytes()
        for t in (1, 2, 5)
    }
    assert len(outs) == 1


def test_lookup_index_does_not_change_output(corpus):
    src, tgt, _ = corpus
    outs = {
        fastalign.train(src, tgt, threads=2, lookup_index=li)
        .alignments(symmetrize="grow-diag-final-and")
        .to_bytes()
        for li in (True, False)
    }
    assert len(outs) == 1


def test_shapes(model, corpus):
    src, tgt, _ = corpus
    assert model.directions == ("fwd", "inv")
    assert model.has_training_alignments and model.n_lines == len(src)
    assert model.vocab_size == 80
    al = model.alignments()
    assert len(al) == len(src)
    for k in (0, 5, -1):
        links = al[k]
        assert all(0 <= i < len(src[k]) and 0 <= j < len(tgt[k]) for i, j in links)
    assert al[3:6] == [al[3], al[4], al[5]]
    assert list(al)[:3] == al[:3]
    assert al.pharaoh(0) == " ".join(f"{i}-{j}" for i, j in al[0])
    with pytest.raises(IndexError):
        al[len(al)]
    p = model.parameters("fwd")
    assert p["iterations"] == 5 and p["tension"] > 0 and p["p0"] == 0.08
    assert "ITERATION 1" in model.training_log("fwd")
    assert "fastalign.Model" in repr(model)


def test_alignment_quality(model, corpus):
    """The synthetic corpus is nearly monotone: most links are the gold ones."""
    perms = corpus[2]
    hits = total = 0
    for k, links in enumerate(model.alignments(direction="fwd")):
        gold = {(perms[k][j], j) for j in range(len(perms[k]))}
        hits += len(gold & set(links))
        total += len(gold)
    assert hits / total > 0.8


def test_directions_and_symmetrize(corpus):
    src, tgt, _ = corpus
    both = fastalign.train(src, tgt, threads=2)
    fwd = fastalign.train(src, tgt, directions="fwd", threads=2)
    inv = fastalign.train(src, tgt, directions="inv", threads=2)
    assert fwd.directions == ("fwd",) and inv.directions == ("inv",)
    assert fwd.alignments().to_bytes() == both.alignments(direction="fwd").to_bytes()
    assert inv.alignments().to_bytes() == both.alignments(direction="inv").to_bytes()
    with pytest.raises(ValueError, match="both directions"):
        fwd.alignments(symmetrize="union")
    with pytest.raises(ValueError, match="no 'inv'"):
        fwd.alignments(direction="inv")
    with pytest.raises(ValueError, match="not both"):
        both.alignments(symmetrize="union", direction="fwd")
    f, i = both.alignments(direction="fwd"), both.alignments(direction="inv")
    for m in fastalign.METHODS:
        s = both.alignments(symmetrize=m)
        for k in (0, 1, 17):
            assert s[k] == fastalign.symmetrize(f[k], i[k], m)
    assert both.alignments(symmetrize="intersection").to_bytes() == (
        both.alignments(symmetrize="intersect").to_bytes()
    )
    default = both.alignments()
    assert default.to_bytes() == both.alignments(symmetrize="grow-diag-final-and").to_bytes()


def test_options_change_the_model(corpus):
    src, tgt, _ = corpus
    # The synthetic corpus is easy enough that alignments coincide; the statistics differ.
    base = fastalign.train(src, tgt, directions="fwd").training_log("fwd")
    seen = {base}
    for kw in (
        {"diagonal": False},
        {"optimize_tension": False},
        {"variational_bayes": False},
        {"iterations": 2},
        {"null": False},
        {"tension": 2.0, "p0": 0.3},
    ):
        seen.add(fastalign.train(src, tgt, directions="fwd", **kw).training_log("fwd"))
    assert len(seen) == 7
    # -b only groups sums; it must never be an error
    fastalign.train(src, tgt, directions="fwd", buffer_size=7)


def test_scores(corpus, tmp_path):
    src, tgt, _ = corpus
    m = fastalign.train(src, tgt, directions="fwd", scores=True)
    m.write_alignments(tmp_path / "s.out", direction="fwd")
    text = lines(tmp_path / "s.out")
    assert len(text) == len(src) and all(" ||| " in t for t in text)
    plain = fastalign.train(src, tgt, directions="fwd")
    assert m.alignments().to_list() == plain.alignments().to_list()


def test_max_len_keeps_line_correspondence(corpus):
    src, tgt, _ = corpus
    m = fastalign.train(src, tgt, directions="fwd", max_len=5)
    al = m.alignments()
    assert len(al) == len(src)
    for s, t, links in zip(src, tgt, al):
        if len(s) > 5 or len(t) > 5:
            assert links == []
    assert any(links for links in al)


def test_empty_sides_give_empty_alignments(corpus):
    src, tgt, _ = corpus
    s2, t2 = [list(x) for x in src[:50]], [list(x) for x in tgt[:50]]
    s2[3] = []
    t2[7] = []
    m = fastalign.train(s2, t2)
    al = m.alignments(direction="fwd")
    assert len(al) == 50 and al[3] == [] and al[7] == []
    assert "Error in line" in m.training_log("fwd")


def test_progress_callback_and_logging(corpus, caplog):
    src, tgt, _ = corpus
    events = []
    with caplog.at_level(logging.INFO, logger="fastalign"):
        fastalign.train(src, tgt, threads=2, callback=events.append, iterations=3)
    assert sorted((e.direction, e.iteration) for e in events) == [
        (d, i) for d in ("fwd", "inv") for i in (1, 2, 3)
    ]
    e = next(e for e in events if e.direction == "fwd" and e.iteration == 2)
    assert e.iterations == 3 and e.perplexity > 1 and e.tension > 0 and e.seconds >= 0
    assert e.log_likelihood < 0
    assert "iteration 2/3" in str(e)
    assert sum("iteration" in r.getMessage() for r in caplog.records) == 6


def test_callback_exception_stops_training(corpus):
    src, tgt, _ = corpus

    class Boom(Exception):
        pass

    def cb(_):
        raise Boom

    with pytest.raises(Boom):
        fastalign.train(src, tgt, callback=cb)
    # the library is not poisoned
    assert len(fastalign.train(src[:30], tgt[:30]).alignments()) == 30


# -- training: bad input --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        ({"directions": "x"}, ValueError, "directions"),
        ({"iterations": -1}, ValueError, "iterations"),
        ({"iterations": True}, ValueError, "iterations"),
        ({"threads": -1}, ValueError, "threads"),
        ({"buffer_size": -1}, ValueError, "buffer_size"),
        ({"max_len": 0}, ValueError, "max_len"),
        ({"alpha": 0.0}, ValueError, "alpha"),
        ({"p0": 1.0}, ValueError, "p0"),
        ({"p0": float("nan")}, ValueError, "p0"),
        ({"diagonal": False, "tension": 2.0}, ValueError, "tension"),
        ({"callback": 3}, TypeError, "callback"),
    ],
)
def test_train_validation(corpus, kwargs, exc, match):
    src, tgt, _ = corpus
    with pytest.raises(exc, match=match):
        fastalign.train(src[:10], tgt[:10], **kwargs)


def test_train_input_errors(corpus, tmp_path):
    src, tgt, _ = corpus
    with pytest.raises(ValueError, match="sentences but|different numbers"):
        fastalign.train(src[:10], tgt[:9])
    with pytest.raises(ValueError, match="no sentence pairs"):
        fastalign.train([], [])
    with pytest.raises(TypeError):
        fastalign.train(src[:3])  # a single list is not a joined file
    with pytest.raises(FileNotFoundError):
        fastalign.train(tmp_path / "nope.src", tmp_path / "nope.tgt")
    with pytest.raises(FileNotFoundError):
        fastalign.train(tmp_path / "nope.txt")
    with pytest.raises(TypeError, match="bytes"):
        fastalign.train([b"a b"], [b"x"])
    with pytest.raises(TypeError, match="tokens must be str"):
        fastalign.train([["a", 1]], [["x"]])
    with pytest.raises(ValueError, match="whitespace"):
        fastalign.train([["a b"]], [["x"]])
    with pytest.raises(ValueError, match="whitespace"):
        fastalign.train([[""]], [["x"]])
    with pytest.raises(ValueError, match=r"\|\|\|"):
        fastalign.train([["a", "|||"]], [["x"]])
    with pytest.raises(TypeError):
        fastalign.train(5, 6)


def test_string_sentences_are_split_on_ascii_whitespace():
    m = fastalign.train(["a  b\tc\n", "a b"], ["x y z", "x\ny"], directions="fwd")
    assert m.n_lines == 2
    assert len(m.alignments()[0]) == 3


def test_non_ascii_tokens(tmp_path):
    src = [["maison", "grüne", "日本語"], ["ça", "va"]] * 20
    tgt = [["house", "green", "Japanese"], ["it", "goes"]] * 20
    m = fastalign.train(src, tgt, threads=1)
    assert m.vocab_size == 10
    path = tmp_path / "m.fastalign"
    m.save(path)
    loaded = fastalign.Model.load(path)
    assert loaded.align(["日本語"], ["Japanese"]) == [(0, 0)]


# -- save / load / forced alignment ----------------------------------------------------------


def test_save_load_roundtrip(model, corpus, tmp_path):
    src, tgt, _ = corpus
    path = tmp_path / "m.fastalign"
    model.save(path)
    assert not (tmp_path / "m.fastalign.tmp").exists()
    loaded = fastalign.Model.load(path)
    assert loaded.directions == model.directions and loaded.vocab_size == model.vocab_size
    assert not loaded.has_training_alignments and loaded.n_lines is None
    with pytest.raises(ValueError, match="loaded from a file"):
        loaded.alignments()
    with pytest.raises(ValueError, match="loaded from a file"):
        loaded.write_alignments(tmp_path / "x")
    assert loaded.parameters("inv") == model.parameters("inv")
    # forced alignment of the training corpus reproduces the training alignments
    for kw in ({"direction": "fwd"}, {"direction": "inv"}, {"symmetrize": "grow-diag-final-and"}):
        assert loaded.align_many(src, tgt, **kw).to_bytes() == model.alignments(**kw).to_bytes()
        assert loaded.align_many(src, tgt, threads=1, **kw).to_bytes() == (
            model.alignments(**kw).to_bytes()
        )


def test_align_one_pair(model, corpus):
    src, tgt, _ = corpus
    expected = model.alignments()
    for k in (0, 1, 2, 99):
        assert model.align(src[k], tgt[k]) == expected[k]
        assert model.align(" ".join(src[k]), " ".join(tgt[k])) == expected[k]
    assert model.align(["s1"], ["t1"], direction="fwd") == [(0, 0)]
    assert model.align([], ["t1"]) == []
    # unseen words are handled
    links = model.align(["zzz", "s1"], ["t1", "qqq"], direction="fwd")
    assert all(0 <= i < 2 and 0 <= j < 2 for i, j in links)


def test_align_many_inputs(model, corpus, corpus_files, joined_file):
    src, tgt, _ = corpus
    a, b = corpus_files
    ref = model.align_many(src, tgt, direction="fwd").to_bytes()
    assert model.align_many(a, b, direction="fwd").to_bytes() == ref
    assert model.align_many(joined_file, direction="fwd").to_bytes() == ref
    with pytest.raises(ValueError, match="different numbers"):
        model.align_many(src[:5], tgt[:4])
    with pytest.raises(ValueError, match="scores"):
        model.align_many(src, tgt, symmetrize="union", scores=True)
    scored = model.align_many(src[:5], tgt[:5], direction="fwd", scores=True).to_bytes()
    assert all(b" ||| " in ln for ln in scored.split(b"\n")[:-1])


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        fastalign.Model.load(tmp_path / "missing")
    bad = tmp_path / "bad"
    bad.write_bytes(b"not a model at all")
    with pytest.raises(ValueError, match="bad"):
        fastalign.Model.load(bad)


def test_truncated_model_is_a_clean_error(model, tmp_path):
    path = tmp_path / "m.fastalign"
    model.save(path)
    data = path.read_bytes()
    for cut in (0, 7, 100, len(data) // 2, len(data) - 1):
        p = tmp_path / f"t{cut}"
        p.write_bytes(data[:cut])
        with pytest.raises(ValueError):
            fastalign.Model.load(p)


def test_write_table_and_align_with_table(corpus, tmp_path):
    src, tgt, _ = corpus
    m = fastalign.train(src, tgt, directions="fwd")
    table = tmp_path / "t"
    m.write_table(table)
    assert table.read_text().count("\n") > 30
    p = m.parameters("fwd")
    res = fastalign.align_with_table(
        table, src[:20], tgt[:20], tension=p["tension"], mean_length_multiplier=1.0
    )
    assert res.failed_line is None and "TOTAL LOG PROB" in res.log
    assert res.log.startswith("Loaded ")
    out = res.output.decode().split("\n")[:-1]
    assert len(out) == 20
    assert out[0].startswith(" ".join(src[0]) + " ||| " + " ".join(tgt[0]) + " ||| ")
    # an empty side stops fast_align at that line
    res = fastalign.align_with_table(table, [["s1"], []], [["t1"], ["t1"]])
    assert res.failed_line == 2 and "Error in line 2" in res.log
    with pytest.raises(FileNotFoundError):
        fastalign.align_with_table(tmp_path / "nope", src[:2], tgt[:2])
    with pytest.raises(ValueError, match="no 'inv'"):
        m.write_table(table, "inv")


# -- symmetrisation --------------------------------------------------------------------------


def test_symmetrize_functions():
    fwd = [(0, 0), (1, 1), (2, 1)]
    inv = [(0, 0), (1, 1), (1, 2)]
    assert fastalign.symmetrize(fwd, inv, "intersect") == [(0, 0), (1, 1)]
    assert fastalign.symmetrize(fwd, inv, "intersection") == [(0, 0), (1, 1)]
    assert fastalign.symmetrize(fwd, inv, "union") == [(0, 0), (1, 1), (1, 2), (2, 1)]
    assert fastalign.symmetrize_many([fwd, []], [inv, []], "grow-diag")[1] == []
    assert fastalign.invert(fwd) == [(0, 0), (1, 1), (1, 2)]
    assert fastalign.parse_pharaoh("0-0 1-2") == [(0, 0), (1, 2)]
    assert fastalign.parse_pharaoh("0-0 ||| -3.5") == [(0, 0)]
    assert fastalign.parse_pharaoh("") == []
    with pytest.raises(ValueError):
        fastalign.parse_pharaoh("0:1")
    with pytest.raises(ValueError, match="unknown"):
        fastalign.symmetrize(fwd, inv, "nope")
    with pytest.raises(ValueError, match="invert"):
        fastalign.symmetrize(fwd, inv, "invert")
    with pytest.raises(ValueError, match="same size"):
        fastalign.symmetrize_many([fwd], [], "union")


def test_symmetrize_matches_atools(model, tmp_path, atools_binary):
    """Every method, on the model's own directional alignments, equals `atools -c METHOD`."""
    model.write_alignments(tmp_path / "f", direction="fwd")
    model.write_alignments(tmp_path / "i", direction="inv")
    for m in fastalign.METHODS:
        want = subprocess.run(
            [str(atools_binary), "-c", m, "-i", str(tmp_path / "f"), "-j", str(tmp_path / "i")],
            capture_output=True,
            check=True,
        ).stdout
        assert model.alignments(symmetrize=m).to_bytes() == want, m


# -- exactness against fast_align ------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, flags",
    [
        ({}, ["-d", "-o", "-v"]),
        ({"diagonal": False, "optimize_tension": False, "variational_bayes": False}, []),
        ({"optimize_tension": False}, ["-d", "-v"]),
        ({"null": False}, ["-d", "-o", "-v", "-N"]),
        ({"iterations": 3, "buffer_size": 50}, ["-d", "-o", "-v", "-I", "3", "-b", "50"]),
        (
            {"tension": 2.0, "p0": 0.2, "alpha": 0.1},
            ["-d", "-o", "-v", "-T", "2", "-q", "0.2", "-a", "0.1"],
        ),
    ],
)
@pytest.mark.parametrize("threads", [1, 3])
def test_identical_to_fast_align(joined_file, fast_align_binary, kwargs, flags, threads):
    """Both directions, byte for byte equal to `OMP_NUM_THREADS=1 fast_align`."""
    model = fastalign.train(joined_file, threads=threads, **kwargs)
    for d, extra in (("fwd", []), ("inv", ["-r"])):
        ref = subprocess.run(
            [str(fast_align_binary), "-i", str(joined_file), *extra, *flags],
            capture_output=True,
            check=True,
            env={"OMP_NUM_THREADS": "1"},
        ).stdout
        assert model.alignments(direction=d).to_bytes() == ref


def test_identical_to_native_binary(joined_file, native_binary, model, tmp_path):
    prefix = tmp_path / "n"
    subprocess.run(
        [str(native_binary), "train", str(joined_file), "-o", str(prefix), "-s", "union", "-q"],
        check=True,
    )
    py = fastalign.train(joined_file)
    assert (tmp_path / "n.fwd.pharaoh").read_bytes() == py.alignments(direction="fwd").to_bytes()
    assert (tmp_path / "n.inv.pharaoh").read_bytes() == py.alignments(direction="inv").to_bytes()
    assert (tmp_path / "n.union.pharaoh").read_bytes() == py.alignments(
        symmetrize="union"
    ).to_bytes()
