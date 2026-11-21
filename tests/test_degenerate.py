"""Degenerate training semantics (Step 22; code review #5, #10 and `tension=0` from #11).

Decision C: empty-sided pairs keep fast_align's behaviour by default, `empty_pairs` /
`--empty-pairs` skips them (positions kept) or rejects them (line numbers, sides, counts);
nonfinite trained parameters are reported as warnings outside the reference-format logs. "No
decoding iteration" (`-I 0`) is an explicit outcome (no lines at all) through filtering and
subword projection; an empty input and an input filtered to zero pairs both train (nothing) and
differ only in the number of empty output lines; both front ends behave the same.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import fastalign
import pytest
from textio import write_text

GOOD = "a b c ||| x y z\n" * 20
EXTRAS = {"empty-source": (" ||| x\n", "source side empty"), "blank": ("\n", "both sides empty")}
MARK = "▁"


def _pycli():
    return [sys.executable, "-m", "fastalign.cli"]


@pytest.fixture(scope="module", params=["native", "python"])
def cli(request, native_binary):
    return [str(native_binary)] if request.param == "native" else _pycli()


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, check=False, stdin=subprocess.DEVNULL)


def _fa(fast_align_binary, path, *flags):
    r = subprocess.run(
        [str(fast_align_binary), "-i", str(path), *flags],
        capture_output=True,
        env={"OMP_NUM_THREADS": "1"},
        check=True,
    )
    return r.stdout


def _train_quiet(*args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = fastalign.train(*args, **kwargs)
    return m, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


def _cli_train(cli, inp, prefix, *flags):
    return _run([*cli, "train", str(inp), "-o", str(prefix), "-t", "2", "-q", "--log", *flags])


# -- review #5: 20 good pairs + one empty-sided pair --------------------------------------------


@pytest.mark.parametrize("extra", list(EXTRAS))
@pytest.mark.parametrize("mode", ["keep", "skip", "error"])
def test_twenty_good_plus_one_empty_pair_api(tmp_path, extra, mode, fast_align_binary):
    line, side = EXTRAS[extra]
    path, clean = tmp_path / "c.txt", tmp_path / "clean.txt"
    write_text(path, GOOD + line)
    write_text(clean, GOOD)
    if mode == "error":
        with pytest.raises(
            ValueError, match=rf"1 line pair with an empty side.*line 21 \({side}\)"
        ):
            fastalign.train(path, threads=2, empty_pairs=mode)
        return
    m, warned = _train_quiet(path, threads=2, empty_pairs=mode)
    assert len(m.alignments(direction="fwd")) == 21
    model_file = tmp_path / "m.fastalign"
    m.save(model_file)
    loaded = fastalign.Model.load(model_file)
    for d, flags in (("fwd", ["-d", "-o", "-v"]), ("inv", ["-d", "-o", "-v", "-r"])):
        got = m.alignments(direction=d).to_bytes()
        if mode == "keep":
            # fast_align's own result, damage included (forward: nothing aligned for empty-source)
            assert got == _fa(fast_align_binary, path, *flags)
        else:
            assert got == _fa(fast_align_binary, clean, *flags) + b"\n"
        # through save and inference on the whole file (the empty pair gets an empty line)
        assert loaded.align_many(path, direction=d).to_bytes() == got
    linked = sum(1 for links in m.alignments(direction="fwd") if links)
    if mode == "skip":
        assert warned == [] and m.training_warnings == [] and linked == 20
        return
    assert warned and warned == m.training_warnings
    assert all(f"line 21 ({side})" in w and "'skip'" in w for w in warned), warned
    if extra == "empty-source":
        assert linked == 0  # the review's finding, reported now
        assert warned[0].startswith("fwd: training ended with nonfinite parameters (tension ")
        assert len(warned) == 1  # the inverse direction is unaffected
    else:
        assert linked == 20 and len(warned) == 2  # scores of both directions are NaN
        assert all("mean length ratio" in w for w in warned)


@pytest.mark.parametrize("extra", list(EXTRAS))
def test_twenty_good_plus_one_empty_pair_front_ends(tmp_path, cli, extra):
    line, side = EXTRAS[extra]
    path = tmp_path / "c.txt"
    write_text(path, GOOD + line)
    for mode in ("keep", "skip"):
        p = tmp_path / f"{mode}"
        r = _cli_train(
            cli,
            path,
            p,
            "--empty-pairs",
            mode,
            "-s",
            "grow-diag-final-and",
            "--save",
            str(tmp_path / f"{mode}.fastalign"),
        )
        assert r.returncode == 0, r.stderr.decode()
        m, _ = _train_quiet(path, threads=2, empty_pairs=mode)
        for d in ("fwd", "inv"):
            assert (tmp_path / f"{mode}.{d}.pharaoh").read_bytes() == m.alignments(
                direction=d
            ).to_bytes()
            # the reference-format log never carries the warning
            assert b"warning" not in (tmp_path / f"{mode}.{d}.log").read_bytes()
            # inference with the saved model through the same front end
            a = _run(
                [*cli, "align", str(tmp_path / f"{mode}.fastalign"), str(path), "--direction", d]
            )
            assert a.returncode == 0 and a.stdout == m.alignments(direction=d).to_bytes()
        gdfa = (tmp_path / f"{mode}.grow-diag-final-and.pharaoh").read_bytes()
        assert gdfa == m.alignments(symmetrize="grow-diag-final-and").to_bytes()
        warned = [x for x in r.stderr.decode().splitlines() if x.startswith("fastalign: warning: ")]
        # printed even with -q, never for skip
        assert (mode == "keep") == bool(warned), r.stderr
        assert all(f"line 21 ({side})" in w for w in warned)
    r = _cli_train(cli, path, tmp_path / "e", "--empty-pairs", "error")
    assert r.returncode == 1 and f"line 21 ({side})" in r.stderr.decode()
    assert not (tmp_path / "e.fwd.pharaoh").exists()


# -- review #10: iterations x exclusion x word/subword ------------------------------------------


def _pairs():
    from conftest import make_corpus

    src, tgt, _ = make_corpus(n=60, seed=5, min_len=2, max_len=9)
    return [" ".join(s) for s in src], [" ".join(t) for t in tgt]


def _segment(sentence: str) -> str:
    """`s12` -> `▁s 12`: every word is two pieces."""
    return " ".join(f"{MARK}{w[0]} {w[1:]}" for w in sentence.split())


@pytest.mark.parametrize("subword", [False, True], ids=["word", "subword"])
@pytest.mark.parametrize("iterations", [0, 1, 2])
@pytest.mark.parametrize("exclusion", ["none", "partial", "total"])
def test_iterations_by_exclusion(tmp_path, cli, subword, iterations, exclusion):
    src, tgt = _pairs()
    if subword:
        src, tgt = [_segment(s) for s in src], [_segment(t) for t in tgt]
    cap = {"none": None, "partial": 14 if subword else 7, "total": 1}[exclusion]
    n_tok = [max(len(a.split()), len(b.split())) for a, b in zip(src, tgt)]
    dropped = [k for k, n in enumerate(n_tok) if cap is not None and n > cap]
    assert (exclusion == "none") == (not dropped)
    assert (exclusion == "total") == (len(dropped) == len(src))
    kw = {"iterations": iterations, "threads": 2, "max_len": cap, "directions": "both"}
    if subword:
        kw["subword"] = fastalign.PreSegmented()
    m, warned = _train_quiet(src, tgt, **kw)
    assert bool(warned) == (exclusion == "total"), warned

    # what each direction must be
    if iterations == 0:
        want = {"fwd": b"", "inv": b"", "sym": b""}
    elif exclusion == "total":
        want = {k: b"\n" * len(src) for k in ("fwd", "inv", "sym")}
    else:
        keep = [k for k in range(len(src)) if k not in dropped]
        alone, _ = _train_quiet(
            [src[k] for k in keep], [tgt[k] for k in keep], **{**kw, "max_len": None}
        )
        want = {}
        for key, sel in (
            ("fwd", {"direction": "fwd"}),
            ("inv", {"direction": "inv"}),
            ("sym", {"symmetrize": "grow-diag-final-and"}),
        ):
            lines = iter(alone.alignments(**sel).to_bytes().splitlines(keepends=True))
            want[key] = b"".join(b"\n" if k in dropped else next(lines) for k in range(len(src)))
    assert m.alignments(direction="fwd").to_bytes() == want["fwd"]
    assert m.alignments(direction="inv").to_bytes() == want["inv"]
    assert m.alignments(symmetrize="grow-diag-final-and").to_bytes() == want["sym"]

    # both command lines agree with the API
    a, b = tmp_path / "c.src", tmp_path / "c.tgt"
    write_text(a, "".join(s + "\n" for s in src))
    write_text(b, "".join(t + "\n" for t in tgt))
    flags = ["-I", str(iterations), "-s", "grow-diag-final-and"]
    if cap is not None:
        flags += ["--max-len", str(cap)]
    if subword:
        flags += ["--subword-marker", MARK]
    r = _run([*cli, "train", str(a), str(b), "-o", str(tmp_path / "p"), "-q", "-t", "2", *flags])
    assert r.returncode == 0, r.stderr.decode()
    assert (b"fastalign: warning:" in r.stderr) == (exclusion == "total"), r.stderr
    assert (tmp_path / "p.fwd.pharaoh").read_bytes() == want["fwd"]
    assert (tmp_path / "p.inv.pharaoh").read_bytes() == want["inv"]
    assert (tmp_path / "p.grow-diag-final-and.pharaoh").read_bytes() == want["sym"]

    # a saved checkpoint reproduces the training alignments of the kept lines (also -I 1)
    if iterations >= 1 and exclusion != "total":
        keep = [k for k in range(len(src)) if k not in dropped]
        model_file = tmp_path / "m.fastalign"
        m.save(model_file)
        loaded = fastalign.Model.load(model_file)
        for d in ("fwd", "inv"):
            got = loaded.align_many([src[k] for k in keep], [tgt[k] for k in keep], direction=d)
            lines = want[d].splitlines(keepends=True)
            assert got.to_bytes() == b"".join(lines[k] for k in keep), d


def test_iterations_zero_with_filtering_matches_fast_align(tmp_path, fast_align_binary):
    """`-I 0` writes nothing, also when lines are filtered (the review's `max_len=9` case)."""
    path = tmp_path / "one.txt"
    write_text(path, "a b c ||| x y z\n")
    want = _fa(fast_align_binary, path, "-d", "-o", "-v", "-I", "0")
    assert want == b""
    for kw in ({}, {"max_len": 9}, {"subword": fastalign.PreSegmented()}, {"empty_pairs": "skip"}):
        m = fastalign.train(["a b c"], ["x y z"], threads=1, iterations=0, **kw)
        assert m.alignments(direction="fwd").to_bytes() == want, kw
        assert m.alignments().to_bytes() == want, kw


def test_one_iteration_checkpoint(tmp_path):
    """Review #10: `-I 1` training gives links; inference with the saved model gave none."""
    m = fastalign.train(["a b c"], ["x y z"], threads=1, iterations=1)
    assert m.alignments(direction="fwd").to_bytes() == b"0-0 1-1 2-2\n"
    m.save(tmp_path / "m")
    loaded = fastalign.Model.load(tmp_path / "m")
    for d in ("fwd", "inv"):
        assert loaded.align_many(["a b c"], ["x y z"], direction=d).to_bytes() == b"0-0 1-1 2-2\n"
    # the exported table stays fast_align's (never estimated: every value -inf)
    loaded.write_table(tmp_path / "t")
    lines = (tmp_path / "t").read_text().splitlines()
    assert lines and all(x.endswith("\t-inf") for x in lines)


# -- genuinely empty input vs nonempty input filtered to zero pairs ----------------------------


def test_empty_input_and_all_filtered_are_defined_once(tmp_path, cli):
    empty, long = tmp_path / "empty.txt", tmp_path / "long.txt"
    write_text(empty, "")
    write_text(long, "a b c d ||| x\ne f g h ||| y\n")
    cases = [(empty, [], b"", "no line pairs"), (long, ["--max-len", "2"], b"\n\n", "dropped")]
    for path, flags, want, message in cases:
        r = _run([*cli, "train", str(path), "--directions", "fwd", "-q", *flags])
        assert (r.returncode, r.stdout) == (0, want), r.stderr
        assert "fastalign: warning: " in r.stderr.decode() and message in r.stderr.decode()
        m, warned = _train_quiet(path, directions="fwd", threads=1, max_len=2 if flags else None)
        assert m.alignments(direction="fwd").to_bytes() == want
        assert len(warned) == 1 and message in warned[0]
    # skip that drops everything is the same outcome as max_len
    blank = tmp_path / "blank.txt"
    write_text(blank, "\n ||| x\n")
    r = _run([*cli, "train", str(blank), "--directions", "fwd", "-q", "--empty-pairs", "skip"])
    assert (r.returncode, r.stdout) == (0, b"\n\n") and b"dropped" in r.stderr


# -- tension 0 ----------------------------------------------------------------------------------


@pytest.mark.parametrize("tension", ["0", "1e-300", "-0.0", "nan"])
def test_singular_tension_is_rejected_before_reading(tmp_path, cli, tension):
    missing = tmp_path / "does-not-exist.txt"
    for args in (
        ["train", str(missing), "--directions", "fwd"],
        ["align", "--table", str(missing), str(missing)],
    ):
        r = _run([*cli, *args, "--tension", tension])
        assert r.returncode != 0 and b"tension" in r.stderr, r.stderr
        assert b"does-not-exist" not in r.stderr, r.stderr
    with pytest.raises(ValueError, match="tension"):
        fastalign.train(str(missing), threads=1, tension=float(tension))
    with pytest.raises(ValueError, match="tension"):
        fastalign.align_with_table(missing, str(missing), tension=float(tension))
    # without the diagonal prior the tension is not used (and must stay at its default)
    with pytest.raises(ValueError, match="tension"):
        fastalign.train(str(missing), diagonal=False, tension=float(tension))


def test_negative_tension_is_fast_aligns(tmp_path, fast_align_binary):
    path = tmp_path / "c.txt"
    write_text(path, GOOD + "a b ||| x y\n")
    want = _fa(fast_align_binary, path, "-d", "-T", "-1")
    m = fastalign.train(
        path,
        threads=1,
        directions="fwd",
        tension=-1.0,
        optimize_tension=False,
        variational_bayes=False,
    )
    assert m.alignments(direction="fwd").to_bytes() == want


def test_tokeniser_subword_with_empty_pairs():
    pytest.importorskip("sentencepiece")
    with pytest.raises(ValueError, match="no words"):
        fastalign.train([], [], subword="auto")
    cfg = fastalign.SubwordConfig(vocab_size=300)
    src, tgt = ["a b c"] * 20 + [""], ["x y z"] * 20 + ["x"]
    with pytest.raises(ValueError, match=r"line 21 \(source side empty\)"):
        fastalign.train(src, tgt, subword=cfg, empty_pairs="error", threads=1)
    m, warned = _train_quiet(src, tgt, subword=cfg, empty_pairs="skip", threads=1)
    fwd = m.alignments(direction="fwd")
    assert not warned and len(fwd) == 21 and fwd[20] == [] and all(fwd[k] for k in range(20))
    m, warned = _train_quiet(src, tgt, subword=cfg, threads=1)
    assert warned and "line 21 (source side empty)" in warned[0]
    assert not any(m.alignments(direction="fwd"))
