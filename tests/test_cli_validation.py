"""Step 23: invalid invocations fail before loading anything and leave the destination alone;
the documented meaning of ``scores`` is what the code computes."""

from __future__ import annotations

import math
import subprocess

import fastalign
import pytest
from fastalign.cli import main
from textio import read_text, write_text


def _native(binary, *args):
    return subprocess.run([str(binary), *args], capture_output=True, text=True, check=False)


@pytest.fixture()
def saved(tmp_path, joined_file, native_binary):
    m = tmp_path / "both.model"
    r = _native(
        native_binary, "train", str(joined_file), "-o", str(tmp_path / "t"), "-q", "--save", str(m)
    )
    assert r.returncode == 0, r.stderr
    return m


def _missing_model_cases(tmp_path):
    # a model path that does not exist: if validation ran after loading, the error would be
    # "No such file"; the conflict message must win
    return str(tmp_path / "no-such.model")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["-s", "union", "--scores"], "--scores belongs to one direction"),
        (["-s", "invert"], "invert"),
        (["--direction", "both"], "both"),
    ],
)
def test_native_align_conflicts_fail_before_loading(
    tmp_path, joined_file, native_binary, extra, message
):
    out = tmp_path / "out"
    write_text(out, "keep")
    r = _native(
        native_binary,
        "align",
        _missing_model_cases(tmp_path),
        str(joined_file),
        "-o",
        str(out),
        *extra,
    )
    assert r.returncode != 0 and message in r.stderr
    assert "No such file" not in r.stderr and "os error" not in r.stderr
    assert read_text(out) == "keep"


def test_native_align_symmetrize_with_scores_on_real_model(
    tmp_path, joined_file, native_binary, saved
):
    out = tmp_path / "out"
    write_text(out, "keep")
    r = _native(
        native_binary,
        "align",
        str(saved),
        str(joined_file),
        "-s",
        "union",
        "--scores",
        "-o",
        str(out),
    )
    assert r.returncode != 0
    assert read_text(out) == "keep"


def test_native_export_direction_both_rejected(tmp_path, native_binary, saved):
    out = tmp_path / "t"
    write_text(out, "keep")
    r = _native(native_binary, "export-table", str(saved), "--direction", "both", "-o", str(out))
    assert r.returncode != 0 and "both" in r.stderr
    assert read_text(out) == "keep"


def test_native_export_threads_and_beam(tmp_path, native_binary, saved):
    a, b = tmp_path / "a", tmp_path / "b"
    assert (
        _native(native_binary, "export-table", str(saved), "-t", "1", "-o", str(a)).returncode == 0
    )
    assert (
        _native(native_binary, "export-table", str(saved), "-t", "3", "-o", str(b)).returncode == 0
    )
    assert a.read_bytes() == b.read_bytes() and a.stat().st_size > 0
    c = tmp_path / "c"
    r = _native(native_binary, "export-table", str(saved), "--beam-threshold", "nan", "-o", str(c))
    assert r.returncode != 0 and not c.exists()


@pytest.mark.parametrize(
    "bad",
    [
        ["--p0", "2"],
        ["--p0", "-0.1"],
        ["--p0", "nan"],
        ["--alpha", "nan"],
        ["--alpha", "inf"],
        ["--alpha", "0"],
        ["--beam-threshold", "inf"],
    ],
)
def test_native_train_bad_numbers_write_nothing(tmp_path, joined_file, native_binary, bad):
    prefix = tmp_path / "t"
    r = _native(native_binary, "train", str(joined_file), "-o", str(prefix), "-q", *bad)
    assert r.returncode != 0
    assert not list(tmp_path.glob("t.*"))


def test_native_train_alpha_unused_without_vb_still_must_be_finite(
    tmp_path, joined_file, native_binary
):
    ok = _native(
        native_binary, "train", str(joined_file), "-o", str(tmp_path / "a"), "-q",
        "--no-variational-bayes", "--alpha", "0",
    )  # fmt: skip
    assert ok.returncode == 0, ok.stderr
    bad = _native(
        native_binary, "train", str(joined_file), "-o", str(tmp_path / "b"), "-q",
        "--no-variational-bayes", "--alpha", "nan",
    )  # fmt: skip
    assert bad.returncode != 0


@pytest.mark.parametrize(
    "bad", [["--p0", "1"], ["--p0", "nan"], ["-m", "nan"], ["--no-diagonal", "-T", "2"]]
)
def test_native_align_table_bad_options(tmp_path, native_binary, bad):
    table = tmp_path / "table"
    write_text(table, "a\tx\t-0.5\n")
    inp = tmp_path / "in.txt"
    write_text(inp, "a ||| x\n")
    out = tmp_path / "out"
    write_text(out, "keep")
    r = _native(native_binary, "align", "--table", str(table), str(inp), "-o", str(out), *bad)
    assert r.returncode != 0
    assert read_text(out) == "keep"


def test_python_cli_agrees_on_conflicts(tmp_path, joined_file, saved, capsys):
    out = tmp_path / "out"
    write_text(out, "keep")
    assert (
        main(["align", str(saved), str(joined_file), "-s", "union", "--scores", "-o", str(out)])
        != 0
    )
    assert read_text(out) == "keep"


# -- what `scores` is -------------------------------------------------------------------------


def test_score_is_marginal_not_best_path(tmp_path):
    """Two source words are equally likely to have produced the one target word. The score
    sums over both alignments (log 0.5), the best path alone would give log 0.25."""
    table = tmp_path / "table"
    write_text(table, "e1\tf\t%r\ne2\tf\t%r\n" % (math.log(0.5), math.log(0.5)))
    inp = tmp_path / "in.txt"
    write_text(inp, "e1 e2 ||| f\n")
    res = fastalign.align_with_table(
        table, str(inp), diagonal=False, null=False, mean_length_multiplier=1.0
    )
    fields = res.output.decode().strip().split(" ||| ")
    score = float(fields[3])
    n, m = 2, 1
    length_term = m * math.log(0.05 + n) - (0.05 + n) - math.lgamma(m + 1)
    marginal = length_term + math.log(0.5)  # sum over i of p / n = 0.5
    best_path = length_term + math.log(0.25)  # the winning position alone
    assert score == pytest.approx(marginal, abs=1e-4)
    assert abs(score - best_path) > 0.5
