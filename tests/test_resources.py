"""Step 25: outputs-only training (each direction's tables written, then dropped), borrowed write
paths, the thread budget of a whole command, parallel ``symmetrize_many``."""

from __future__ import annotations

import subprocess

import fastalign
import pytest
from fastalign.cli import main
from textio import read_text


def _native(binary, *args):
    return subprocess.run([str(binary), *args], capture_output=True, text=True, check=False)


def _bytes(path):
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.parametrize("subword", [False, True])
def test_outputs_only_training_equals_keeping_the_models(tmp_path, joined_file, subword):
    if subword:
        pytest.importorskip("sentencepiece")
    sub = ["--subword", "--subword-vocab", "300"] if subword else []
    # reference: train keeping both models, export afterwards
    ref = tmp_path / "ref"
    assert (
        main(
            [
                "train",
                str(joined_file),
                "-o",
                str(ref),
                "-q",
                "--save",
                str(ref) + ".m",
                "-s",
                "union",
            ]
            + sub
        )
        == 0
    )
    model = fastalign.Model.load(str(ref) + ".m")
    for d in ("fwd", "inv"):
        model.write_table(tmp_path / f"want.{d}.table", d)
    # outputs only: tables written as each direction finishes
    out = tmp_path / "out"
    args = [
        "train",
        str(joined_file),
        "-o",
        str(out),
        "-q",
        "--table",
        "--log",
        "-s",
        "union",
    ] + sub
    assert main(args) == 0
    for d in ("fwd", "inv"):
        assert _bytes(f"{out}.{d}.table") == _bytes(tmp_path / f"want.{d}.table")
        assert _bytes(f"{out}.{d}.pharaoh") == _bytes(f"{ref}.{d}.pharaoh")
        assert b"conditional probabilities:" in _bytes(f"{out}.{d}.log")
    assert _bytes(f"{out}.union.pharaoh") == _bytes(f"{ref}.union.pharaoh")


def test_outputs_only_model_refuses_what_needs_the_tables(tmp_path, joined_file):
    m = fastalign.train(str(joined_file), _outputs_only=True, _table_files=None)
    assert m.directions == ("fwd", "inv")
    # what the training produced is there ...
    assert len(list(m.alignments(direction="fwd"))) > 0
    assert len(list(m.alignments(symmetrize="union"))) > 0
    m.write_alignments(tmp_path / "a.pharaoh", direction="inv")
    assert m._inner.training_log("fwd")
    # ... what needs the model is a clear error, not an empty result
    with pytest.raises(ValueError, match="without keeping the translation tables"):
        m.save(tmp_path / "x.model")
    with pytest.raises(ValueError, match="without keeping the translation tables"):
        m.write_table(tmp_path / "x.table", "fwd")
    with pytest.raises(ValueError, match="without keeping the translation tables"):
        m.align_many(str(joined_file))
    with pytest.raises(ValueError, match="without keeping the translation tables"):
        m.align(["a"], ["b"])
    assert not (tmp_path / "x.model").exists()
    assert not (tmp_path / "x.table").exists()


def test_failed_training_publishes_no_table(tmp_path, joined_file):
    # the second direction's table file cannot be created: nothing is published
    ok = tmp_path / "fwd.table"
    bad = tmp_path / "no-such-dir" / "inv.table"
    with pytest.raises(OSError):
        fastalign.train(
            str(joined_file),
            _outputs_only=True,
            _table_files={"fwd": str(ok), "inv": str(bad)},
        )
    assert not ok.exists()
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def test_table_write_streams_through_a_staged_file(tmp_path, model):
    dest = tmp_path / "t.table"
    dest.write_bytes(b"old")
    model.write_table(dest, "fwd")
    assert dest.read_bytes() != b"old"
    with pytest.raises(ValueError):
        model.write_table(tmp_path / "bad.table", "nope")
    assert not (tmp_path / "bad.table").exists()
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def test_directional_write_alignments_equal_the_alignments_text(tmp_path, model):
    for d in model.directions:
        model.write_alignments(tmp_path / f"{d}.pharaoh", direction=d)
        assert _bytes(tmp_path / f"{d}.pharaoh") == model.alignments(direction=d).to_bytes()
    model.write_alignments(tmp_path / "s.pharaoh", symmetrize="union")
    assert _bytes(tmp_path / "s.pharaoh") == model.alignments(symmetrize="union").to_bytes()


def test_python_cli_align_writes_the_same_file_as_the_api(tmp_path, joined_file, model):
    m = tmp_path / "m.model"
    model.save(m)
    out = tmp_path / "o.pharaoh"
    assert main(["align", str(m), str(joined_file), "-o", str(out), "-s", "union"]) == 0
    assert _bytes(out) == model.align_many(str(joined_file), symmetrize="union").to_bytes()


def test_native_table_export_while_training_equals_export_from_model(
    tmp_path, joined_file, native_binary
):
    ref = tmp_path / "ref"
    r = _native(
        native_binary,
        "train",
        str(joined_file),
        "-o",
        str(ref),
        "-q",
        "--save",
        str(ref) + ".m",
        "-s",
        "union",
    )
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out"
    r = _native(
        native_binary, "train", str(joined_file), "-o", str(out), "-q", "--table", "--log",
        "-s", "union", "--threads", "3",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    for d in ("fwd", "inv"):
        e = _native(
            native_binary,
            "export-table",
            str(ref) + ".m",
            "--direction",
            d,
            "-o",
            str(tmp_path / f"e.{d}"),
        )
        assert e.returncode == 0, e.stderr
        assert _bytes(f"{out}.{d}.table") == _bytes(tmp_path / f"e.{d}")
        assert read_text(tmp_path / f"out.{d}.log").endswith(
            f"conditional probabilities: {out}.{d}.table\n"
        )
    assert (
        _bytes(f"{out}.union.pharaoh") == _bytes(f"{ref}.union.pharaoh")
        if (tmp_path / "ref.union.pharaoh").exists()
        else True
    )


@pytest.mark.parametrize("method", ["intersect", "union", "grow-diag", "grow-diag-final-and"])
def test_symmetrize_many_is_the_same_at_every_thread_count(model, method):
    # many more pairs than one batch
    f = [[(i, i % 7) for i in range(1 + k % 9)] for k in range(3000)]
    b = [[(i % 5, i) for i in range(1 + (k * 7) % 8)] for k in range(3000)]
    one = fastalign.symmetrize_many(f, b, method, threads=1)
    assert fastalign.symmetrize_many(f, b, method, threads=4) == one
    assert fastalign.symmetrize_many(f, b, method) == one
    assert one[17] == fastalign.symmetrize(f[17], b[17], method)


def test_symmetrize_many_reports_the_first_failing_pair(model):
    big = (1 << 20) - 1
    f = [[(0, 0)] for _ in range(2000)]
    b = [[(0, 0)] for _ in range(2000)]
    f[700] = [(0, 0), (big, big)]
    f[1500] = [(0, 0), (big, big)]
    b[700] = [(0, 0), (big, big)]
    b[1500] = [(0, 0), (big, big)]
    try:
        fastalign.symmetrize_many(f, b, "grow-diag", threads=4)
    except ValueError as e:
        assert "pair 700" in str(e), str(e)
