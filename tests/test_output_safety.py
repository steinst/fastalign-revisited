"""File output is atomic and validated before any destination is touched (Step 17)."""

from __future__ import annotations

import os
import subprocess
import threading

import fastalign
import pytest
from fastalign.cli import main
from textio import write_text


def _train(words: str):
    src = [[f"{words}{i}", "a"] for i in range(30)]
    tgt = [[f"t{words}{i}", "b"] for i in range(30)]
    return fastalign.train(src, tgt, threads=1)


def test_save_keeps_unrelated_tmp_sidecar(tmp_path, model):
    side = tmp_path / "checkpoint.tmp"
    side.write_bytes(b"independent data")
    model.save(tmp_path / "checkpoint")
    assert side.read_bytes() == b"independent data"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint", "checkpoint.tmp"]
    assert fastalign.Model.load(tmp_path / "checkpoint").vocab_size == model.vocab_size


def test_concurrent_saves_leave_one_complete_model(tmp_path):
    a, b = _train("x"), _train("y")
    path = tmp_path / "m"
    a.save(path)
    errors: list[BaseException] = []

    def run(m):
        try:
            for _ in range(15):
                m.save(path)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=run, args=(m,)) for m in (a, b, a, b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    got = fastalign.Model.load(path)
    assert got.vocab_size in (a.vocab_size, b.vocab_size)
    assert [p.name for p in tmp_path.iterdir()] == ["m"]


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions")
def test_write_failure_keeps_old_checkpoint(tmp_path, model):
    d = tmp_path / "ro"
    d.mkdir()
    model.save(d / "m")
    before = (d / "m").read_bytes()
    d.chmod(0o555)
    try:
        with pytest.raises(OSError):
            _train("z").save(d / "m")
    finally:
        d.chmod(0o755)
    assert (d / "m").read_bytes() == before


def test_python_cli_failures_keep_destination(tmp_path):
    out = tmp_path / "out"
    out.write_text("keep")
    bad = tmp_path / "bad.pharaoh"
    write_text(bad, "0-0\nnot a link\n", encoding="utf-8")
    ok = tmp_path / "ok.pharaoh"
    write_text(ok, "0-0\n0-0\n", encoding="utf-8")
    assert main(["symmetrize", str(bad), str(ok), "-m", "union", "-o", str(out)]) != 0
    assert out.read_text() == "keep"


def _native(binary, *args):
    return subprocess.run([str(binary), *args], capture_output=True, text=True, check=False)


def test_native_export_wrong_direction_keeps_destination(tmp_path, joined_file, native_binary):
    m = tmp_path / "fwd.model"
    r = _native(
        native_binary, "train", str(joined_file), "--directions", "fwd", "-q", "--save", str(m)
    )
    assert r.returncode == 0, r.stderr
    out = tmp_path / "table"
    out.write_text("keep")
    r = _native(native_binary, "export-table", str(m), "--direction", "inv", "-o", str(out))
    assert r.returncode != 0 and "no 'inv' direction" in r.stderr
    assert out.read_text() == "keep"
    r = _native(native_binary, "export-table", str(m), "-o", str(out))
    assert r.returncode == 0 and out.read_text() != "keep"


@pytest.mark.parametrize("bad_at", ["first", "late"])
def test_native_symmetrize_bad_line_keeps_destination(tmp_path, native_binary, bad_at):
    n = 300_000 if bad_at == "late" else 3  # past the first 1 MiB output chunk
    good = "0-0 1-1 2-2 3-3\n"
    lines = [good] * n
    lines[0 if bad_at == "first" else -1] = "0-0 oops\n"
    f, i = tmp_path / "f", tmp_path / "i"
    write_text(f, "".join(lines), encoding="utf-8")
    write_text(i, good * n, encoding="utf-8")
    out = tmp_path / "out"
    out.write_text("keep")
    r = _native(native_binary, "symmetrize", str(f), str(i), "-m", "union", "-o", str(out))
    assert r.returncode != 0
    assert out.read_text() == "keep"
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def test_native_train_save_replaces_without_clobbering_sidecar(
    tmp_path, joined_file, native_binary
):
    (tmp_path / "m.tmp").write_bytes(b"independent")
    r = _native(
        native_binary,
        "train",
        str(joined_file),
        "-q",
        "--save",
        str(tmp_path / "m"),
        "-o",
        str(tmp_path / "o"),
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "m.tmp").read_bytes() == b"independent"
    assert fastalign.Model.load(tmp_path / "m").vocab_size > 0


def test_native_align_table_empty_side_keeps_partial_output(tmp_path, native_binary):
    """fast_align's -f mode: an empty side stops the run with status 1; output so far stays."""
    src = tmp_path / "c"
    write_text(src, "a b ||| x y\na b ||| x y\n", encoding="utf-8")
    t = tmp_path / "t"
    r = _native(
        native_binary, "train", str(src), "-q", "--directions", "fwd", "--table", "-o", str(t)
    )
    assert r.returncode == 0, r.stderr
    bad = tmp_path / "bad"
    write_text(bad, "a b ||| x y\n ||| y\n", encoding="utf-8")
    out = tmp_path / "out"
    out.write_text("old")
    r = _native(native_binary, "align", "--table", str(t) + ".fwd.table", str(bad), "-o", str(out))
    assert r.returncode == 1
    assert out.read_text().startswith("a b ||| x y ||| ") or out.read_text().startswith("0-0")
