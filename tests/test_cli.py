from __future__ import annotations

import subprocess
import sys

import fastalign
import pytest
from fastalign.cli import main


def test_no_command(capsys):
    assert main([]) == 2
    assert "command is required" in capsys.readouterr().err


def test_version(capsys):
    assert main(["-V"]) == 0
    assert fastalign.__version__ in capsys.readouterr().out


def test_train_align_symmetrize_export(tmp_path, corpus_files, corpus, capsys):
    a, b = corpus_files
    src = corpus[0]
    prefix = tmp_path / "out"
    model_path = tmp_path / "m.fastalign"
    rc = main(
        ["train", str(a), str(b), "-o", str(prefix), "-t", "2", "-s", "grow-diag-final-and"]
        + ["-s", "intersection", "--save", str(model_path), "--table", "--log", "-q"]
    )
    assert rc == 0
    for name in (
        "fwd.pharaoh",
        "inv.pharaoh",
        "grow-diag-final-and.pharaoh",
        "intersect.pharaoh",
        "fwd.table",
        "inv.log",
    ):
        assert (tmp_path / f"out.{name}").exists(), name
    assert len((tmp_path / "out.fwd.pharaoh").read_text().split("\n")) == len(src) + 1
    log = (tmp_path / "out.fwd.log").read_text()
    assert "ITERATION 5" in log and log.rstrip().endswith("out.fwd.table")

    m = fastalign.Model.load(model_path)
    assert (tmp_path / "out.fwd.pharaoh").read_bytes() == m.align_many(
        a, b, direction="fwd"
    ).to_bytes()

    # align: stdout, file, direction, scores
    capsys.readouterr()
    assert main(["align", str(model_path), str(a), str(b), "-s", "grow-diag-final-and"]) == 0
    assert capsys.readouterr().out == (tmp_path / "out.grow-diag-final-and.pharaoh").read_text()
    dest = tmp_path / "al.out"
    assert (
        main(["align", str(model_path), str(a), str(b), "--direction", "inv", "-o", str(dest)]) == 0
    )
    assert dest.read_bytes() == (tmp_path / "out.inv.pharaoh").read_bytes()
    assert main(["align", str(model_path), str(a), str(b), "--scores", "-o", str(dest)]) == 0
    assert all(" ||| " in ln for ln in dest.read_text().split("\n")[:-1])

    # symmetrize reproduces the training-time symmetrisation; invert is its own command
    capsys.readouterr()
    fwd, inv = tmp_path / "out.fwd.pharaoh", tmp_path / "out.inv.pharaoh"
    assert main(["symmetrize", str(fwd), str(inv), "-m", "grow-diag-final-and"]) == 0
    assert capsys.readouterr().out == (tmp_path / "out.grow-diag-final-and.pharaoh").read_text()
    assert main(["symmetrize", str(fwd), "-m", "invert"]) == 0
    inverted = capsys.readouterr().out.split("\n")[:-1]
    first = fastalign.parse_pharaoh(fwd.read_text().split("\n")[0])
    assert inverted[0] == " ".join(f"{i}-{j}" for i, j in fastalign.invert(first))

    # export-table equals the table written during training; align --table works with it
    table = tmp_path / "exported"
    assert main(["export-table", str(model_path), "-o", str(table)]) == 0
    assert table.read_bytes() == (tmp_path / "out.fwd.table").read_bytes()
    assert main(["export-table", str(model_path), "--direction", "inv"]) == 0
    assert capsys.readouterr().out == (tmp_path / "out.inv.table").read_text()
    assert main(["align", "--table", str(table), str(a), str(b), "-o", str(dest)]) == 0
    assert capsys.readouterr().err.startswith("Loaded ")
    assert " ||| " in dest.read_text()


def test_train_single_direction_to_stdout(corpus_files, joined_file, capfdbinary):
    a, b = corpus_files
    assert main(["train", str(a), str(b), "--directions", "inv", "-q"]) == 0
    out = capfdbinary.readouterr().out
    assert out == fastalign.train(joined_file, directions="inv").alignments().to_bytes()


def test_train_reads_joined_file(tmp_path, joined_file, capsys):
    assert main(["train", str(joined_file), "-o", str(tmp_path / "j"), "-q"]) == 0
    assert (tmp_path / "j.fwd.pharaoh").exists()


def test_progress_is_logged(tmp_path, joined_file, caplog):
    with caplog.at_level("INFO", logger="fastalign"):
        assert main(["train", str(joined_file), "-o", str(tmp_path / "j"), "-I", "2"]) == 0
    text = caplog.text
    assert "[fwd] iteration 2/2" in text and "done in" in text


@pytest.mark.parametrize(
    "argv, message",
    [
        (["train", "{j}"], "need --output"),
        (["train", "{j}", "-o", "{p}", "--directions", "fwd", "-s", "union"], "--directions both"),
        (["train", "{j}", "-s", "union"], "needs --output"),
        (["train", "{j}", "--directions", "fwd", "--table"], "need --output"),
        (["train", "{j}", "-o", "{p}", "--no-diagonal", "-T", "2"], "tension"),
        (["train", "{j}", "-o", "{p}", "-I", "-1"], "iterations"),
        (["train", "/nonexistent/file", "-o", "{p}"], "/nonexistent/file"),
        (["align", "/nonexistent/model", "{j}"], "/nonexistent/model"),
        (["symmetrize", "{j}"], "needs two"),
        (["align", "--table", "{j}", "{j}", "-s", "union"], "--table"),
    ],
)
def test_errors_exit_1(tmp_path, joined_file, capsys, argv, message):
    argv = [x.format(j=joined_file, p=tmp_path / "o") for x in argv]
    assert main(argv) == 1
    assert message in capsys.readouterr().err


def test_bad_option_exits_2(capsys):
    with pytest.raises(SystemExit) as e:
        main(["train", "x", "-s", "bogus"])
    assert e.value.code == 2


def test_align_table_exit_status_on_empty_side(tmp_path, capsys):
    corpus = tmp_path / "c"
    corpus.write_text("a b ||| x y\nb ||| y\n" * 5)
    assert (
        main(
            [
                "train",
                str(corpus),
                "-o",
                str(tmp_path / "t"),
                "--directions",
                "fwd",
                "--table",
                "-q",
            ]
        )
        == 0
    )
    bad = tmp_path / "bad"
    bad.write_text("a ||| x\n ||| y\n")
    capsys.readouterr()
    assert main(["align", "--table", str(tmp_path / "t.fwd.table"), str(bad)]) == 1
    assert "Error in line 2" in capsys.readouterr().err


def test_module_entry_point(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "fastalign.cli", "-V"], check=False, capture_output=True, text=True
    )
    assert r.returncode == 0 and fastalign.__version__ in r.stdout


def test_cli_equals_native_binary(tmp_path, joined_file, native_binary):
    args = ["train", str(joined_file), "--table", "--log", "--scores", "-s", "union", "-q"]
    subprocess.run([str(native_binary), *args, "-o", str(tmp_path / "n")], check=True)
    assert main([*args, "-o", str(tmp_path / "p")]) == 0
    for ext in ("fwd.pharaoh", "inv.pharaoh", "union.pharaoh", "fwd.table", "inv.table"):
        assert (tmp_path / f"n.{ext}").read_bytes() == (tmp_path / f"p.{ext}").read_bytes(), ext
