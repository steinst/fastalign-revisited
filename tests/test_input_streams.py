"""Corpus input from streams (review #8): ``-``, ``/dev/stdin``, FIFOs and process
substitutions give the same output as a regular file in both command lines, for joined and
paired input; a truly empty file is an empty corpus; standard input can be only one input."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import fastalign
import pytest
from conftest import make_corpus
from textio import write_text

posix = pytest.mark.skipif(sys.platform == "win32", reason="needs /dev/stdin and bash")


def _pycli():
    return [sys.executable, "-m", "fastalign.cli"]


@pytest.fixture(scope="module", params=["native", "python"])
def cli(request, native_binary):
    return [str(native_binary)] if request.param == "native" else _pycli()


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("streams")
    src, tgt, _ = make_corpus(n=150, seed=11)
    a, b, j = d / "c.src", d / "c.tgt", d / "c.txt"
    write_text(a, "".join(" ".join(s) + "\n" for s in src))
    write_text(b, "".join(" ".join(t) + "\n" for t in tgt))
    write_text(j, "".join(f"{' '.join(s)} ||| {' '.join(t)}\n" for s, t in zip(src, tgt)))
    return a, b, j


def _run(cmd, stdin_path=None, shell=False):
    cmd = ["bash", "-c", cmd] if shell else cmd
    if stdin_path is None:
        return subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, check=False)
    with open(stdin_path, "rb") as stdin:
        return subprocess.run(cmd, stdin=stdin, capture_output=True, check=False)


def _train(cli, *inputs, stdin=None, shell=False):
    args = [*inputs, "--directions", "fwd", "-q", "-t", "2", "-I", "3"]
    if shell:
        r = _run(" ".join([*map(_q, cli), "train", *args]), shell=True)
    else:
        r = _run([*cli, "train", *args], stdin_path=stdin)
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout


def _q(s: str) -> str:
    return s if s.startswith("<(") else "'" + s.replace("'", "'\\''") + "'"


def _pipe(cmd, data: bytes):
    """Run with standard input a pipe (not a file)."""
    return subprocess.run(cmd, input=data, capture_output=True, check=False)


@posix
def test_joined_stdin_dev_stdin_fifo_and_process_substitution_agree(cli, files, tmp_path):
    _, _, j = files
    want = _train(cli, str(j))
    assert want.count(b"\n") == 150
    data = j.read_bytes()
    args = ["--directions", "fwd", "-q", "-t", "2", "-I", "3"]
    for path in ("-", "/dev/stdin"):
        r = _pipe([*cli, "train", path, *args], data)
        assert r.returncode == 0, r.stderr.decode()
        assert r.stdout == want, path
    # a redirected regular file through `-` and /dev/stdin
    assert _train(cli, "-", stdin=j) == want
    assert _train(cli, "/dev/stdin", stdin=j) == want
    # process substitution: /dev/fd/N, a pipe
    assert _train(cli, f"<(cat '{j}')", shell=True) == want
    # a named FIFO
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    writer = subprocess.Popen(["bash", "-c", f"cat '{j}' > '{fifo}'"])
    try:
        assert _train(cli, str(fifo)) == want
    finally:
        writer.wait()


@posix
def test_paired_stdin_on_either_side(cli, files):
    a, b, j = files
    want = _train(cli, str(a), str(b))
    assert want == _train(cli, str(j))  # no tabs or `|||` in the sides
    args = ["--directions", "fwd", "-q", "-t", "2", "-I", "3"]
    r = _pipe([*cli, "train", "-", str(b), *args], a.read_bytes())
    assert r.returncode == 0 and r.stdout == want, r.stderr.decode()
    r = _pipe([*cli, "train", str(a), "-", *args], b.read_bytes())
    assert r.returncode == 0 and r.stdout == want, r.stderr.decode()
    r = _pipe([*cli, "train", "/dev/stdin", str(b), *args], a.read_bytes())
    assert r.returncode == 0 and r.stdout == want, r.stderr.decode()
    assert _train(cli, f"<(cat '{a}')", f"<(cat '{b}')", shell=True) == want


@posix
def test_stdin_only_once_and_line_count_mismatch(cli, files):
    a, b, _ = files
    r = _pipe([*cli, "train", "-", "-", "--directions", "fwd", "-q"], a.read_bytes())
    assert r.returncode != 0 and b"standard input" in r.stderr
    short = b"".join(b.read_bytes().splitlines(keepends=True)[:-1])
    r = _pipe([*cli, "train", str(a), "-", "--directions", "fwd", "-q"], short)
    assert r.returncode != 0 and b"different numbers of lines" in r.stderr
    r = _run(
        " ".join([*map(_q, cli), "train", f"<(cat '{a}')", f"<(head -n 3 '{b}')", "-q"])
        + " --directions fwd",
        shell=True,
    )
    assert r.returncode != 0 and b"(150 vs 3)" in r.stderr
    r = _pipe([*cli, "symmetrize", "-", "-"], b"0-0\n")
    assert r.returncode != 0 and b"standard input" in r.stderr


@posix
def test_align_reads_stdin(cli, files, tmp_path):
    a, b, j = files
    model = tmp_path / "m.fastalign"
    r = _run([*cli, "train", str(j), "-o", str(tmp_path / "o"), "-q", "--save", str(model)])
    assert r.returncode == 0, r.stderr.decode()
    want = _run([*cli, "align", str(model), str(a), str(b)]).stdout
    assert want.count(b"\n") == 150
    for args, data in [
        (["-", str(b)], a.read_bytes()),
        ([str(a), "-"], b.read_bytes()),
        (["-"], j.read_bytes()),
        (["/dev/stdin"], j.read_bytes()),
    ]:
        r = _pipe([*cli, "align", str(model), *args], data)
        assert r.returncode == 0 and r.stdout == want, (args, r.stderr.decode())
    table = tmp_path / "o.table"
    _run([*cli, "export-table", str(model), "-o", str(table)])
    want = _run([*cli, "align", "--table", str(table), str(j)])
    r = _pipe([*cli, "align", "--table", str(table), "-"], j.read_bytes())
    assert r.returncode == 0 and r.stdout == want.stdout and want.stdout.count(b"\n") == 150


def test_empty_regular_file_and_empty_pipe_agree(cli, tmp_path):
    e = tmp_path / "empty"
    e.write_bytes(b"")
    # Both front ends train nothing, write nothing and warn (Step 22); an empty file and an
    # empty pipe agree.
    want = (0, b"", b"fastalign: warning: the input has no line pairs")
    for r in (
        _run([*cli, "train", str(e), "--directions", "fwd", "-q"]),
        _pipe([*cli, "train", "-", "--directions", "fwd", "-q"], b""),
        _run([*cli, "train", str(e), str(e), "--directions", "fwd", "-q"]),
        _pipe([*cli, "train", "-", str(e), "--directions", "fwd", "-q"], b""),
    ):
        assert (r.returncode, r.stdout) == want[:2] and want[2] in r.stderr, r.stderr


@posix
def test_api_reads_a_fifo_path(files, tmp_path):
    """The Python API reads a non-regular path to the end too (it was an empty corpus)."""
    _, _, j = files
    want = fastalign.train(j, directions="fwd", iterations=3, threads=2)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    writer = subprocess.Popen(["bash", "-c", f"cat '{j}' > '{fifo}'"])
    try:
        got = fastalign.train(fifo, directions="fwd", iterations=3, threads=2)
    finally:
        writer.wait()
    assert got.alignments().to_bytes() == want.alignments().to_bytes()


@posix
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_subword_training_spools_a_pipe(files, tmp_path):
    """Tokeniser training reads the corpus twice: a pipe is spooled once, so a process
    substitution gives the model of the regular file (Python CLI)."""
    pytest.importorskip("sentencepiece")
    _, _, j = files
    flags = "--directions fwd -q -t 2 -I 3 --subword --subword-vocab 300"
    want = _run(" ".join([*map(_q, _pycli()), "train", _q(str(j)), flags]), shell=True)
    assert want.returncode == 0, want.stderr.decode()
    got = _run(" ".join([*map(_q, _pycli()), "train", f"<(cat '{j}')", flags]), shell=True)
    assert got.returncode == 0, got.stderr.decode()
    assert got.stdout == want.stdout and want.stdout.count(b"\n") == 150
    r = _pipe([*_pycli(), "train", "-", *flags.split()], j.read_bytes())
    assert r.returncode == 0 and r.stdout == want.stdout, r.stderr.decode()
