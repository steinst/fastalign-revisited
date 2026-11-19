"""Edge cases: behaviour of the Python API and CLI against fast_align and the native binary."""

import subprocess

import fastalign
import pytest

TRICKY = (
    b"a b ||| x y\n"
    b"\n"
    b" ||| x\n"
    b"c d |||\n"
    b"no separator here\n"
    b"a ||| b ||| c\n"
    b"caf\xe9 \xe9t\xe9 ||| \xc3\xa9 z\r\n"
    b"<eps> a ||| x <eps>\n"
    b"e f g ||| x y z"  # no trailing newline
)


@pytest.fixture()
def tricky(tmp_path):
    p = tmp_path / "tricky.txt"
    p.write_bytes(TRICKY)
    return p


def test_iterations_zero_matches_fast_align(tmp_path, tricky, fast_align_binary):
    ref = subprocess.run(
        [str(fast_align_binary), "-i", str(tricky), "-d", "-o", "-v", "-I", "0"],
        capture_output=True,
        env={"OMP_NUM_THREADS": "1"},
        check=True,
    )
    m = fastalign.train(str(tricky), directions="fwd", iterations=0, threads=1)
    m.write_alignments(tmp_path / "o.txt", direction="fwd")
    assert (tmp_path / "o.txt").read_bytes() == ref.stdout


@pytest.mark.parametrize("flags", [["-d", "-o", "-v"], [], ["-d", "-o", "-v", "-s"]])
def test_tricky_input_equals_fast_align(tmp_path, tricky, fast_align_binary, flags):
    ref = subprocess.run(
        [str(fast_align_binary), "-i", str(tricky), *flags],
        capture_output=True,
        env={"OMP_NUM_THREADS": "1"},
        check=True,
    )
    kw = {
        "diagonal": "-d" in flags,
        "optimize_tension": "-o" in flags,
        "variational_bayes": "-v" in flags,
        "scores": "-s" in flags,
    }
    for threads in (1, 3):
        m = fastalign.train(str(tricky), directions="fwd", threads=threads, **kw)
        out = tmp_path / f"o{threads}.txt"
        m.write_alignments(out, direction="fwd")
        assert out.read_bytes() == ref.stdout


def test_corpus_without_tokens(tmp_path):
    """A file of empty lines used to panic in the lookup of the NULL row."""
    p = tmp_path / "blank.txt"
    p.write_bytes(b"\n\n")
    m = fastalign.train(str(p), directions="fwd", threads=2)
    m.write_alignments(tmp_path / "o.txt", direction="fwd")
    assert (tmp_path / "o.txt").read_bytes() == b"\n\n"


def test_max_len_native_matches_python(tmp_path, joined_file, native_binary):
    """Both front ends write one line per input line when --max-len drops lines."""
    subprocess.run(
        [
            str(native_binary),
            "train",
            str(joined_file),
            "-o",
            str(tmp_path / "n"),
            "--max-len",
            "5",
            "--directions",
            "fwd",
            "-q",
            "-t",
            "2",
        ],
        check=True,
    )
    m = fastalign.train(str(joined_file), directions="fwd", max_len=5, threads=2)
    m.write_alignments(tmp_path / "p.txt", direction="fwd")
    native = (tmp_path / "n.fwd.pharaoh").read_bytes()
    assert native == (tmp_path / "p.txt").read_bytes()
    n_input = len(joined_file.read_bytes().splitlines())
    assert native.count(b"\n") == n_input


def test_missing_table_is_an_error(native_binary, joined_file):
    r = subprocess.run(
        [str(native_binary), "align", "--table", "/nonexistent/table", str(joined_file)],
        capture_output=True,
        check=False,
    )
    assert r.returncode == 1 and b"/nonexistent/table" in r.stderr
