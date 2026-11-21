"""Reading alignment (Pharaoh) files: one definition of the layouts, CR rule, atools parity
(Step 20)."""

from __future__ import annotations

import math
import subprocess

import fastalign
import pytest
from fastalign.cli import main
from textio import read_text, write_text

FWD = ["0-0 1-1 2-2", "", "0-1 0-1 3-0", "1-0 2-1", "0-0"]
INV = ["0-0 1-2", "0-0", "", "0-1 1-0 2-1", "0-0 1-1"]


def _atools(atools, method, a, b, tmp_path):
    (tmp_path / "a").write_bytes(a)
    (tmp_path / "b").write_bytes(b)
    cmd = [str(atools), "-c", method, "-i", str(tmp_path / "a")]
    if method != "invert":
        cmd += ["-j", str(tmp_path / "b")]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def _sym(a: bytes, b: bytes, method: str) -> bytes:
    return fastalign._core.symmetrize_text(a, None if method == "invert" else b, method)


def _join(lines, eol=b"\n", final=True):
    text = eol.join(x.encode() for x in lines)
    return text + eol if final else text


@pytest.mark.parametrize("method", fastalign.METHODS)
@pytest.mark.parametrize(
    "eol, blank",
    [(b"\n", False), (b"\r\n", False), (b"\n", True), (b"\r\n", True)],
    ids=["lf", "crlf", "lf-blank", "crlf-blank"],
)
def test_methods_equal_atools(atools_binary, tmp_path, method, eol, blank):
    f, i = list(FWD), list(INV)
    if blank:  # blank lines at the start, middle and end
        f = ["", *f, "", ""]
        i = ["", *i, "", ""]
    a, b = _join(f, eol), _join(i, eol)
    want = _atools(atools_binary, method, a, b, tmp_path)
    assert _sym(a, b, method) == want
    # CRLF reads like LF
    assert _sym(_join(f), _join(i), method) == want


@pytest.mark.parametrize("method", fastalign.METHODS)
@pytest.mark.parametrize("eol", [b"\n", b"\r\n"])
def test_unterminated_final_line_is_processed_once(atools_binary, tmp_path, method, eol):
    # atools duplicates an unterminated last line; we deliberately do not (documented). The
    # result equals atools on the same text with the final newline added.
    a, b = _join(FWD, eol, final=False), _join(INV, eol, final=False)
    want = _atools(atools_binary, method, _join(FWD, eol), _join(INV, eol), tmp_path)
    assert _sym(a, b, method) == want


def test_parse_layouts():
    p = fastalign.parse_pharaoh
    assert p("0-0 1-1") == [(0, 0), (1, 1)]
    assert p("0-0 1-1 ||| -4.8351") == [(0, 0), (1, 1)]
    assert p(" ||| -2") == []
    assert p("a b ||| x y ||| 1-0 0-1") == [(0, 1), (1, 0)]
    assert p("a ||| x ||| 0-0 ||| -3.5") == [(0, 0)]
    assert p("0-0 1-1\r") == [(0, 0), (1, 1)]
    for bad in ["0-1 ||| abc", "0-1 ||| 1 ||| 2", "-3.5", "0:1", "0-"]:
        with pytest.raises(ValueError):
            p(bad)


def test_invert_removes_duplicates(tmp_path):
    assert fastalign.invert([(0, 1), (0, 1)]) == [(1, 0)]
    src = tmp_path / "d"
    write_text(src, "0-1 0-1\n")
    out = tmp_path / "o"
    assert main(["symmetrize", str(src), "-m", "invert", "-o", str(out)]) == 0
    assert read_text(out) == "1-0\n"


SCORES = ["-4.8351", "nan", "-nan", "inf", "-inf", "0", "1e-05"]


@pytest.mark.parametrize("method", fastalign.METHODS)
def test_symmetrize_reads_scored_output(tmp_path, method):
    plain_f, plain_i = _join(FWD), _join(INV)
    want = _sym(plain_f, plain_i, method)
    scored_f = _join([f"{x} ||| {SCORES[k % 7]}" for k, x in enumerate(FWD)])
    scored_i = _join([f"{x} ||| {SCORES[(k + 3) % 7]}" for k, x in enumerate(INV)])
    # empty link sets are ` ||| score`
    assert b"\n ||| " in scored_f
    assert _sym(scored_f, scored_i, method) == want
    # atools-prefixed (with and without a trailing score), CRLF scored
    pre_f = _join([f"s{k} w ||| t{k} ||| {x}" for k, x in enumerate(FWD)])
    pre_i = _join([f"s{k} w ||| t{k} ||| {x} ||| -2.5" for k, x in enumerate(INV)])
    assert _sym(pre_f, pre_i, method) == want
    assert (
        _sym(_join([f"{x} ||| {SCORES[0]}" for x in FWD], b"\r\n"), _join(INV, b"\r\n"), method)
        == want
    )


def test_symmetrize_reads_train_scores_output(tmp_path, native_binary, corpus_files):
    a, b = corpus_files
    pre = tmp_path / "s"
    subprocess.run(
        [str(native_binary), "train", str(a), str(b), "-o", str(pre), "--scores", "-t", "1"],
        check=True,
        capture_output=True,
    )
    f, i = tmp_path / "s.fwd.pharaoh", tmp_path / "s.inv.pharaoh"
    assert b" ||| " in f.read_bytes()
    plain_f = b"".join(x.split(b" ||| ")[0] + b"\n" for x in f.read_bytes().splitlines())
    plain_i = b"".join(x.split(b" ||| ")[0] + b"\n" for x in i.read_bytes().splitlines())
    for method in fastalign.METHODS:
        standalone = subprocess.run(
            [str(native_binary), "symmetrize", str(f), str(i), "-m", method],
            check=True,
            capture_output=True,
        ).stdout
        assert standalone == _sym(plain_f, plain_i, method), method


@pytest.mark.parametrize("method", [m for m in fastalign.METHODS if m != "invert"])
def test_file_in_memory_and_links_agree(method):
    dup_f = ["0-0 0-0 1-1", "0-1 0-1", ""]
    dup_i = ["1-1 1-1 0-0", "", "0-0 0-0"]
    text = _sym(_join(dup_f), _join(dup_i), method).decode().split("\n")[:-1]
    lists = fastalign.symmetrize_many(
        [fastalign.parse_pharaoh(x) for x in dup_f],
        [fastalign.parse_pharaoh(x) for x in dup_i],
        method,
    )
    assert text == [" ".join(f"{i}-{j}" for i, j in links) for links in lists]
    # raw (duplicated) python link lists agree too
    raw = fastalign.symmetrize_many(
        [[(0, 0), (0, 0), (1, 1)], [(0, 1), (0, 1)], []],
        [[(1, 1), (1, 1), (0, 0)], [], [(0, 0), (0, 0)]],
        method,
    )
    assert raw == lists
    assert not any(math.isnan(i) for links in lists for pair in links for i in pair)
