"""Sentence sides do not depend on the input form (review #9, decision B).

Two files are the two sides: a tab or ``|||`` inside a line stays on its side. A joined
``source ||| target`` file keeps fast_align's byte-exact rules (tab is a separator, the first
separator splits). Subword preprocessing sees exactly the words and sides of the word workflow.
Word counts are checked through the trainer's mean target/source length ratio (fast_align's
``expected target length = source length * R``), which is computed from the core's own counts.
"""

from __future__ import annotations

import random
import re
import subprocess

import fastalign
import pytest
from fastalign import subword as sw

# (source file, target file) lines -> (source words, target words)
PAIRED = [
    (b"a\tb c", b"x y z", [b"a", b"b", b"c"], [b"x", b"y", b"z"]),  # the review's case
    (b"a ||| b", b"x", [b"a", b"|||", b"b"], [b"x"]),
    (b"a b\r", b"x y\r", [b"a", b"b\r"], [b"x", b"y\r"]),
    (b"a\t\tb", b"x ||| ||| y", [b"a", b"b"], [b"x", b"|||", b"|||", b"y"]),
    (b"a b \r", b"x", [b"a", b"b", b"\r"], [b"x"]),
    (b"  a\t |||\t", b"\tx\t", [b"a", b"|||"], [b"x"]),
]

# joined line -> (source words, target words), fast_align's rules
JOINED = [
    (b"a\tb c ||| x y z", [b"a"], [b"b", b"c", b"|||", b"x", b"y", b"z"]),
    (b"a ||| b\tc", [b"a"], [b"b", b"|||", b"c"]),
    (b"a b ||| x y\r", [b"a", b"b"], [b"x", b"y\r"]),
    (b"a ||| ||| x", [b"a"], [b"|||", b"x"]),
    (b"a\t\tb", [b"a"], [b"|||", b"b"]),
    (b"a b ||| x \r", [b"a", b"b"], [b"x", b"\r"]),
    (b"a b\t|||x", [b"a", b"b"], [b"|||x"]),
]


def write_bytes(path, data: bytes) -> None:
    path.write_bytes(data)


def _ratio_py(*inputs) -> float:
    m = fastalign.train(*inputs, directions="fwd", iterations=1, threads=1)
    return m.parameters("fwd")["mean_srclen_multiplier"]


def _ratio_native(binary, tmp_path, *inputs) -> float:
    out = tmp_path / "o"
    r = subprocess.run(
        [str(binary), "train", *map(str, inputs), "--directions", "fwd", "-I", "1", "-q"]
        + ["-o", str(out), "--log"],
        capture_output=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "o.fwd.log").read_text()
    return float(re.search(r"expected target length = source length \* (\S+)", log).group(1))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-5 * max(1.0, abs(b))  # the native log has 6 digits


@pytest.mark.parametrize("case", range(len(PAIRED)))
def test_paired_files_are_the_sides(case, tmp_path, native_binary):
    a_line, b_line, sw_, tw_ = PAIRED[case]
    a, b = tmp_path / "a", tmp_path / "b"
    write_bytes(a, a_line + b"\n")
    write_bytes(b, b_line + b"\n")
    want = len(tw_) / len(sw_)
    assert _close(_ratio_py(a, b), want)
    assert _close(_ratio_native(native_binary, tmp_path, a, b), want)
    # Python's subword reader sees the same words
    sa, sb = sw.corpus_sides(a, b)
    for side, words in ((sa, sw_), (sb, tw_)):
        if b"|||" in words:  # subwords cannot carry the word `|||`: an error, not a new split
            with pytest.raises(ValueError, match="'\\|\\|\\|' cannot be aligned"):
                list(side())
        else:
            assert list(side()) == [words]
    # token lists give the same corpus (when the words can be token-list words)
    if all(w != b"|||" and not re.search(rb"\s", w) for w in sw_ + tw_):
        lists = [[w.decode() for w in sw_]], [[w.decode() for w in tw_]]
        assert _close(_ratio_py(*lists), want)


@pytest.mark.parametrize("case", range(len(JOINED)))
def test_joined_files_keep_fast_align_rules(case, tmp_path, native_binary):
    line, sw_, tw_ = JOINED[case]
    j = tmp_path / "j"
    write_bytes(j, line + b"\n")
    want = len(tw_) / len(sw_)
    assert _close(_ratio_py(j), want)
    assert _close(_ratio_native(native_binary, tmp_path, j), want)
    assert sw.joined_words(line) == (sw_, tw_)
    sa, sb = sw.corpus_sides(j, None)
    assert list(sa()) == [sw_]
    if b"|||" in tw_:
        # a later separator is the target word `|||`, which subwords cannot carry: an error,
        # never a different side split
        with pytest.raises(ValueError, match="'\\|\\|\\|' cannot be aligned"):
            list(sb())
    else:
        assert list(sb()) == [tw_]


def _quirky(n: int, seed: int):
    """Sides with tabs, repeated blanks, CRLF and lone-CR words (no `|||`)."""
    rng = random.Random(seed)
    vocab = [f"w{k}" for k in range(30)]

    def line() -> bytes:
        words = [rng.choice(vocab).encode() for _ in range(rng.randint(1, 8))]
        out = b""
        for k, w in enumerate(words):
            if k:
                out += rng.choice([b" ", b"\t", b"  ", b" \t ", b"\t\t"])
            out += w
        if rng.random() < 0.3:
            out += b" \r"  # a word of its own
        if rng.random() < 0.3:
            out = rng.choice([b"\t", b" "]) + out
        return out + rng.choice([b"\n", b"\r\n"])

    return [line() for _ in range(n)], [line() for _ in range(n)]


def _expected_ratio(src_words, tgt_words) -> float:
    return sum(len(t) / len(s) for s, t in zip(src_words, tgt_words)) / len(src_words)


def test_quirky_paired_and_joined_word_counts_match_the_core(tmp_path, native_binary):
    src, tgt = _quirky(200, 5)
    a, b, j = tmp_path / "a", tmp_path / "b", tmp_path / "j"
    write_bytes(a, b"".join(src))
    write_bytes(b, b"".join(tgt))
    # joined form: the first tab of the source becomes the split point in the core, so use
    # spaces in the source and a tab as the separator
    joined = [
        s.rstrip(b"\r\n").replace(b"\t", b" ") + b"\t" + t.rstrip(b"\n") + b"\n"
        for s, t in zip(src, tgt)
    ]
    write_bytes(j, b"".join(joined))
    pa, pb = sw.corpus_sides(a, b)
    ps, pt = list(pa()), list(pb())
    assert [len(x) for x in ps] == [len(sw.side_words(s.rstrip(b"\n"))) for s in src]
    want = _expected_ratio(ps, pt)
    assert _close(_ratio_py(a, b), want)
    assert _close(_ratio_native(native_binary, tmp_path, a, b), want)
    # joined: the source words are the paired source's (tabs were spaces; the CR of a CRLF
    # source line is gone with the line end); a tab in the target is a `|||` word there
    js = [sw.joined_words(x.rstrip(b"\n"))[0] for x in joined]
    assert js == [sw.side_words(s.rstrip(b"\r\n")) for s in src]
    jwant = _expected_ratio(*zip(*[sw.joined_words(x.rstrip(b"\n")) for x in joined]))
    assert _close(_ratio_py(j), jwant)
    assert _close(_ratio_native(native_binary, tmp_path, j), jwant)
    assert len(js) == 200


def _small_config():
    return fastalign.SubwordConfig(vocab_size=300)


def test_subword_tab_case_ratio_one(tmp_path):
    """The review's tab case through subwords: source words equal target words, so with a
    joint tokeniser the piece corpus has ratio 1.0 (it was 1.0 in no form before the fix only
    for the core's paired reader)."""
    pytest.importorskip("sentencepiece")
    src, _ = _quirky(120, 9)
    a, b = tmp_path / "a", tmp_path / "b"
    write_bytes(a, b"".join(src))
    write_bytes(b, b"".join(s.replace(b"\t", b" ") for s in src))
    m = fastalign.train(a, b, directions="fwd", iterations=1, threads=2, subword=_small_config())
    assert m.parameters("fwd")["mean_srclen_multiplier"] == 1.0
    assert _ratio_py(a, b) == 1.0
    lists = [[w.decode() for w in sw.side_words(s.rstrip(b"\r\n"))] for s in src]
    lists = [[w for w in x if w != "\r"] for x in lists]
    assert _ratio_py(lists, lists) == 1.0


def test_subword_piece_corpus_keeps_the_core_words(tmp_path):
    """Every word of the word workflow (CR words, tab-separated words) begins exactly one
    marked piece in the piece corpus, so projected indices refer to the core's words; and
    links of the subword run stay inside those word counts."""
    pytest.importorskip("sentencepiece")
    src, tgt = _quirky(150, 21)
    a, b = tmp_path / "a", tmp_path / "b"
    write_bytes(a, b"".join(src))
    write_bytes(b, b"".join(tgt))
    sides = sw.corpus_sides(a, b)
    segs = sw.train_stream_segmenters(sides, _small_config())[:2]
    pieces = tmp_path / "pieces"
    assert sw.write_piece_corpus(str(pieces), sides, segs) == 150
    ws, wt = list(sides[0]()), list(sides[1]())
    for k, line in enumerate(pieces.read_bytes().decode("utf-8").split("\n")[:-1]):
        ps, pt = line.split(" ||| ")
        starts = [sum(p.startswith(sw.MARKER) for p in x.split(" ")) for x in (ps, pt)]
        assert starts == [len(ws[k]), len(wt[k])], k
    m = fastalign.train(a, b, threads=2, iterations=2, subword=_small_config())
    for d in ("fwd", "inv"):
        for k, links in enumerate(m.alignments(direction=d)):
            assert all(i < len(ws[k]) and j < len(wt[k]) for i, j in links), (d, k)
    # the same words without tabs (tabs -> spaces) give the same subword alignments
    a2, b2 = tmp_path / "a2", tmp_path / "b2"
    write_bytes(a2, b"".join(s.replace(b"\t", b" ") for s in src))
    write_bytes(b2, b"".join(t.replace(b"\t", b" ") for t in tgt))
    m2 = fastalign.train(a2, b2, threads=2, iterations=2, subword=_small_config())
    for d in ("fwd", "inv"):
        assert m.alignments(direction=d).to_bytes() == m2.alignments(direction=d).to_bytes()
