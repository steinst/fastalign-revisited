"""Subword alignment plumbing (Step 12): identity, parity on pieces, projection, round trip,
edge cases. Needs ``sentencepiece`` (optional extra); parity and corpus tests need the
reference files (``reference/``, not in the repository) and skip without them."""

from __future__ import annotations

import os
import random
import re
import subprocess
from pathlib import Path

import fastalign
import pytest
from fastalign import subword as sw

pytest.importorskip("sentencepiece")

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "target" / "release" / "fastalign"
FAST_ALIGN = ROOT / "reference" / "fast_align-src" / "build" / "fast_align"
CORPORA = ROOT / "reference" / "data" / "corpora"
PROGRESS = re.compile(r"^\.*( \[\d+\])?$")

needs_corpora = pytest.mark.skipif(not CORPORA.exists(), reason="reference corpora not present")
needs_fast_align = pytest.mark.skipif(
    not (FAST_ALIGN.exists() and NATIVE.exists()), reason="fast_align or native binary missing"
)


def words(text: str) -> list[list[bytes]]:
    return [line.encode().split() for line in text.split("\n")]


def corpus_lines(pair: str, size: str) -> tuple[list[list[bytes]], list[list[bytes]]]:
    return (
        sw.read_lines(str(CORPORA / pair / f"{size}.src")),
        sw.read_lines(str(CORPORA / pair / f"{size}.tgt")),
    )


@pytest.fixture(scope="module")
def tiny_models() -> dict:
    out = {}
    for pair in ("enfr", "roen"):
        src, tgt = corpus_lines(pair, "tiny")
        seg = sw.train_segmenters(src, tgt, sw.SubwordConfig(vocab_size=3000))[0]
        out[pair] = (src, tgt, seg)
    return out


# -- identity ---------------------------------------------------------------------------------


@pytest.mark.parametrize("marker", [False, True])
def test_identity_segmentation_equals_word_level(corpus, marker):
    src, tgt, _ = corpus
    s_lines = [[w.encode() for w in line] for line in src]
    t_lines = [[w.encode() for w in line] for line in tgt]
    seg = sw.WholeWords(marker=marker)
    es, et = sw.encode(s_lines, seg), sw.encode(t_lines, seg)
    assert all(m == array_range(len(m)) for m in es.word_of)

    word = fastalign.train(src, tgt, threads=1)
    piece = fastalign.train(es.pieces, et.pieces, threads=1)
    for direction, by in (("fwd", "tgt"), ("inv", "src")):
        a = word.alignments(direction=direction)
        b = sw.project_many(piece.alignments(direction=direction), es, et, by=by)
        assert [a.pharaoh(i) for i in range(len(a))] == [
            " ".join(f"{i}-{j}" for i, j in line) for line in b
        ]
    for method in fastalign.METHODS:
        a = word.alignments(symmetrize=method)
        pf = sw.project_many(piece.alignments(direction="fwd"), es, et, "tgt")
        pi = sw.project_many(piece.alignments(direction="inv"), es, et, "src")
        c = fastalign.symmetrize_many(pf, pi, method)
        assert [list(a[i]) for i in range(len(a))] == c
        # symmetrise-then-project is the same for identity maps
        d = sw.project_many(piece.alignments(symmetrize=method), es, et, "src")
        assert [list(a[i]) for i in range(len(a))] == d


def array_range(n):
    from array import array

    return array("I", range(n))


# -- parity on the piece corpus -----------------------------------------------------------------


@needs_corpora
@needs_fast_align
@pytest.mark.parametrize("pair", ["enfr", "roen"])
def test_parity_on_pieces(pair, tiny_models, tmp_path):
    src, tgt, seg = tiny_models[pair]
    es, et = sw.encode(src, seg), sw.encode(tgt, seg)
    assert sum(len(p) for p in es.pieces) > sum(es.n_words)  # really segmented
    path = tmp_path / "pieces.txt"
    sw.write_pair_file(str(path), es, et)

    g = subprocess.run(
        [str(FAST_ALIGN), "-i", str(path), "-d", "-o", "-v"],
        capture_output=True,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert g.returncode == 0
    prefix = tmp_path / "ours"
    subprocess.run(
        [str(NATIVE), "train", str(path), "-o", str(prefix), "--directions", "fwd", "--log"]
        + ["-q", "-t", "1"],
        check=True,
    )
    assert (tmp_path / "ours.fwd.pharaoh").read_bytes() == g.stdout

    def stats(text: str) -> list[str]:
        return [
            ln for ln in text.split("\n") if not ln.startswith("ARG=") and not PROGRESS.match(ln)
        ]

    ours = (tmp_path / "ours.fwd.log").read_text()
    assert stats(g.stderr.decode()) == stats(ours)


# -- projection ---------------------------------------------------------------------------------


def random_line(rng: random.Random, max_words: int = 8) -> list[int]:
    """Piece -> word map of one line: words of 1-4 pieces."""
    m: list[int] = []
    for w in range(rng.randint(0, max_words)):
        m.extend([w] * rng.randint(1, 4))
    return m


def random_links(rng: random.Random, n: int, m: int) -> list[tuple[int, int]]:
    if n == 0 or m == 0:
        return []
    return [(rng.randrange(n), rng.randrange(m)) for _ in range(rng.randint(0, 3 * max(n, m)))]


@pytest.mark.parametrize("seed", range(40))
def test_projection_properties(seed):
    rng = random.Random(seed)
    sm, tm = random_line(rng), random_line(rng)
    links = random_links(rng, len(sm), len(tm))
    out = sw.project(links, sm, tm)

    # every projected link points at words that exist
    n_src = max(sm, default=-1) + 1
    n_tgt = max(tm, default=-1) + 1
    assert all(0 <= i < n_src and 0 <= j < n_tgt for i, j in out)
    # no duplicates, sorted by target then source
    assert out == sorted(set(out), key=lambda link: (link[1], link[0]))
    # exactly the words that had a linked piece
    assert set(out) == {(sm[p], tm[q]) for p, q in links}
    # a word whose pieces are all unlinked stays unlinked
    linked_src = {sm[p] for p, _ in links}
    assert {i for i, _ in out} == linked_src
    # projection of the inverse is the inverse of the projection
    inv = sw.project([(q, p) for p, q in links], tm, sm, by="src")
    assert set(inv) == {(j, i) for i, j in out}
    assert inv == sorted(inv)
    # by="src" is the same set
    assert set(sw.project(links, sm, tm, by="src")) == set(out)


def test_projection_errors_and_empties():
    assert sw.project([], [], []) == []
    assert sw.project([], [0, 0], []) == []
    with pytest.raises(ValueError):
        sw.project([(2, 0)], [0, 0], [0])
    with pytest.raises(ValueError):
        sw.project([(0, 1)], [0], [0])
    with pytest.raises(ValueError):
        sw.project([(-1, 0)], [0], [0])
    with pytest.raises(ValueError):
        sw.project([(0, 0)], [0], [0], by="both")
    with pytest.raises(ValueError):
        sw.project_many(
            [[]], sw.encode([[]], sw.WholeWords()), sw.encode([[], []], sw.WholeWords())
        )


def test_projection_merges_and_orders():
    # source word 0 = pieces 0,1; word 1 = piece 2.  target word 0 = piece 0; word 1 = pieces 1,2
    sm, tm = [0, 0, 1], [0, 1, 1]
    assert sw.project([(1, 2), (0, 1), (2, 0), (0, 2)], sm, tm) == [(1, 0), (0, 1)]
    assert sw.project([(1, 2), (0, 1), (2, 0)], sm, tm, by="src") == [(0, 1), (1, 0)]


# -- round trip ---------------------------------------------------------------------------------


@needs_corpora
@pytest.mark.parametrize("pair", ["enfr", "roen"])
def test_round_trip_medium(pair):
    src, tgt = corpus_lines(pair, "medium")
    seg = sw.train_segmenters(src, tgt, sw.SubwordConfig(vocab_size=4000))[0]
    n_split = 0
    for side in (src, tgt):
        enc = sw.encode(side, seg)
        for line, pieces, m in zip(side, enc.pieces, enc.word_of):
            assert len(pieces) == len(m)
            assert sw.split_into_words(pieces) == list(m)
            k = 0
            for w, word in enumerate(line):
                group = []
                while k < len(m) and m[k] == w:
                    group.append(pieces[k])
                    k += 1
                assert sw.piece_text(group) == sw.decode_word(word)
                n_split += len(group) > 1
            assert k == len(m)
    assert n_split > 0


def test_round_trip_latin1_and_bytes(tmp_path):
    latin = "le café est fermé , l'été dernier".encode("latin-1")
    utf = "le café est fermé".encode()
    lines = [latin.split(), utf.split(), b"plain words only".split()] * 30
    seg = sw.train_segmenters(lines, lines, sw.SubwordConfig(vocab_size=400))[0]
    enc = sw.encode([latin.split(), b"\xff\xfe caf\xc3 x".split()], seg)
    for line, pieces in zip([latin.split(), b"\xff\xfe caf\xc3 x".split()], enc.pieces):
        rebuilt = sw.split_into_words(pieces)
        for w, word in enumerate(line):
            group = [p for p, i in zip(pieces, rebuilt) if i == w]
            assert sw.piece_text(group) == sw.decode_word(word)


# -- edge cases ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_seg():
    lines = words("the house is big\nthe cat sat on the mat\nla maison est grande\n" * 20)
    return sw.train_segmenters(lines, lines, sw.SubwordConfig(vocab_size=300))[0]


def test_empty_lines_and_sides(small_seg, tmp_path):
    src = [[], words("the cat")[0], []]
    tgt = [words("la maison")[0], [], []]
    es, et = sw.encode(src, small_seg), sw.encode(tgt, small_seg)
    assert es.pieces[0] == [] and es.word_of[0] == type(es.word_of[0])("I")
    assert es.n_words == [0, 2, 0]
    path = tmp_path / "e.txt"
    sw.write_pair_file(str(path), es, et)
    lines = path.read_text().split("\n")
    assert lines[0].startswith(" ||| ") and lines[2] == " ||| " and lines[3] == ""
    if NATIVE.exists():
        subprocess.run(
            [str(NATIVE), "train", str(path), "-o", str(tmp_path / "o"), "-q", "-t", "1"],
            check=True,
        )
    model = fastalign.train(es.pieces, et.pieces, threads=1)
    fwd = model.alignments(direction="fwd")
    out = sw.project_many(fwd, es, et)
    assert out[0] == [] and out[1] == [] and out[2] == []


def test_separator_token_rejected(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"a b\nc ||| d\n")
    with pytest.raises(ValueError, match="'\\|\\|\\|'"):
        sw.read_lines(str(p))


def test_crlf_and_missing_final_newline(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"a b\r\nc d\r\n\r\ne")
    assert sw.read_lines(str(p)) == [[b"a", b"b"], [b"c", b"d"], [], [b"e"]]
    p.write_bytes(b"a\n\n")
    assert sw.read_lines(str(p)) == [[b"a"], []]


def test_word_is_only_the_marker_or_contains_it(small_seg):
    for w in ("▁", "a▁b", "▁▁", "▁x"):
        pieces = small_seg.encode_word(w.encode())
        assert pieces == ["▁" + w]
        assert sw.piece_text(pieces) == w
    enc = sw.encode([["▁".encode(), b"cat", "a▁b".encode()]], small_seg)
    assert set(enc.word_of[0]) == {0, 1, 2}
    assert sw.split_into_words(enc.pieces[0]) == list(enc.word_of[0])


def test_separator_inside_a_word_never_becomes_a_separator_piece(small_seg):
    for w in (b"x|||y", b"|||x", b"x|||", b"||||", b"a|||"):
        pieces = small_seg.encode_word(w)
        assert "|||" not in pieces
        assert sw.piece_text(pieces) == w.decode()


@pytest.mark.parametrize("n", [1000, 5000, 20000])
def test_very_long_word(small_seg, n):
    w = (b"abcde" * n)[:n]
    pieces = small_seg.encode_word(w)
    assert sw.piece_text(pieces) == w.decode()
    assert all(" " not in p for p in pieces)


def test_unseen_characters_round_trip(small_seg):
    for w in ("été", "中文", "\U0001f600", "café"):
        pieces = small_seg.encode_word(w.encode())
        assert sw.piece_text(pieces) == w


def test_no_byte_fallback_unknown_character_round_trips():
    lines = words("the cat sat\nthe mat\n" * 30)
    seg = sw.train_segmenters(lines, lines, sw.SubwordConfig(vocab_size=100, byte_fallback=False))[
        0
    ]
    pieces = seg.encode_word("中".encode())
    # an unknown character comes back as its own surface text, so the word still round-trips
    assert sw.piece_text(pieces) == "中" and pieces[0].startswith("▁")


def test_training_is_reproducible():
    rng = random.Random(3)
    lines = [
        [f"w{rng.randrange(300)}x{rng.randrange(20)}".encode() for _ in range(6)]
        for _ in range(400)
    ]
    for kind in ("unigram", "bpe"):
        a = sw.train_model(lines, 400, kind, True)
        b = sw.train_model(lines, 400, kind, True)
        assert a == b


def test_config_validation():
    with pytest.raises(ValueError):
        sw.SubwordConfig(model_type="wordpiece")
    with pytest.raises(ValueError):
        sw.SubwordConfig(segment="neither")
    with pytest.raises(ValueError):
        sw.SubwordConfig(vocab_size=10)


def test_one_side_only_and_separate_models():
    lines = words("the cat sat on the mat\n" * 40)
    a, b = sw.train_segmenters(lines, lines, sw.SubwordConfig(vocab_size=300, segment="src"))
    assert isinstance(a, sw.SentencePieceSegmenter) and isinstance(b, sw.WholeWords)
    assert b.encode_word(b"cat") == ["cat"]
    a, b = sw.train_segmenters(lines, lines, sw.SubwordConfig(vocab_size=300, joint=False))
    assert a is not b
    a, b = sw.train_segmenters(lines, lines, sw.SubwordConfig(vocab_size=300, joint=True))
    assert a is b
