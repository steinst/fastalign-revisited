from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import aer
from textio import write_text

# Hand-computed example.
# Sentence 1: sure {(0,0),(1,1)}, possible adds {(2,1)}; hypothesis {(0,0),(2,1),(3,3)}
#   A&S = {(0,0)} = 1; A&P = {(0,0),(2,1)} = 2; |A| = 3; |S| = 2
# Sentence 2: sure {(0,0)}, no extra possible; hypothesis {} -> contributes |S| = 1 only
# Totals: |A|=3, |S|=3, A&S=1, A&P=2
#   precision = 2/3, recall = 1/3, AER = 1 - (1+2)/(3+3) = 0.5
GOLD = [
    "0001 1 1 S",
    "0001 2 2 S",
    "0001 3 2 P",
    "0002 1 1 S",
    "0002 0 4 P",  # NULL link: ignored
]


def test_hand_computed_example() -> None:
    sure, poss = aer.read_gold(GOLD)
    hyp = [aer.parse_pharaoh("0-0 2-1 3-3"), aer.parse_pharaoh("")]
    r = aer.score(hyp, sure, poss)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(1 / 3)
    assert r["aer"] == pytest.approx(0.5)


def test_perfect_and_empty() -> None:
    sure, poss = aer.read_gold(GOLD)
    perfect = [aer.parse_pharaoh("0-0 1-1"), aer.parse_pharaoh("0-0")]
    assert aer.score(perfect, sure, poss)["aer"] == pytest.approx(0.0)
    empty = [set(), set()]
    r = aer.score(empty, sure, poss)
    assert r["aer"] == pytest.approx(1.0)
    assert r["precision"] == 0.0


def test_sure_link_counts_as_possible() -> None:
    # Hypothesis hits a sure link only: precision must count it as possible too.
    sure, poss = aer.read_gold(["1 1 1 S"])
    r = aer.score([{(0, 0)}], sure, poss)
    assert r == {"precision": 1.0, "recall": 1.0, "aer": 0.0}


def test_length_mismatch_is_error() -> None:
    sure, poss = aer.read_gold(GOLD)
    with pytest.raises(ValueError):
        aer.score([set()], sure, poss)


def test_bad_gold_line() -> None:
    with pytest.raises(ValueError):
        aer.read_gold(["1 2 3"])


def test_cli_offset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gold = tmp_path / "gold"
    write_text(gold, "\n".join(GOLD) + "\n")
    hyp = tmp_path / "hyp"
    write_text(hyp, "5-5\n0-0 2-1 3-3\n\n")  # first line is training junk
    assert aer.main([str(hyp), str(gold), "--offset", "-2"]) == 0
    out = capsys.readouterr().out
    assert "AER 50.00" in out


# --- fail-closed behaviour (Step 16) -------------------------------------------------------------

import subprocess

import numpy as np

SUBWORD = Path(__file__).resolve().parent.parent / "benchmarks" / "subword"
sys.path.insert(0, str(SUBWORD))


def _bootstrap():
    return pytest.importorskip("bootstrap")


TWO = ["1 1 1 S", "2 1 1 S"]  # two single-link gold sentences


def test_read_window_rejects_short_and_surplus() -> None:
    assert aer.read_window(["0-0\n", "0-0\n"], 0, 2) == [{(0, 0)}, {(0, 0)}]
    assert aer.read_window(["9-9\n", "0-0\n", "0-0\n"], -2, 2) == [{(0, 0)}, {(0, 0)}]
    with pytest.raises(ValueError):
        aer.read_window(["0-0\n"], 0, 2)  # short
    with pytest.raises(ValueError):
        aer.read_window(["0-0\n"], -2, 2)  # short for a tail
    with pytest.raises(ValueError):
        aer.read_window(["0-0\n"] * 3, 0, 2)  # surplus after the window
    with pytest.raises(ValueError):
        aer.read_window(["0-0\n"] * 3, -3, 2)  # tail longer than the gold


def _cli(tmp_path: Path, gold: str, hyp: str, *args: str):
    g, h = tmp_path / "gold", tmp_path / "hyp"
    write_text(g, gold)
    write_text(h, hyp)
    script = Path(__file__).resolve().parent.parent / "benchmarks" / "aer.py"
    return subprocess.run(
        [sys.executable, str(script), str(h), str(g), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_fixtures(tmp_path: Path) -> None:
    gold = "\n".join(TWO) + "\n"
    ok = _cli(tmp_path, gold, "0-0\n0-0\n")
    assert ok.returncode == 0 and "AER 0.00" in ok.stdout
    assert _cli(tmp_path, gold, "0-0\n").returncode == 1  # short
    assert _cli(tmp_path, gold, "0-0\n0-0\n0-0\n").returncode == 1  # surplus
    assert _cli(tmp_path, gold, "").returncode == 1  # empty hypothesis
    assert _cli(tmp_path, "", "0-0\n").returncode == 1  # empty gold
    assert _cli(tmp_path, gold, "5-5\n0-0\n0-0\n", "--offset", "-2").returncode == 0


def test_zero_denominator_is_zero() -> None:
    # no links, no sure links: aer.score defines 0, and the bootstrap must agree
    r = aer.score([set()], [set()], [set()])
    assert r["aer"] == 0.0
    bs = _bootstrap()
    c = bs.counts([[]], [set()], [set()])
    assert bs.aer_of(c.sum(0)) == 0.0


def test_bootstrap_counts_validate_lengths() -> None:
    bs = _bootstrap()
    sure, poss = aer.read_gold(TWO)
    # one perfect hypothesis for two gold sentences must not score as 0 % AER
    with pytest.raises(ValueError):
        bs.counts([[(0, 0)]], sure, poss)
    with pytest.raises(ValueError):
        bs.counts([[(0, 0)]] * 3, sure, poss)  # surplus
    with pytest.raises(ValueError):
        bs.counts([[(0, 0)]], [], [])  # gold missing
    c = bs.counts([[(0, 0)], []], sure, poss)  # an explicit empty prediction scores
    assert float(bs.aer_of(c.sum(0))) == pytest.approx(100 * (1 - 2 / 3))
    assert isinstance(c, np.ndarray)


def test_compare_files_tail(tmp_path: Path) -> None:
    pytest.importorskip("bootstrap")
    cf = pytest.importorskip("compare_files")
    p = tmp_path / "x.pharaoh"
    write_text(p, "0-0\n1-1\n2-2\n")
    assert len(cf.tail(str(p), 2)) == 2
    with pytest.raises(ValueError):
        cf.tail(str(p), 4)
