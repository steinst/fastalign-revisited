from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import aer

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
    gold.write_text("\n".join(GOLD) + "\n")
    hyp = tmp_path / "hyp"
    hyp.write_text("5-5\n0-0 2-1 3-3\n\n")  # first line is training junk
    assert aer.main([str(hyp), str(gold), "--offset", "-2"]) == 0
    out = capsys.readouterr().out
    assert "AER 50.00" in out
