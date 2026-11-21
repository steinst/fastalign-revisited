"""The test harness itself must fail closed (Step 16)."""

from __future__ import annotations

import conftest
import pytest


def test_missing_executable_skips_unless_required(monkeypatch) -> None:
    monkeypatch.setattr(conftest, "REQUIRED", set())
    with pytest.raises(pytest.skip.Exception):
        conftest._missing("native", "binary not built")
    monkeypatch.setattr(conftest, "REQUIRED", {"native"})
    monkeypatch.setenv("FASTALIGN_REQUIRE", "native")
    with pytest.raises(pytest.fail.Exception, match="FASTALIGN_REQUIRE"):
        conftest._missing("native", "binary not built")
    with pytest.raises(pytest.skip.Exception):  # only the named kind is required
        conftest._missing("reference", "fast_align not built")


def test_native_path_has_exe_suffix_on_windows() -> None:
    assert conftest.NATIVE.name == ("fastalign.exe" if conftest.EXE else "fastalign")
