"""Platform-independent text I/O for the tests: UTF-8 and "\\n" line ends on every OS.

`Path.write_text` and `Path.read_text` use the platform's default encoding and, on Windows,
translate "\\n" to "\\r\\n", which turns a trailing "\\r" into part of the last token (as in
fast_align) and so changes vocabularies and alignments. These helpers write and read bytes,
so a test that wants CRLF writes it explicitly (see tests/test_edge.py).
"""

from __future__ import annotations

from pathlib import Path


def write_text(path: Path, data: str, encoding: str = "utf-8") -> int:
    """Write ``data`` encoded, without any newline translation."""
    return path.write_bytes(data.encode(encoding))


def read_text(path: Path, encoding: str = "utf-8") -> str:
    """Read ``path`` decoded, without any newline translation."""
    return path.read_bytes().decode(encoding)
