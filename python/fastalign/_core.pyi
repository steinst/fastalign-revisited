"""Type stubs for the native module (internal)."""

import os
from collections.abc import Iterator
from typing import Callable, overload

from typing_extensions import TypeAlias

_Link: TypeAlias = tuple[int, int]
_Path: TypeAlias = str | os.PathLike[str]

def version() -> str: ...

class Alignments:
    """A read-only sequence of alignments: one list of ``(source, target)`` links per line."""

    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> list[_Link]: ...
    @overload
    def __getitem__(self, index: slice) -> list[list[_Link]]: ...
    def __iter__(self) -> Iterator[list[_Link]]: ...
    def pharaoh(self, i: int) -> str:
        """Pharaoh-format line of alignment ``i`` (no score)."""
    def to_list(self) -> list[list[_Link]]:
        """All alignments as a list of lists."""
    def to_bytes(self) -> bytes:
        """The Pharaoh text, as :meth:`write` writes it."""
    def write(self, path: _Path) -> None:
        """Write the alignments in Pharaoh format."""

class Model:
    @property
    def directions(self) -> list[str]: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def has_training_alignments(self) -> bool: ...
    @property
    def n_lines(self) -> int | None: ...
    @property
    def training_warnings(self) -> list[str]: ...
    @property
    def subword(self) -> tuple[str, str, list[bytes], bytes] | None:
        """``(kind, marker, tokenizer models, settings)`` of a model trained on subword pieces."""
    def parameters(self, direction: str) -> tuple[float, float, int, float]: ...
    def training_log(self, direction: str) -> bytes: ...
    def save(self, path: _Path) -> None: ...
    @staticmethod
    def load(path: _Path) -> Model: ...
    def write_table(
        self, path: _Path, direction: str = "fwd", beam_threshold: float = -4.0, threads: int = 0
    ) -> None: ...
    def training_alignments(
        self,
        symmetrize: str | None = None,
        direction: str | None = None,
        pieces: bool = False,
    ) -> Alignments: ...
    def write_training_alignments(
        self,
        path: _Path,
        symmetrize: str | None = None,
        direction: str | None = None,
        pieces: bool = False,
    ) -> None: ...
    def align_many(
        self,
        src: str | list[str],
        tgt: str | list[str] | None = None,
        symmetrize: str | None = None,
        direction: str | None = None,
        threads: int = 0,
        scores: bool = False,
    ) -> Alignments: ...

def train(
    src: str | list[str],
    tgt: str | list[str] | None,
    threads: int,
    directions: str,
    iterations: int,
    diagonal: bool,
    optimize_tension: bool,
    variational_bayes: bool,
    alpha: float,
    tension: float,
    p0: float,
    null: bool,
    buffer_size: int,
    max_len: int | None,
    empty_pairs: str,
    scores: bool,
    progress: Callable[[str, int, int, float, float, float, float, float], None] | None = None,
    lookup_index: bool = True,
    subword: tuple[str, bytes, list[bytes], bytes] | None = None,
    subword_output: bool = False,
    keep_models: bool = True,
    table_files: list[tuple[str, _Path]] | None = None,
    table_beam: float = -4.0,
    offset_cache_mb: int = 0,
) -> Model: ...
def align_table(
    table: _Path,
    src: str | list[str],
    tgt: str | list[str] | None,
    reverse: bool,
    diagonal: bool,
    null: bool,
    p0: float,
    tension: float,
    mean_length_multiplier: float,
    threads: int,
) -> tuple[bytes, bytes, int | None]: ...
def symmetrize_many(
    fwd: list[list[_Link]], inv: list[list[_Link]], method: str, threads: int = 0
) -> list[list[_Link]]: ...
def symmetrize_text(fwd: bytes, inv: bytes | None, method: str, threads: int = 0) -> bytes: ...
def invert_many(fwd: list[list[_Link]]) -> list[list[_Link]]: ...
def parse_pharaoh(line: str) -> list[_Link]: ...
