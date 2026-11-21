"""Normalisation of user-supplied text arguments (internal)."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Union

    #: A sentence as a string (split on whitespace) or as a list of tokens.
    Sentence = Union[str, Sequence[str]]
    #: A path to a text file, or sentences held in memory.
    TextSource = Union[str, os.PathLike[str], Iterable[Sentence]]

# fast_align splits on C whitespace; so do we, and only on that (not on Unicode spaces).
_WHITESPACE = re.compile(r"[ \t\n\r\x0b\x0c]+")
_SEPARATOR = "|||"


def sentence_to_line(sentence: Sentence, where: str) -> str:
    """One sentence as a single-space-joined line, validating tokens.

    A ``str`` is split on ASCII whitespace (so a trailing newline is harmless); a list must
    hold non-empty tokens without whitespace. ``|||`` is the corpus separator of fast_align and
    cannot be a word.
    """
    if isinstance(sentence, str):
        tokens = [t for t in _WHITESPACE.split(sentence) if t]
    elif isinstance(sentence, (bytes, bytearray)):
        raise TypeError(f"{where}: sentences must be str or a list of str, not bytes")
    else:
        try:
            tokens = list(sentence)
        except TypeError:
            raise TypeError(
                f"{where}: a sentence must be a str or a list of str, not {type(sentence).__name__}"
            ) from None
        for tok in tokens:
            if not isinstance(tok, str):
                raise TypeError(f"{where}: tokens must be str, got {type(tok).__name__} ({tok!r})")
            if not tok or _WHITESPACE.search(tok):
                raise ValueError(
                    f"{where}: token {tok!r} is empty or contains whitespace, which would be "
                    "split into several words"
                )
    if _SEPARATOR in tokens:
        raise ValueError(
            f"{where}: '|||' separates the two sides in fast_align and cannot be a word"
        )
    return " ".join(tokens)


class StdinText(bytes):
    """The bytes of standard input, read once by the command line for ``-`` (internal).

    Passed through to the core as raw text in the corpus file format, like a file's content.
    """


def text_source(source: TextSource, name: str) -> str | list[str] | StdinText:
    """A file path (as ``str``), a list of single-space-joined sentences, or standard input
    already read (:class:`StdinText`, from the command line)."""
    if isinstance(source, StdinText):
        return source
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)
    if isinstance(source, (bytes, bytearray)):
        raise TypeError(f"{name} must be a file path or a list of sentences, not bytes")
    try:
        items = list(source)
    except TypeError:
        raise TypeError(
            f"{name} must be a file path or a list of sentences, not {type(source).__name__}"
        ) from None
    return [sentence_to_line(s, f"{name}[{i}]") for i, s in enumerate(items)]
