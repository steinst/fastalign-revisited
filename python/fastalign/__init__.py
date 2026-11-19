"""fastalign revisited: a fast, byte-exact reimplementation of fast_align word alignment.

Quick start::

    import fastalign

    model = fastalign.train("corpus.en", "corpus.fr")     # -d -o -v, 5 iterations, both directions
    model.save("en-fr.fastalign")

    for links in model.alignments(symmetrize="grow-diag-final-and"):
        ...                                               # list of (source, target) links

    model = fastalign.Model.load("en-fr.fastalign")
    model.align(["the", "house"], ["la", "maison"])       # [(0, 0), (1, 1)]

See :func:`train`, :class:`Model` and :func:`symmetrize`. Progress is reported through the
``fastalign`` logger and an optional callback.
"""

from __future__ import annotations

import logging as _logging

from fastalign._core import version as _version
from fastalign.model import (
    METHODS,
    Alignments,
    Link,
    Model,
    PreSegmented,
    Progress,
    SubwordConfig,
    TableAlignment,
    align_with_table,
    invert,
    parse_pharaoh,
    symmetrize,
    symmetrize_many,
    train,
)

__version__: str = _version()

_logging.getLogger("fastalign").addHandler(_logging.NullHandler())

__all__ = [
    "METHODS",
    "Alignments",
    "Link",
    "Model",
    "PreSegmented",
    "Progress",
    "SubwordConfig",
    "TableAlignment",
    "__version__",
    "align_with_table",
    "invert",
    "parse_pharaoh",
    "symmetrize",
    "symmetrize_many",
    "train",
]
