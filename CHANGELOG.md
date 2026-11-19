# Changelog

## 0.1.0 (unreleased; not published on PyPI)

First version. Build from source (see the README).

### Added
- Rust core (`fastalign-core`), native `fastalign` binary and Python package (`fastalign`,
  PyO3, abi3-py39) implementing fast_align's reparameterised IBM Model 2: diagonal tension (`-d`),
  tension optimisation (`-o`), variational Bayes (`-v`, `-a`), NULL word, `-q`, `-T`, `-I`, `-b`,
  reverse direction, `-s` scores, `-p` probability tables, and `-f` forced alignment from a text
  table.
- Output byte-identical to `OMP_NUM_THREADS=1 fast_align` at 1, 2, 4, 8 and 20 threads
  (Linux/glibc), including the statistics on stderr; symmetrisation identical to `atools -c`
  (intersect, union, grow-diag, grow-diag-final, grow-diag-final-and, invert).
- Deterministic parallel trainer; both directions from one corpus load.
- Versioned binary model file (`.fastalign`): save, load, align new sentences, export fast_align
  tables. Version 2 also stores subword tokenisers; version 1 files still load.
- Python API (`train`, `Model`, `align_with_table`, `symmetrize`, progress callback, Ctrl-C
  support) and a Python command line with the same subcommands as the native binary.
- Subword alignment: SentencePiece segmentation per word (`subword="auto"`, `SubwordConfig`,
  `PreSegmented`; extra `fastalign[subword]`), projection of the links back to words in the Rust
  core, native `--subword-marker` / `--subword-continuation` for pre-segmented text.
- Benchmarks and harnesses: parity (`benchmarks/parity.sh`), edge cases, fuzz targets, subword
  experiments (`benchmarks/SUBWORD.md`), performance results (`benchmarks/RESULTS.md`).

### Performance (en-fr large, 1.13M lines, both directions + grow-diag-final-and)
- 14.4 s at 20 threads, 112 s at 1 thread; fast_align + atools 78 s and 616 s.
- Peak memory 1.52 GB (CLI); fast_align 2.9 GB per direction.

### Known differences from fast_align
- NaN signs on stderr for an empty source side with `-d -o` (see README, Parity).
- A last input line without a trailing newline is symmetrised once (atools does it twice).
- Bit-exactness is defined on Linux/glibc only.
