# Changelog

## 0.2.0 (unreleased; not published on PyPI)

Fixes from an external code review, plus
new opt-in options. Output of the default word-alignment workflow is unchanged and still
byte-identical to `OMP_NUM_THREADS=1 fast_align` (`parity.sh full` 1570/1570, edge suite 58/58 +
66/66).

### Migration from 0.1.0
Behaviour changes, each with what to do:
- **Two input files are the two sides** (decision B). A tab or a `|||` inside a line of
  `train SRC TGT` / `align MODEL SRC TGT` / `fastalign.train(src_path, tgt_path)` no longer moves
  the rest of the line to the other side. Files without tabs or `|||` words give identical
  results. To keep the old reading, join the files into one `source ||| target` file (parsing
  unchanged, byte-identical to fast_align).
- **Native `align` on a model with a stored tokeniser fails** (decision A) with an error that
  names the Python command, because the native binary never runs SentencePiece. Use
  `fastalign align` (Python), or pass `--input-segmented` if the input is already segmented with
  the model's tokeniser. Models with only a subword marker, and word models, work as before.
- **An empty corpus trains** in the Python API (with a `RuntimeWarning`), as in the native
  binary and fast_align; `fastalign.train` used to raise `ValueError`. Check
  `Model.training_warnings` or turn the warning into an error if you relied on the exception.
- **Tension** must be finite with magnitude at least 1e-6 when the diagonal prior is on
  (`--tension 0` made every value NaN). Native and Python command lines now validate options
  before reading any input or loading a model: conflicting options (`--symmetrize` with
  `--scores`, `-s invert` with a saved model, a direction of `both` for `align`/`export-table`)
  are errors.
- **`parse_pharaoh` / alignment-file reading** follows atools: links are sorted and
  de-duplicated, only spaces separate them (tabs no longer do), and a line ends at its first CR.
  `invert` removes duplicates.
- **Python `symmetrize`**: no "grid too large" error any more; any size is accepted.
- **Cancellation** is per call: Ctrl-C or an exception in a progress callback stops only that
  call (before, one process-wide flag could end an unrelated concurrent call).
- `-I 0` writes no alignment lines in every combination; `-I 1` models align new text as the
  training run did.
- Saved files (models, tables, alignments) are written atomically through a temporary file and
  a rename; an existing destination that is not a regular file (such as `/dev/stdout`) is
  written directly.

### Added
- `empty_pairs="keep"|"skip"|"error"` / `--empty-pairs` (both command lines): keep fast_align's
  behaviour for line pairs with an empty side (default), skip them (an empty output line each)
  or reject them with their line numbers.
- Warnings when training ends with nonfinite parameters (`RuntimeWarning`; a
  `fastalign: warning:` line on stderr; `--log` statistics unchanged).
- `-` reads standard input in both command lines; pipes, FIFOs, `/dev/stdin` and process
  substitutions are read to the end.
- `--input-segmented` for native `align`; `-t/--threads` for `export-table` and `symmetrize`.
- Opt-in `--offset-cache MB` / `offset_cache_mb`: keeps per-line cell offsets between
  iterations for more memory and shorter training on short, high-vocabulary corpora; no output
  change. Advanced tuning variables `FA_LINE_CELLS` / `ParallelOptions::line_cells`.
- Corpus limits (word ids, lines, tokens per line) reported as errors instead of wrapping.
- Harness: `parity.sh` and the scorers fail closed; new CI jobs (frontends, minimum Rust 1.85,
  minimum Python 3.9); `FASTALIGN_REQUIRE=native,reference` turns skips into failures.

### Changed
- **Two input files are the two sides.** In the two-file form (`train SRC TGT`, `align MODEL
  SRC TGT`, `fastalign.train(src_path, tgt_path)`) a tab or a `|||` inside a line no longer
  moves the rest of the line to the other side: a tab is whitespace and `|||` is an ordinary
  word of its side. Before, the pair was read as the joined line `SRC ||| TGT`, so a source
  line `a<TAB>b c` with target `x y z` trained on source `a` and target `b c ||| x y z`.
  *Migration:* nothing changes for files without tabs or `|||` words (the usual case: results
  are identical). To keep the old reading, join the files yourself into one
  `source ||| target` file, whose parsing is unchanged and byte-identical to fast_align.
- Subword preprocessing (`subword=...`, `--subword`) now reads words and sides exactly as the
  word-level trainer does: only space and tab separate words (a CR stays part of a word, as in
  fast_align), and a joined line splits at its first tab or `|||`. A `|||` word, which subwords
  cannot carry, is an error as before.
- Inputs that are not regular files (pipes, FIFOs, process substitutions, `/dev/stdin`) are read
  to the end; before, the native reader saw them as empty. `-` means standard input for the
  corpus files of `train` and `align` in both command lines (joined or either paired file, at
  most one input); subword training spools such an input once.
- Corpus limits are checked when reading (at most 2^32 − 2 word ids, 2^32 − 2 lines and
  2^32 − 2 tokens per side of a line) and reported as errors instead of wrapping; posting
  arrays are split so that their 32-bit offsets cannot overflow on very large corpora.
- **Line pairs with an empty side** keep fast_align's behaviour by default and can now be
  skipped or rejected: `empty_pairs="keep"|"skip"|"error"` / `--empty-pairs` (both command
  lines). `skip` drops them before reading words (each gets an empty output line), `error`
  lists their line numbers and sides. Training that ends with nonfinite parameters (an empty
  source side makes the forward tension NaN, so that direction aligns nothing; a blank line
  makes every score NaN) is reported as a `RuntimeWarning` in Python and a
  `fastalign: warning:` line on stderr; the `--log` statistics stay fast_align's.
  *Migration:* nothing changes unless you pass the option; to protect a pipeline from one bad
  line, pass `skip`.
- **An empty corpus trains** (nothing) in the Python API as in the native binary and fast_align,
  with a warning; before, `fastalign.train` raised `ValueError("the corpus has no sentence
  pairs")`. An input whose every line is dropped (`max_len`, `skip`) gives one empty line per
  input line, also with a warning. *Migration:* check `Model.training_warnings` or turn the
  `RuntimeWarning` into an error if you relied on the exception.
- `-I 0` writes no alignment lines in every combination (with `max_len` it wrote one empty line,
  with a subword marker it failed); `-I 1` models now align new text as the training run did
  (they aligned nothing, since the table was never estimated; the exported table stays
  fast_align's all `-inf`).
- `--tension`/`tension=` must be finite with magnitude at least 1e-6 when the diagonal prior is
  on (0 made every value NaN, as in fast_align); negative values are accepted by the native
  command line too, as in fast_align.
- **Saved-model inference is faster and its memory bounded.** A model builds its word index once
  (on first use, shared by concurrent callers) instead of once per call and direction; a new
  corpus is mapped to it once for both directions. `Model.align` aligns its pair directly (no
  corpus, thread or Pharaoh text): 128 calls on a 40,000-word model went from 131 ms to 0.14 ms,
  now no slower per pair than one `align_many` call. With a stored tokeniser, every
  `align_many` call encodes with a word cache of its own (bounded, dropped afterwards) and
  `Model.align` shares one bounded cache (16,384 words), so memory no longer grows with the
  number of distinct words a model has seen; files are cut into pieces through temporary piece
  files instead of Python lists. Results are unchanged.

### Performance (en-fr large, 1.13M lines, both directions + grow-diag-final-and)
- Median of three, `quiet_run` (`IDLE_CPUS=1.5`; other load 0.7-4.4 CPUs, all 20 CPUs allowed): 13.6 s at 20
  threads (13.1-15.5), 111.4 s at 1 thread (111.0-116.7). Peak RSS 1.44 GB (20 threads), 1.37 GB (1 thread).
  Outputs identical to golden in every run.

## 0.1.0 (first version; not published on PyPI)

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
  `PreSegmented`; extra `fastalign-revisited[subword]`), projection of the links back to words in the Rust
  core, native `--subword-marker` / `--subword-continuation` for pre-segmented text.
- Benchmarks and harnesses: parity (`benchmarks/parity.sh`), edge cases, fuzz targets, subword
  experiments (`benchmarks/SUBWORD.md`), performance results (`benchmarks/RESULTS.md`).

### Performance (en-fr large, 1.13M lines, both directions + grow-diag-final-and)
- 14.4 s at 20 threads, 112 s at 1 thread; fast_align + atools 78 s and 616 s.
- Peak memory 1.52 GB (CLI); fast_align 2.9 GB per direction.

### Platforms
- CI builds and tests on Linux (glibc and musl), macOS arm64 and Windows; parity with a fast_align
  built on the runner: Linux 58/58 + 66/66, macOS arm64 53/58 + 66/66 (see README).

### Known differences from fast_align
- NaN signs on stderr for an empty source side with `-d -o` (see README, Parity).
- A last input line without a trailing newline is symmetrised once (atools does it twice).
- Bit-exactness is defined against fast_align on the same machine (Linux/glibc; the libm result depends on the CPU's FMA variant). The stored maths test vectors are bit-exact only on the machine that made them; CI compares them with a tolerance.
