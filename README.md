# fastalign revisited

A modern, fast reimplementation of [fast_align](https://github.com/clab/fast_align) (Dyer,
Chahuneau and Smith 2013) in Rust, with Python bindings and a native command-line tool.

* **Same alignments as fast_align.** Output is byte-identical to `OMP_NUM_THREADS=1 fast_align`
  with the same options, at every thread count of this tool (see [Parity](#parity)). That
  includes the `-s` scores, the `-p` probability tables, the statistics fast_align prints, and
  the symmetrisation heuristics of `atools`.
* **Faster, and deterministic at any thread count.** About 5.4× faster than fast_align + `atools`
  on a 1.1M-sentence corpus, in about half the memory (1.5 GB against fast_align's 3.0 GB for one direction) (see [Results](#results)). fast_align itself
  gives different output with more than one thread; this tool does not.
* **A model you can keep.** Train once, save, then align new sentences with the saved model
  (fast_align has no such mode other than re-reading its text table).
* **Optional subword alignment**, which helps morphologically rich languages (see
  [Subword alignment](#subword-alignment)).

Version 0.2.0, not released on PyPI (build it from this repository). Sister project: *gizapp*
(giza++ revisited), which has the same command line and Python API shape for GIZA++.

## Install

From a checkout of this repository. You need a Rust toolchain (`rustup`, Rust 1.85 or newer) and
Python 3.9+ for the Python package.

```bash
# Python package (builds the Rust extension with maturin)
uv venv .venv && source .venv/bin/activate        # or: python -m venv .venv
uv pip install .                                  # or: pip install .
uv pip install '.[subword]'                       # adds SentencePiece for subword alignment

# Native binary, no Python needed
cargo install --path crates/fastalign-core        # installs `fastalign` into ~/.cargo/bin
```

For development: `uv pip install -e '.[dev]'`, then `maturin develop --release`, `pytest`, `cargo test --workspace`, `cargo clippy`.

## Quick start (Python)

```python
import fastalign

model = fastalign.train("corpus.en", "corpus.fr")      # -d -o -v, 5 iterations, both directions
model.save("en-fr.fastalign")

for links in model.alignments(symmetrize="grow-diag-final-and"):
    ...                                                # list of (source, target) links

model.write_alignments("out.align")                    # Pharaoh format, one line per input line

model = fastalign.Model.load("en-fr.fastalign")
model.align(["the", "house"], ["la", "maison"])        # [(0, 0), (1, 1)]
model.align_many("new.en", "new.fr", symmetrize="union").write("new.align")
```

Other ways to give the corpus: one file of `source ||| target` lines
(`fastalign.train("corpus.en-fr")`), or lists of sentences, each a string or a list of tokens
(`fastalign.train(src_sentences, tgt_sentences)`). Words are separated by spaces and tabs. In a
`source ||| target` file a tab also separates the sides, exactly as in fast_align; with two files
each file is one side, and a tab or `|||` inside a line stays on that side. Pipes and process
substitutions work as file names; on the command line `-` is standard input.

`train` takes fast_align's options by name: `iterations` (`-I`), `diagonal` (`-d`),
`optimize_tension` (`-o`), `variational_bayes` (`-v`), `alpha` (`-a`), `tension` (`-T`), `p0`
(`-q`), `null` (`-N` is `null=False`), `buffer_size` (`-b`), `directions` (`-r` is
`directions="inv"`), `scores` (`-s`). `threads=0` uses all cores and never changes the output.
Progress goes to the `fastalign` logger and to an optional `callback`; Ctrl-C stops training
within a fraction of a second.

Reproducing fast_align's own files: `model.write_table("t")` writes its `-p` table;
`fastalign.align_with_table("t", src, tgt)` is its `-f` mode; `fastalign.symmetrize` is
`atools -c`.

## Subword alignment

For morphologically rich languages (Romanian-English alignment improves by about 2 AER points in
the measurements of `benchmarks/SUBWORD.md`; English-French does not), align *pieces* of words and
get word alignments back. The output stays word-level Pharaoh, and the default is off.

```python
import fastalign

# trains a SentencePiece tokeniser on the corpus (pip install 'fastalign-revisited[subword]'):
# BPE, one joint model of 8000 pieces
model = fastalign.train("corpus.en", "corpus.fr", subword="auto")
model.write_alignments("subword.align")                # word links, not piece links
model.save("en-fr-subword.fastalign")                  # the tokeniser is stored in the model

model = fastalign.Model.load("en-fr-subword.fastalign")
model.align(["the", "houses"], ["les", "maisons"])     # words in, word links out
```

`subword=fastalign.SubwordConfig(model_type="unigram", vocab_size=4000, joint=False)` changes the
tokeniser; `subword=fastalign.PreSegmented("@@", continuation=True)` is for text you segmented
yourself. On the command line: `fastalign train corpus.en corpus.fr -o out --subword`. The native
binary has no tokeniser and takes pre-segmented text:
`fastalign train pieces.txt -o out --subword-marker ▁` (SentencePiece's marker; a piece starting
with it begins a word) or `--subword-continuation @@` (subword-nmt style).

## Command line

```bash
fastalign train corpus.en corpus.fr -o out -s grow-diag-final-and --save en-fr.fastalign
#   writes out.fwd.pharaoh out.inv.pharaoh out.grow-diag-final-and.pharaoh
fastalign align en-fr.fastalign new.en new.fr -s grow-diag-final-and > new.align
fastalign symmetrize out.fwd.pharaoh out.inv.pharaoh -m union
fastalign export-table en-fr.fastalign --direction fwd -o fwd.table
```

The Python console script and the native binary (`cargo build --release`, then
`target/release/fastalign`) take the same subcommands and options; flag mapping from
fast_align is [below](#fast_align-flags-and-their-equivalents).

## Fast_align flags and their equivalents

| fast_align | fastalign | note |
|---|---|---|
| `-i FILE` | positional `INPUT [TGT]` | one `src ||| tgt` file, or two line-aligned files |
| `-r` | `--directions inv` (default `both`; or `fwd`) | output is in source-target orientation |
| `-d` `-o` `-v` | on by default; `--no-diagonal` `--no-optimize-tension` `--no-variational-bayes` | `-T` implies the diagonal, as in fast_align |
| `-I` `-T` `-a` `-N` `-b` | same short flags (`--iterations --tension --alpha --no-null --buffer-size`) | |
| `-q` | `--p0` | |
| `-s` (scores) | `--scores` | `-s` is `--symmetrize` here |
| `-t` (beam threshold) | `--beam-threshold` | `-t` is `--threads` here |
| `-p FILE` | `--table` (writes `PREFIX.<dir>.table`), or `export-table MODEL` | byte-identical to fast_align's table |
| `-f FILE` | `align --table FILE INPUT [TGT]` | prints fast_align's `-f` output and its stderr lines |
| stderr statistics | `--log` (writes `PREFIX.<dir>.log`) | |
| `atools -c METHOD` | `symmetrize FWD INV -m METHOD` | `intersect`, `union`, `grow-diag`, `grow-diag-final`, `grow-diag-final-and`, `invert` |

Unlike fast_align the tool does not drop long sentences by default (`--max-len` does; dropping
lines changes the result), and the default is fast_align's recommended `-d -o -v` with 5
iterations in both directions.

### Empty sides and other degenerate input

fast_align trains on a line pair with an empty side, and that pair changes the results of **all
the other pairs**: one empty *source* side makes the forward direction's mean length ratio
infinite and, with tension optimisation, its tension NaN, so the forward direction aligns no
line at all; one blank line makes the ratio NaN and every `-s` score NaN. This tool does the same
by default (it is a drop-in), and says so: training that ends with nonfinite parameters prints a
`fastalign: warning:` line (Python: a `RuntimeWarning`, also in `Model.training_warnings`)
naming the empty-sided lines; the `--log` statistics stay fast_align's. Pass `--empty-pairs skip`
(`empty_pairs="skip"`) to drop such pairs before training (each keeps an empty output line, the
rest is what fast_align gives on the file without them), or `--empty-pairs error` to refuse
them with their line numbers.

An empty input trains nothing and writes nothing (as fast_align); an input whose every line is
dropped writes one empty line per input line; both warn. `-I 0` decodes nothing and writes no
lines. A model trained with `-I 1` has no estimated probabilities (its exported table is all
`-inf`, as fast_align's) and aligns new text from the alignment prior alone, as the training
run did. The starting tension must not be 0 (the diagonal prior is then 0/0).

## Results

Measured on one machine (a 20-core desktop CPU with 8 performance and 12 efficiency cores, 125 GB
RAM, Linux), against fast_align commit `cab1e9a` with `-d -o -v`. Corpus: the 1,130,551-line
English-French Hansards (not included, see [Data](#data)). Full tables and the method are in
`benchmarks/RESULTS.md` and `benchmarks/BASELINE.md`.

Both directions plus `grow-diag-final-and`:

| | fast_align + atools | fastalign revisited | |
|---|---|---|---|
| 20 threads, wall time | 78.4 s | **14.4 s** | 5.4× faster |
| 1 thread, wall time (pinned to a performance core) | 615.5 s | **112.3 s** | 5.5× faster |
| peak memory, both directions | 2.9 GB (per direction) | **1.52 GB** (1.94 GB through the Python API, which keeps both models) | |

* **Accuracy is identical to fast_align's** because the output is. Single-threaded fast_align on
  this corpus: AER 16.68 (source to target), 16.25 (target to source), 15.29 with
  `grow-diag-final-and`; the same numbers come out of this tool at 1, 2, 4, 8 and 20 threads.
  (fast_align at 20 threads differs from its own 1-thread output on 1-2 % of the lines.)
* Scaling from 1 to 20 threads is 7.8×; the later phases are bound by memory bandwidth, not by
  the number of cores. The goal of 12× in the project plan was not met (`benchmarks/RESULTS.md`).
* Timing varies by a few percent between runs, and by more if the machine is shared.

### Subword results

Aligning SentencePiece pieces and projecting the links back to words (`benchmarks/SUBWORD.md`;
about 230 configurations, paired bootstrap over the gold sentences):

| corpus | word level (`grow-diag-final-and`) | with subwords (BPE, joint, 8000) | |
|---|---|---|---|
| Romanian-English, 48,689 lines | AER 32.06 | **29.74** | −2.32, 95 % interval −3.14 … −1.43 |
| English-French, 1,130,551 lines | 15.29 | 15.52 | +0.23, within noise |

So: use it for morphologically rich languages, leave it off for English-French. On the large
corpus the alignment step on the prepared pieces takes about 1.1× the time of the word-level
run and a similar amount of memory (1.33 against 1.56 GB peak); training the tokeniser, encoding
and projecting come on top (the tokeniser alone took 12.5 s on 1.1M lines). End to end on the
100,447-line English-French medium corpus (both directions, three word-alignment files, 20
threads) one run took 7.1 s and 505 MB with `subword="auto"` against 3.1 s and 470 MB for words.

## Parity

The target is exact: for every option combination in the test matrix, the output is
**byte-identical** to `OMP_NUM_THREADS=1 fast_align` with the same options, at 1, 2, 4, 8 and 20
threads and every `-b` value. "Output" means the alignment lines, `-s` scores, the `-p` table and
the statistics on stderr; symmetrised output equals `atools -c`. The harness is
`benchmarks/parity.sh` (`quick`: tiny corpora, about a minute; `full`: also the medium and large
corpora, 1570 comparisons) and `benchmarks/edge_parity.py` (edge cases: empty sides, CRLF,
Latin-1, 5000-token sentences, `-I 0`, …). The one documented difference: with an empty *source*
side and `-d -o`, where the tension becomes NaN, fast_align prints some NaNs as `nan` and others as
`-nan`, and this tool sometimes picks the other sign (stdout and the `-p` table are still
identical).

Platform note: bit-exactness is defined against a fast_align running **on the same machine**,
on Linux with glibc, because the model depends on the C library's `exp`, `log`, `pow` and
`lgamma`, and glibc picks different code for them (FMA or not) depending on the CPU. Both tools
call the same library, so they agree with each other; the stored test vectors of the maths
functions (`crates/fastalign-core/tests/data/math`) come from one machine and differ in the last
bits on another (GitHub's runners, or the same CPU with `GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA,-AVX2`).
Hence CI runs them with a 1e-12 tolerance, and `benchmarks/parity.sh`, which compares with a
fast_align you build yourself, is the real check. On macOS, Windows or musl the results are those
of that platform's maths library and may differ in the last bits, and so, rarely, in an alignment
line. Only Linux/glibc on one CPU has been tested for the bit-exact claim; the continuous
integration (GitHub Actions) also builds and tests the project elsewhere:

| platform (CI job) | build + tests | parity with a fast_align built on the same runner |
|---|---|---|
| Linux x86-64, glibc | pass | **identical**: 58/58 training cases and 66/66 forced-alignment cases of `benchmarks/edge_parity.py` |
| Linux x86-64, musl | pass | not run |
| macOS arm64 | pass | 53/58 training cases identical, 5 differ (four `-p` tables and one random corpus), 66/66 forced alignments identical; informational: Apple's C++ library orders hash tables differently from libstdc++ and its maths library differs |
| Windows x86-64 (MSVC) | pass | not run (fast_align does not build there) |

On Windows, files with CRLF line ends keep the `\r` as part of the last token of each line, as
in fast_align.

With subwords off, nothing in training, decoding or output changes. With subwords on, the
aligner is the same one (our output on a piece file equals fast_align's on the same file), but the
scores, tables and saved model are over pieces.

## Data

The corpora used for the measurements (Hansards English-French; the Romanian-English shared task
data) are licensed for research use and are **not** in this repository. The tests use a small
synthetic corpus. Benchmarks and the parity harness need fast_align built under `reference/` and
the corpora under `reference/data/`; the scripts in `benchmarks/` document how.

## Citation and credits

If you use this tool, please cite the model it implements:

> Chris Dyer, Victor Chahuneau and Noah A. Smith. 2013. A Simple, Fast, and Effective
> Reparameterization of IBM Model 2. In *Proceedings of NAACL-HLT 2013*, pages 644-648.

and for subword alignment the tokeniser:

> Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword
> tokenizer and detokenizer for Neural Text Processing. In *Proceedings of EMNLP 2018: System
> Demonstrations*, pages 66-71.

fast_align (Apache-2.0) is by Chris Dyer and contributors, <https://github.com/clab/fast_align>.
No fast_align code is included in this repository; it is the reference for parity testing.

## Licence

MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`). See `NOTICE`.
