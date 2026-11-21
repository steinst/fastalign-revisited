# Results: fastalign revisited against fast_align (en-fr large and medium)

Step 9b's results come first; the Step 9a measurements (before any optimisation) follow unchanged.

## Step 24: saved-model inference (2026-11-20)

`benchmarks/infer_latency.py` (the review's case: a two-iteration model on distinct one-word
pairs, saved and loaded; 128 `Model.align` calls vs one `align_many` of 128 lines; median of 3,
`quiet_run --idle-cpus 1.5`, other_cpus 0.15–0.63), ms for all 128 queries:

| model | words | before: align / align_many | after: align / align_many | after, per-query ratio |
|---|---:|---:|---:|---:|
| fwd only (the review's) | 200 | 4.02 / 0.23 | 0.135 / 0.179 | 0.76 |
| fwd only (the review's) | 40,000 | 131.5 / 1.36 | 0.135 / 0.204 | 0.66 |
| both, gdfa | 40,000 | 288.5 / 2.76 | 0.138 / 0.195 | 0.71 |
| both + joint tokeniser, gdfa | 40,000 | 9.20 / 0.51 | 0.225 / 0.367 | 0.61 |
| same, 128 distinct queries (`--distinct`) | 40,000 | – | 0.299 / 1.317 | 0.23 |

Target (align per query within 2× of align_many's): met everywhere. Before, every call rebuilt a
hash map of the whole vocabulary per direction, started a thread and went through Pharaoh text;
now the index is built once per model and one pair is aligned directly. A first version without
a model-level tokeniser cache gave 2.6× for the subword model (SentencePiece costs 2 µs per word,
the binding 0.56 µs per pair), hence the bounded 16,384-word cache for `align`.

`benchmarks/subword_memory.py` (joint tokeniser, 10 batches of 8,000 never-seen words through
`align_many` lists): before, the segmenter cache grew by 16,000 entries per batch (160,000 at
the end), traced Python memory 6.8 → 82.8 MB, RSS 83 → 294 MB; after, nothing is cached
between calls (traced memory flat, RSS 52 → 58 MB). Through `Model.align`: cache ≤ 16,384
entries, traced ≈ 7–8.6 MB, RSS flat at 69 MB. Outputs equal a cold model's.

Native `align MODEL -s grow-diag-final-and -t 20` on en-fr large (1.12 GB model, corpus mapped
once for both directions now): 7.12/7.21/7.22 s before, 7.12/7.25/7.27 s after, peak RSS
1.83–1.88 GB both, interleaved; other_cpus 0.67–0.94 (busier than the 9b cutoff), so only "no
change" is claimed. Output identical to the fwd/rev/gdfa goldens.

## Step 9b, third session (2026-11-19)

Commits `97c5928`, `1927524`, `b661264` and the narrow hash order (PERF-TODO 7) on top of
`6ad7d97`, one candidate each, each after `parity.sh full` 1570/1570. Same machine, corpus,
configuration and scripts as the sessions below (`IDLE_CPUS=1.5`; every run had 0.98–1.5 cores
of other load). **Every output of every run is byte-identical to golden.** Raw data:
`reference/runs/9b/enfr-large/` (`s3-*`, `s3fin-*`, `u16cmp-*`).

### Summary, en-fr large, both directions + grow-diag-final-and (CLI)

| threads | 9a | 9b session 2 | **9b session 3** | peak RSS (session 3) |
|---|---|---|---|---|
| 1 (P-core) | 356.0 s | 193.3 s | **112.3 s** | 1.39 GB |
| 2 | 210.2 s | 99.4 s | 60.5 s | 1.42 GB |
| 4 | 99.4 s | 51.8 s | 32.8 s | 1.45 GB |
| 8 | 53.2 s | 29.0 s | 19.4 s | 1.50 GB |
| 16 | 33.9 s | 19.9 s | 14.1 s | 1.56 GB |
| 20 | 32.0 s | 19.48 s | **14.41 s** (median of 14.24, 14.41, 14.43) | 1.58–1.59 GB; **1.52 GB** with the narrow order (14.43 s) |

These rows were measured before the narrow order, which changes memory only (14.43 vs 14.45 s
interleaved at 20 threads).

Python API (`fastalign.train(path, threads=20)` + three `write_alignments`): 14.5–14.7 s, peak
2.07–2.08 GB; with the narrow order **14.4–14.5 s, peak 1.94–1.96 GB**.

Against fast_align + atools (78.4 s at 20 threads, 615.5 s at 1): **5.4× faster at 20 threads,
5.5× at 1 thread.**

### Targets after session 3

| target | session 2 | session 3 | status |
|---|---|---|---|
| parity at every thread count | met | met (1570/1570 after every commit; all runs above) | **met** |
| ≤ 15 s at 20 threads | 19.48 s | 14.41 s (CLI), 14.4–14.7 s (API) | **met** |
| ≤ 200 s at 1 thread | 193.3 s | 112.3 s | **met** |
| ≥ 12× scaling | 9.9× | 112.3 / 14.41 = **7.8×** | **not met** (see below) |
| ≤ 2 GB peak, both directions | CLI 1.59 GB, API 2.09 GB | CLI 1.52 GB, API 1.94–1.96 GB | **met** |
| ≤ 4 GB per 1M pairs | met | met | **met** |

### What each candidate did (`phases seq`, interleaved against the previous commit)

| # | change | stage effect | kept |
|---|---|---|---|
| 2c | sampled key index: every 16th target id of each row without a dense index (rows ≥ 64 cells, ≈ 9 MB per direction); a search probes the cached samples, then one 16-key block of the row | 20 thr: E-step 4.88 → 3.61 s (−26 %), decode 1.65 → 1.25 s (−22 %), total 19.6 → 16.4 s; 1 thr: E-step −24 %, decode −25 %, total 193.5 → 159.8 s | yes |
| 12 (new) | a line's lookups run in passes with software prefetch (block starts, then offsets and probabilities; dense rows: index entries, then probabilities), so their cache misses overlap | 20 thr: E-step 3.61 → 2.86 s (−21 %), decode 1.25 → 0.97 s (−22 %), total 16.4 → 14.3 s; 1 thr: E-step −34 %, decode −36 %, total 159.8 → 122.3 s | yes |
| 13 (new) | phase B prefetches 8–16 lines ahead in each row's line list (offsets, length pair, target tokens, column sums, prior-table column) | 1 thr: M-step 18.6 → 13.8 s (−26 %), total 122.3 → 112.5 s; 20 thr: 2.53 → 2.42 s (−4 %) | yes (the stage gains −26 % at 1 thread) |
| 7 | hash order as `u16` for rows ≤ 65,536 cells, `u32` for the few longer ones; model file format unchanged (byte-identical files) | memory: CLI 1.59 → 1.52 GB, API 2.07 → 1.94–1.96 GB; time unchanged | yes |
| 2c′ budget | 128 MB dense index instead of 64 MB, on top of 2c | E-step −4 %, decode −5 % at 20 threads | no (< 5 % on the E-step) |
| tuning | sample stride 8 or 32 instead of 16; sampled rows from 16 or 512 cells instead of 64 | stride 8: +4 % total, stride 32: +0.5 %; minimum length: no change | 16 and 64 kept |

Profile before 2c (20 threads, `phases seq`): the binary search of the rows without a dense index
was 42.8 % of all samples; after 2c it was 32 %, and 74 % of those samples sat on the load of the
16-key block (one cache miss per lookup), which is what candidate 12 overlaps.

### Why scaling is 7.8×, not 12×

Per phase (forward direction, `phases seq`, 1 thread pinned vs 20 threads): initial pass 3.22 →
0.33 s (9.7×), E-step 28.0 → 2.86 s (9.8×), decode 8.68 → 1.00 s (8.7×), **M-step 13.8 → 2.41 s
(5.7×)**; plus corpus load (1.06 → 0.19 s) and grow-diag-final-and (0.5 s at 20 threads), which
are serial or nearly so. This session's changes remove cache-miss *latency*, which helps one thread
far more than twenty threads that already share the memory bandwidth: candidate 13 gained 26 % of
the M-step at 1 thread and 4 % at 20, and candidates 2c/12 gained more at 1 thread (−50 % E-step)
than at 20 (−41 %). The M-step's limit was measured in session 1 (`FA_TRACE_B`: jobs balanced, ≈ 3×
per-worker slowdown at 20 threads, memory bound). The 1-thread time fell from 193 to 112 s and the
20-thread time from 19.5 to 14.4 s, so the ratio fell; reaching 12× at today's 1-thread time would
need ≈ 9.4 s at 20 threads.

## Step 9b, second session (2026-11-19)

Commits `56a6535`, `05330ca`, `b5ad759` on top of `070bb23`, one candidate each, each after
`parity.sh full` 1570/1570. Same machine, corpus, configuration and scripts as below
(`bench_cli.sh` / `bench_phases.sh`, now with `IDLE_CPUS=1.5`: the desktop alone uses 1.0–1.2
cores, so the old 1.0 threshold almost never let a run start; every run below had 1.04–1.53
cores of other load, as in the first session). **Every output of every run is byte-identical to
golden.** Raw data: `reference/runs/9b/enfr-large/` (`s1-*`, `s2-*`, `s3-*`, `cli-fin*`).

### Summary, en-fr large, both directions + grow-diag-final-and (CLI)

| threads | 9a | 9b session 1 | **9b session 2** | peak RSS (session 1 → 2) |
|---|---|---|---|---|
| 1 (P-core) | 356.0 s | 208.5 s | **193.3 s** | 1.93 → 1.44 GB |
| 2 | 210.2 s | 102.2 s | 99.4 s | 1.92 → 1.41 GB |
| 4 | 99.4 s | 53.7 s | 51.8 s | 1.95 → 1.45 GB |
| 8 | 53.2 s | 30.3 s | 29.0 s | 1.99 → 1.49 GB |
| 16 | 33.9 s | 21.0 s | 19.9 s | 2.08 → 1.56 GB |
| 20 | 32.0 s | 20.9 s | **19.48 s** (median of 19.39, 19.48, 19.51) | 2.09–2.12 → **1.57–1.59 GB** |

Python API (`fastalign.train(path, threads=20)` + `write_alignments(..., symmetrize=
"grow-diag-final-and")`): 19.5 s, peak **2.09 GB**. The API returns the trained model of both
directions (≈ 1.1 GB), so it keeps the forward model while the reverse direction trains.

Against fast_align + atools (78.4 s at 20 threads, 615.5 s at 1): **4.0× faster at 20 threads,
3.2× at 1 thread.**

### Targets after session 2

| target | session 1 | session 2 | status |
|---|---|---|---|
| parity at every thread count | met | met (1570/1570 after every commit; all runs above) | **met** |
| ≤ 15 s at 20 threads | 20.9 s | 19.48 s | not met (1.30× off) |
| ≤ 200 s at 1 thread | 208.5 s | 193.3 s | **met** |
| ≥ 12× scaling | 10.0× | 193.3 / 19.48 = 9.9× | not met |
| ≤ 2 GB peak, both directions | 2.09–2.12 GB | CLI 1.57–1.59 GB; Python API 2.09 GB | **met for the CLI**; API 5 % over |
| ≤ 4 GB per 1M pairs | met | met | **met** |

### What each candidate did

| # | change | stage effect | kept |
|---|---|---|---|
| 10 | `TrainConfig::keep_models`: the CLI frees each direction's model unless `--save`/`--table` needs it | peak 2.09–2.12 → 1.57–1.59 GB at 20 threads; time unchanged | yes |
| 11 | cached prior tables stored by source position, then `j` (phase B reads one source position over all `j` of a line: contiguous instead of one cache line per `j`) | `phases seq` M-step per direction: 20 thr 3.34–3.54 → 2.74–2.75 s (2 runs each), 1 thr 22.8 → 18.9 s; E-step within noise | yes |
| 9 | phase B walks the dense-index rows (59 % of its work) in line blocks shared by all workers | 20 thr M-step vs 11: 2.75 → 2.62 (16,384 lines), 2.57 (65,536), 2.74 (4,096); 1 thr: 18.9 → 19.8 s (+4 %) | yes, 65,536 lines, **off at 1 thread** |

### Where the remaining time goes (20 threads, per direction, `phases seq` ≈ 9.5 s)

- **E-step 4.5–4.9 s + decode 1.55–1.65 s**: unchanged by this session; lookups outside the
  dense rows and the per-line distinct-word bookkeeping (as session 1).
- **M-step 2.57 s** (was 3.3): still ≈ 2.7× its 1-thread time per worker; line blocking gained
  only 6.5 %, so the shared-cache traffic of the dense rows was not the main cost. The remaining
  phase-B work is the 41 % in the ≈ 130k rows without a dense index, which read random lines.
- **Scaling 9.9×**: the 1-thread time fell more (−15 s) than the 20-thread time (−1.4 s).


## Step 9b: after optimisation (2026-11-18)

Commit `070bb23` (six candidates from `PERF-TODO.md`, one commit each, each after `parity.sh full`
1570/1570). Same machine, corpus and configuration as below. Measured with `benchmarks/bench_cli.sh`
and `bench_phases.sh`, which wait for a quiet machine (`quiet_run.py`: another session was
benchmarking on this machine; every run below had ≤ 1.5 cores of other load, typically 1.1, which
is the desktop) and check every output against golden. **Every output of every run is
byte-identical to `OMP_NUM_THREADS=1 fast_align` and `atools -c grow-diag-final-and`.** Raw data:
`reference/runs/9b/enfr-large/` (gitignored).

### Summary, en-fr large, both directions + grow-diag-final-and (CLI)

| threads | 9a | **9b** | change | peak RSS 9a → 9b |
|---|---|---|---|---|
| 1 (P-core) | 356.0 s | **208.5 s** | −41 % | 2.11 → 1.93 GB |
| 2 | 210.2 s | 102.2 s | −51 % | 2.61 → 1.92 GB |
| 4 | 99.4 s | 53.7 s | −46 % | 2.68 → 1.95 GB |
| 8 | 53.2 s | 30.3 s | −43 % | 2.69 → 1.99 GB |
| 16 | 33.9 s | 21.0 s | −38 % | 2.82 → 2.08 GB |
| 20 | 32.0 s (median of 4) | **20.9 s** (median of 3: 20.74, 20.88, 20.96) | −35 % | 2.71–2.87 → 2.09–2.12 GB |

Against fast_align + atools (78.4 s at 20 threads, 615.5 s at 1): **3.8× faster at 20 threads, 3.0×
at 1 thread**, with less memory for both directions than fast_align uses for one (2.94 GB).

### Targets after 9b

| target | 9a | 9b | status |
|---|---|---|---|
| parity at every thread count | met | met (1570/1570 after every commit; all runs above) | **met** |
| ≤ 15 s at 20 threads | 32.0 s | 20.9 s | not met (1.39× off) |
| ≤ 200 s at 1 thread | 356.0 s | 208.5 s CLI (199.3 s `phases seq`) | not met through the CLI (4 % off) |
| ≥ 12× scaling | 11.1× | 208.5 / 20.9 = 10.0× | not met |
| ≤ 2 GB peak, both directions | 2.71–2.87 GB | 2.09–2.12 GB at 20 threads, 1.92–1.99 GB at 1–8 threads | not met at 16–20 threads (≈ 5 % off) |
| ≤ 4 GB per 1M pairs | met | met | **met** |

Scaling fell from 11.1× to 10.0× because the 1-thread time fell more than the 20-thread time: the
lookup work that was removed scaled well, and what remains at 20 threads is the memory-bound M-step
(below).

### Per phase, `phases seq`, seconds, fwd / rev

| phase | 1 thr 9a | **1 thr 9b** | 20 thr 9a | **20 thr 9b** | candidate |
|---|---|---|---|---|---|
| load | 1.10 | 1.05 | 0.20 | 0.18 | |
| initial_pass | 3.11 / 2.90 | 3.20 / 2.99 | 1.70 / 1.54 | **0.33 / 0.29** | 6 |
| e_step (4 it.) | 108.12 / 108.52 | **54.23 / 52.97** | 8.10 / 7.82 | **4.64 / 4.43** | 1, 2 |
| m_step (4 it.) | 26.49 / 27.67 | **22.79 / 22.57** | 4.20 / 4.11 | **3.38 / 3.31** | 3a |
| decode | 36.31 / 37.85 | **17.47 / 16.85** | 2.82 / 2.77 | **1.59 / 1.53** | 1, 2 |
| direction total | 174.77 / 177.62 | 98.48 / 96.09 | 16.94 / 16.36 | 10.09 / 9.71 | |
| symmetrize + write | 3.58 | 3.59 | 0.56 | 0.56 | |
| **total** | 357.2 | **199.3** | 34.1 | **20.6** | |

### What each candidate did (en-fr large, interleaved runs against the previous commit)

| # | change | stage effect | kept |
|---|---|---|---|
| 1 | each distinct (e, f) pair of a line looked up once (per-line matrix of copied f64s) | 20 thr: E-step 7.85 → 5.8 s, decode 2.7 → 2.1 s per dir.; 1 thr (`-I 3`): E-step ≈ 40 → 26.8 s, decode ≈ 36 → 25 s | yes |
| 2b | lock-step branchless searches of one line (memory-level parallelism) | 1 thr: decode 25.0 → 29.3 s (slower), 20 thr: no change | **reverted** |
| 2c′ | dense target id → offset index for the busiest rows (by phase-B cost), 64 MB per direction; `--no-lookup-index` | 20 thr: E-step 5.94/5.56 → 4.67/4.45 s, decode 2.08/1.90 → 1.67/1.54 s (16 MB: about half the gain; 128 MB: a further −0.3 s E-step per direction) | yes |
| 5 | directions one after the other (`TrainConfig::concurrent_directions` keeps the old mode) | 20 thr: 27.2 → 27.6 s (noise), peak 2.76–2.86 → 2.32–2.34 GB; 2 thr 162.8 → 139.9 s | yes |
| 4 | `malloc_trim(0)` after each direction's setup and after its model is built | peak 2.32–2.34 → 2.03–2.05 GB, time unchanged | yes |
| 6 | `with_max_len(16)` on the initial pass's row loop: the busiest rows have the smallest ids and fell into rayon's first pieces (the loop ran on ~2.4 threads) | initial pass 1.67 → 0.30 s per direction at 20 threads; CLI 24.4 → 22.0 s | yes |
| 3c | fused `[prob, count]` array indexed by target id in phase B (one cache line per addition) | M-step +8 % at 20 threads, −2.5 % at 1 | **reverted** |
| 3a | phase B uses the stored single source position (u16) instead of scanning the line | M-step −15 % at 20 threads, −16 % at 1; +25 MB | yes |

### Where the remaining time goes (20 threads, per direction, 10 s)

- **E-step 4.6 s + decode 1.6 s**: lookups into rows without a dense index (the 56 % of distinct-pair
  lookups outside the 64 busiest rows: `table_stats`) and the per-line distinct-word bookkeeping. A
  larger index (128 MB) gives ≈ −0.3 s per direction for +64 MB, which the 2 GB target does not
  allow at 20 threads.
- **M-step 3.3 s, memory bound.** `FA_TRACE_B=1` shows all 20 workers busy for the whole phase
  (0.91–1.05 s per iteration, no long job, no idle tail), yet 18.5 worker-s against ≈ 5.7 s at 1
  thread: each worker runs ≈ 3× slower when all run. Phase B visits ≈ 23M (row, line) pairs per
  iteration in random line order and reads each line's target tokens, column sums and prior table
  (≈ 9 GB per iteration); the cell-side arrays are not the problem (candidate 3c).
- **Memory 2.09–2.12 GB at 16–20 threads** (1.92–1.99 GB at ≤ 8): the finished forward model
  (546 MiB) and alignment text are held while the reverse direction trains (trainer ≈ 0.9 GiB),
  plus the corpus and per-thread scratch.

## Step 9a baseline

Measured 2026-11-18 (Step 9a), before any optimisation (commit `ea122e5` plus measurement code:
per-phase timers in `train.rs` that change no output, the `phases` and `table_stats` examples, and the
scripts below). Machine: a 20-core desktop CPU (8 P-cores = CPUs 0–7, 12 E-cores = CPUs 8–19, no SMT),
125 GB RAM, Linux, glibc. Corpus en-fr large: 1,130,551 lines, vocabulary 130,295 (one shared dictionary),
34.8M table cells per direction. Configuration: fast_align's recommended `-d -o -v`, 5 iterations, for
both tools.

**Every output of every run below is byte-identical to `OMP_NUM_THREADS=1 fast_align` (fwd, rev) and to
`atools -c grow-diag-final-and`** (checked by `run_scaling.sh` after each run: 36 checks on large, 36 on
medium, plus the repeats and profiled runs).

How it was measured:

- `benchmarks/run_scaling.sh CORPUS "THREADS"` runs, per thread count, (a) the `phases` example in `seq`
  mode: corpus load, the forward direction trained alone, then the reverse direction alone, grow-diag-final-and,
  writing the three files, printing the wall time of every phase (`Trainer::phase_times()`), and (b) the real
  CLI, `fastalign train INPUT -o P -t N -s grow-diag-final-and` (both directions concurrently, as users run it).
  Both under `benchmarks/timed_run.py` (copied from gizapp: timestamps stderr lines, samples RSS every
  0.2 s, exact peak from `getrusage`). Raw data: `reference/runs/9a/<corpus>/` (gitignored).
  Tables generated by `benchmarks/collect_phases.py`.
- 1-thread runs: the unpinned `phases` run took 390 s and was partly on an E-core (the 2-thread run was
  more than twice as fast). The 1-thread numbers below are therefore runs **pinned to a P-core**
  (`taskset -c 2`), for both tools. Unpinned, the CLI took 351 s (pinned 356 s).
- fast_align: same input file, `OMP_NUM_THREADS=N fast_align -i INPUT -d -o -v [-r]`, stderr timestamped by
  `timed_run.py` (its `INITIAL PASS` / `ITERATION n` lines give per-phase times). atools gdfa: 6.6 s
  (gizapp's baseline, not re-run).
- One run per configuration, except the CLI at 20 threads (4 runs: 35.1, 31.4, 32.6, 31.4 s; **median
  32.0 s**). The machine was otherwise idle (desktop session; 1-minute load ≈ 2–3 before runs).
- Phase definitions (`train::PhaseTimes`): `initial_pass` = table structure, hash order, `size_counts`;
  `setup` = length pairs, per-row line lists; `prior` = per-iteration prior cache; `e_step` = phase A of the
  4 training iterations (posterior column sums, statistics); `m_step` = phase B (counts in corpus order fused
  with normalisation); `tension` = tension optimisation; `decode` = final iteration (Viterbi + Pharaoh text).

### Summary, en-fr large, both directions + grow-diag-final-and

| tool | threads | wall | peak RSS |
|---|---|---|---|
| **fastalign revisited** (`fastalign train … -s grow-diag-final-and`) | 20 | **32.0 s** (median of 4; 31.4–35.1) | 2.71–2.87 GB |
| fastalign revisited | 1 (P-core) | 356.0 s | 2.11 GB |
| fast_align fwd + rev + atools | 20 | 35.1 + 36.7 + 6.6 = 78.4 s | 2.94 GB per direction |
| fast_align fwd + rev + atools | 1 (P-core) | 302.8 + 306.1 + 6.6 = 615.5 s | 2.94 GB per direction |

fastalign revisited is **2.4× faster at 20 threads and 1.7× faster at 1 thread** than fast_align run once
per direction, with less peak memory for both directions than fast_align needs for one, and its output is
the same at every thread count (fast_align's is not, `BASELINE.md`).

## Targets (the project plan's "Targets")

| target | measured | status |
|---|---|---|
| parity: byte-identical at every thread count | every output of every run identical to the golden files (1, 2, 4, 8, 16, 20 threads, large and medium) | **met** |
| both directions + gdfa ≤ 15 s at 20 threads | 32.0 s (median) | **not met** (2.1× off) |
| ≤ 200 s at 1 thread | 356.0 s | **not met** (1.8× off) |
| ≥ 12× speed-up 1 → 20 threads | 356.0 / 32.0 = 11.1× (10.1× with the single 35.1 s run) | **not met** (close) |
| peak RSS ≤ 2 GB for both directions | 2.71–2.87 GB at 20 threads, 2.11 GB at 1 thread | **not met** |
| hard ceiling 4 GB per 1M pairs (4.5 GB here) | 2.87 GB | **met** |

## Time per phase, en-fr large

### Each direction trained alone (`phases … seq`), seconds, fwd / rev

| phase | 1 thr (P-core) | 2 thr | 4 thr | 8 thr | 16 thr | 20 thr | 1→20 |
|---|---|---|---|---|---|---|---|
| load (once) | 1.10 | 0.63 | 0.37 | 0.26 | 0.20 | 0.20 | 5.5× |
| initial_pass | 3.11 / 2.90 | 2.79 / 2.60 | 2.46 / 2.26 | 2.09 / 1.86 | 1.75 / 1.46 | 1.70 / 1.54 | **1.9×** |
| setup | 0.10 / 0.13 | 0.07 / 0.07 | 0.04 / 0.04 | 0.03 / 0.03 | 0.03 / 0.02 | 0.02 / 0.02 | 5.7× |
| prior (5 it.) | 0.13 / 0.12 | 0.08 / 0.08 | 0.06 / 0.06 | 0.05 / 0.04 | 0.04 / 0.04 | 0.04 / 0.03 | 3.5× |
| **e_step** (4 it.) | 108.12 / 108.52 | 55.23 / 54.84 | 27.31 / 27.23 | 14.04 / 13.88 | 9.04 / 8.29 | 8.10 / 7.82 | 13.6× |
| **m_step** (4 it.) | 26.49 / 27.67 | 13.84 / 14.29 | 7.55 / 7.67 | 4.39 / 4.56 | 3.98 / 3.56 | 4.20 / 4.11 | **6.5×** |
| tension (3 it.) | 0.50 / 0.42 | 0.27 / 0.23 | 0.15 / 0.13 | 0.09 / 0.07 | 0.06 / 0.05 | 0.06 / 0.05 | 8.2× |
| **decode** (final it.) | 36.31 / 37.85 | 18.88 / 18.85 | 9.76 / 9.64 | 4.90 / 4.83 | 3.04 / 2.93 | 2.82 / 2.77 | 13.3× |
| direction total | 174.77 / 177.62 | 91.18 / 90.97 | 47.35 / 47.05 | 25.60 / 25.28 | 17.94 / 16.37 | 16.94 / 16.36 | 10.6× |
| symmetrize gdfa | 3.51 | 1.91 | 1.13 | 0.73 | 0.54 | 0.49 | 7.2× |
| write 3 files | 0.07 | 0.06 | 0.06 | 0.06 | 0.07 | 0.07 | |
| **total** | **357.2** | 184.8 | 96.1 | 52.0 | 35.2 | 34.1 | 10.5× |
| peak RSS (MB) | 1531 | 1505 | 1541 | 1565 | 1632 | 1642 | |

Shares at 20 threads: E-step 47 %, M-step 24 %, decode 16 %, initial pass 9 %, the rest 3 %. At 1 thread:
E-step 61 %, decode 21 %, M-step 15 %, initial pass 1.7 %.

The first E-step iteration does no table lookup (all probabilities are fast_align's initial constant),
so the 108 s are ≈ 6 s for iteration 1 and ≈ 34 s for each of iterations 2–4; the decode (36 s) is the same
lookup work plus the argmax. **The three looked-up E-steps and the decode are ≈ 138 s of the 175 s of one
direction at 1 thread (79 %).**

### End to end through the CLI (both directions concurrently)

| | 1 thr (P-core) | 2 thr | 4 thr | 8 thr | 16 thr | 20 thr |
|---|---|---|---|---|---|---|
| wall (s) | 356.0 | 210.2 | 99.4 | 53.2 | 33.9 | 32.0 (median; 35.1 / 31.4 / 32.6 / 31.4) |
| speed-up | 1.0× | 1.7× | 3.6× | 6.7× | 10.5× | 11.1× |
| peak RSS (MB) | 2114 | 2613 | 2679 | 2685 | 2816 | 2710–2868 |

Concurrent directions gain nothing over one after the other at ≥ 4 threads (99.4 vs 96.1 s at 4,
32.0 vs 34.1 s at 20) and lose at 2 threads (210.2 vs 184.8 s: two working sets share one L3), while
costing ≈ 1.2 GB of peak memory (2.71–2.87 GB vs 1.64 GB at 20 threads; see Memory).

### Scaling limits seen in the table

- **M-step stops scaling at 8 threads**: 4.39 s (8) → 3.98 (16) → 4.20 (20) for fwd. Extra runs at 20
  threads/fwd: no hot-row splitting (`FA_SPLIT_COST=max`) 3.59 s (splitting does not help); 8 threads pinned
  to the P-cores 4.30 s; 12 threads pinned to the E-cores 4.41 s. In the 20-thread profile the M-step takes
  141.6 CPU-s for both directions against 54.2 s at 1 thread (2.6× inflation; E-step 274 vs 217 CPU-s, 1.27×), while CPU
  utilisation in the phase is 92 % (samples / (wall × 20 × 199 Hz)): the threads are busy but each does
  its work much more slowly, consistent with shared-memory contention in phase B's corpus walks (each
  row walks its lines in corpus order: random access to source/target tokens and the 189 MB column-sum
  array).
- **Initial pass 1.9×**: the NULL row (every target token, 23.6M stamp checks) and the largest rows
  are single tasks; at 20 threads this is 3.2 s of the 34.1 s sequential run.
- E-step and decode scale 13.3–13.6×, like fast_align's own scaling of these loops; beyond 8 threads the
  extra threads are E-cores.

### fast_align, per phase (from its timestamped stderr), seconds, fwd / rev

| phase | 1 thr (P-core) | 20 thr |
|---|---|---|
| initial pass (reading, table inserts) | 27.6 / 25.7 | 8.8 / 9.1 |
| iteration 1 | 35.1 / 35.6 | 3.1 / 3.5 |
| iterations 2–4 (each) | 69.6, 64.1, 70.4 / 72.5, 65.7, 70.6 | 6.2, 5.9, 6.2 / 6.5, 6.0, 6.2 |
| final iteration (Viterbi, output) | 34.6 / 34.9 | 2.9 / 2.9 |
| total | 302.8 / 306.1 | 35.1 / 36.7 |

Against fast_align at 1 thread: our load + initial pass is 6.6× faster (1.1 + 3.1 vs 27.6 s), a looked-up training
iteration ≈ 1.7× (≈ 41 s E + M vs ≈ 68 s), the decode about the same (36 s vs 35 s).

## en-fr medium (100,447 lines)

| phase | 1 thr | 2 thr | 4 thr | 8 thr | 16 thr | 20 thr | 1→20 |
|---|---|---|---|---|---|---|---|
| initial_pass | 0.36 / 0.34 | 0.30 / 0.29 | 0.26 / 0.23 | 0.20 / 0.17 | 0.15 / 0.13 | 0.15 / 0.15 | 2.3× |
| e_step | 6.49 / 6.59 | 3.33 / 3.37 | 1.90 / 1.89 | 1.31 / 1.30 | 0.71 / 0.77 | 0.76 / 0.73 | 8.8× |
| m_step | 2.42 / 2.40 | 1.20 / 1.26 | 0.64 / 0.65 | 0.38 / 0.36 | 0.26 / 0.27 | 0.26 / 0.27 | 9.1× |
| decode | 2.05 / 2.05 | 1.10 / 1.08 | 0.62 / 0.64 | 0.40 / 0.39 | 0.25 / 0.27 | 0.29 / 0.27 | 7.4× |
| direction total | 11.61 / 11.64 | 6.11 / 6.17 | 3.54 / 3.52 | 2.36 / 2.31 | 1.44 / 1.50 | 1.52 / 1.47 | 7.8× |
| total (`phases seq`) | 23.7 | 12.5 | 7.2 | 4.8 | 3.1 | 3.1 | 7.6× |
| CLI end to end | 23.4 | 13.5 | 6.8 | 4.1 | 2.8 | 2.6 | 9.1× |
| CLI peak RSS (MB) | 444 | 540 | 566 | 573 | 596 | 611 | |

(1-thread medium runs not pinned.) Medium has only 11 fast_align buffers of 10,000 lines, which bounds the
E-step's parallelism (Step 6 NOTES). fast_align on medium: 25.4 / 25.6 s at 1 thread, 2.94 / 2.91 s at
20 threads, peak 0.66 GB per direction: fastalign revisited is 2.2× (1 thread) and 2.2× (20 threads, with
both directions) faster.

## Memory, en-fr large

Live heap (`memory_bytes()`): corpus 185 MiB; forward trainer 822 MiB (table structure 268 = 34.8M cells
× (4 B target id + 4 B hash order) + row arrays; probabilities 278; column sums 8 B per target token;
per-row line lists, length pairs); reverse trainer 800 MiB. Both trainers + corpus = 1.81 GiB = **1.94 GB**,
already at the 2 GB target before any output exists. A finished direction keeps table + probabilities
(546 MiB) for the model; the three Pharaoh texts are 110 + 98 + 112 MB.

RSS over the 20-thread CLI run (`cli-t20.rss`):

| point in the run | RSS (MB) |
|---|---|
| corpus read (input mapped) | 279 → 1063 (both initial passes running) |
| both trainers allocated, iteration 1 | 2664 |
| iterations 2–4 | 2653–2674 |
| decode, alignment texts building | 2809 (peak 2863 by `getrusage`) |

During training RSS is **≈ 0.7 GB above the live heap** (2.65 GB vs 1.94 GB). The same gap shows with one
direction: `phases seq` at 20 threads holds 1.38 GB during forward training against 1.06 GB live, and at
1 thread the CLI holds 1.37 GB during forward training. This is memory freed after the initial passes
(130k per-row `Vec`s per direction, the postings and `RowLines` build arrays, spread over 20 threads' malloc
arenas) that glibc does not return to the OS. Peak grows with thread count (2.11 GB at 1, 2.61 at 2,
2.86 at 20), which also points at per-thread arenas.

## Where the time goes (profiles)

perf at 199 Hz (20 threads: the CLI run with both directions, `t20-*`, and the `phases seq` run, `t20-seq-*`)
and 499 Hz (1 thread pinned to a P-core, forward direction, `t1-*`); frame-pointer build (`[profile.profiling]`),
inlined frames resolved. Samples are assigned to phases by the functions on their stack
(`benchmarks/profile_phases.py`; time windows do not work because the two directions overlap).
Flamegraphs, collapsed stacks and top-function tables: `benchmarks/profiles/`.

| phase | share of samples t20 / t20-seq / t1 | inclusive breakdown |
|---|---|---|
| E-step | 54.4 / 52.1 / 60.3 % | `Ctx::prob` (table lookup) 93.6 %, of which binary search 56.8 % (`select_unpredictable` 48.6 % self, the branchless probe); `prob` 35.5 % self (the load of the found probability); all arithmetic (prior, feature, divisions, `ln`) < 5 % |
| decode | 17.2 / 16.8 / 21.8 % | lookup 95.3 % (binary search 58.0 %); argmax and text formatting < 2 % |
| M-step | 24.2 / 26.9 / 14.6 % | `accumulate_lines` self 71 % (t1 55 %): count additions with the dense target-id index; the per-(row, line) scan of the source side for the row's positions `eq` + `line` 16 % (t1 28 %); normalisation (digamma, exp) 3 % |
| initial pass | 1.4 / 1.2 / 1.7 % | `ordinary_row_sequence` → `push_new` (stamp array `seen`, 33 % self), `StlOrder` emulation 10 % |
| tension, prior, setup, load, symmetrise, other | < 3 % together | |

Hardware counters (`perf stat`, `reference/runs/9a/enfr-large/perfstat-*.txt`):

- 1 thread, forward direction, P-core: IPC 0.87; 54 % of 18.7G LLC references miss: 10.1G LLC misses
  for ≈ 2.7G table lookups (3 E-steps + decode × 682M cell visits) plus the M-step: **at most ≈ 3.7 cache misses
  per lookup**.
- 20 threads, both directions (`phases seq`): IPC 0.78 (P-cores) / 0.66 (E-cores); 45 % of LLC
  references miss.

Lookup statistics (`cargo run --release --example table_stats -- INPUT [-r]`):

| | fwd | rev |
|---|---|---|
| cell visits per looked-up iteration (source + NULL) | 657.9M + 23.6M | 657.9M + 20.0M |
| **visits repeating an (e, f) pair already looked up in the same line** | **229.6M (34.9 %)** | 229.6M (34.9 %) |
| visits into rows of ≥ 16,384 entries / ≥ 32,768 / ≥ 65,536 | 59.1 / 43.1 / 19.5 % | 59.6 / 46.9 / 0 % |
| M-step source-side scan compares per count addition | 0.71 | 0.92 |

So the run is bound by memory latency of the table lookup: a binary search of ~15–17 dependent probes into
rows of tens of thousands of `u32`, then a dependent load of the probability, ≈ 3.7 misses each, and a
third of the lookups repeat one made a moment earlier in the same line.

## Context: gizapp's comparable stages

gizapp's Model 1 iteration (no diagonal prior, f32 counts with atomic adds; its `RESULTS.md`, same
machine and corpus but deduplicated and length-filtered to 835,665 pairs) took 34.9 s at 1 thread and
2.41 s at 20 before its Step 11b, and 6.97 s / 0.77 s after it, mostly through a per-pair gather cache
(2 B per cell, 1.1 GiB) that replaced the binary search. A looked-up iteration here (E + M, one direction,
on all 1,130,551 lines) is ≈ 41 s at 1 thread and ≈ 3.0 s at 20: the same cost per cell as gizapp before
its cache, and for the same reason (`PERF-TODO.md` 1–2). gizapp's binary search was 52 % of Model 1
inclusive; here it is 57 % of the E-step, with the probability load another 35 %.

The ranked optimisation list is `PERF-TODO.md`.

## Step 25: copies, thread budget, early release

en-fr large (1,130,551 lines), 20 threads, `benchmarks/step25_memory.py` (one process per scenario
under `/usr/bin/time`; `other_cpus` 0.5–1.1 on the shared desktop, one run each, so wall times
differ by noise only). *before* = commit c536eb2 (Step 24), *after* = this step. Peak RSS in MB.

| scenario | before | after |
|---|---|---|
| native `train -o` (alignments only) | 1497 | 1477 |
| native `train --table` | **1974** | **1472** |
| native `train --save` | 1993 | 1938 |
| native `train --table --save` | 1990 | 1952 |
| native `train -s gdfa` | 1535 | 1480 |
| native `export-table` from a saved model | 1082 | 1082 |
| native `symmetrize` (files) | 360 | 363 |
| Python API `train` + 3 × `write_alignments` | 1973 | 1959 |
| Python CLI `train -o` | 1965 | 1487 |
| Python CLI `train --table` | **1956** | **1496** |
| Python CLI `train --save` | 1956 | 1945 |
| Python CLI `export-table` | 1099 | 1099 |
| Python CLI `symmetrize` | 449 | 449 |

Wall times are unchanged (12.7–13.5 s for every training scenario in both builds).

* Gain: a table-writing job no longer holds a finished direction's model while the other trains
  (`pipeline::train_with_sink`: the sink writes the table into a staged file, the model is dropped
  and the heap trimmed before the next direction). −0.5 GB. A job that needs both models
  (`--save`, the Python API) is unchanged, by design.
* Not a gain here: directional `write_alignments`, the Python CLI's file output, streaming of
  `write_table` and the projection flush save one text-sized copy each (≈ 100–250 MB of alignment
  text on this corpus), which is below the training peak, so they do not move the peak of these
  scenarios; they matter for a loaded model's export and for corpora where text dominates.
  Not measured separately.
* Worker threads (`-t 3`, max OS threads of the process, sampled every 10 ms): native `train`
  (including the corpus load; `-I 0`) 24 → **4** (3 workers + the main thread); `export-table`
  24 → 4; `align` of a model 24 → 4.

## Step 26: sparse grow-diag, bounded line scratch

### Grow-diag (review performance item 4)

Native `symmetrize` on the en-fr large golden alignments (1,130,551 lines), `benchmarks/symmetrize_time.py`
(`quiet_run`, builds interleaved, every output byte-identical to atools' golden). *old* = Step 25's
dense grid over `[0, max i] × [0, max j]`, *new* = this step. Median of 3, wall s (runs with
other_cpus > 5 dropped from the median where a clean run of the same cell exists).

| method | threads | old | new | peak MB old → new |
|---|---|---|---|---|
| grow-diag | 1 | 3.26 | 2.80–2.92 | 245 → 245 |
| grow-diag | 20 | 0.39 | 0.37 | 358 → 347 |
| grow-diag-final | 1 | 3.50 | 3.04 | 246 → 246 |
| grow-diag-final | 20 | 0.42 | 0.39 | 377 → 367 |
| grow-diag-final-and | 1 | 3.38 | 2.97 | 245 → 245 |
| grow-diag-final-and | 20 | 0.40 | 0.38 | 361 → 351 |

Short dense sentences are 7–13 % faster (one merge instead of sort + binary-search intersection, a
grid over the bounding box only, emission from the union instead of a scan of the grid). A line with
one common link at `(20000, 20000)`, which the old build refused ("alignment indices are too large"),
now gives atools' output; atools itself takes 4–5 s and 296 MB for that file (a 20001² grid).

### Line probability scratch (review performance item 8)

`benchmarks/long_lines.py`: 40 pairs × 3000 tokens per side, `phases seq -d -o -v -b 1`,
`FA_GROUP_LINES=1` (one line per phase-A task). *distinct*: every line a permutation of the same
3000 words per side (3000 × 3000 matrix per line = 69 MiB); *repeated*: 40 words per side. Median
of 3, outputs identical across all runs of a corpus.

| corpus | threads | no cap (old): wall s / peak MB | cap 2²⁰ cells (default): wall s / peak MB |
|---|---|---|---|
| distinct | 20 | 5.39 / 1651 | 6.70 / 426 |
| distinct | 1 | 56.66 / 340 | 54.13 / 271 |
| repeated | 20 | 1.95 / 7 | 1.93 / 7 |
| repeated | 1 | 24.78 / 6 | 24.72 / 6 |

Cap sweep, distinct, 20 threads, one run each: 2¹⁸ 7.71 s / 342 MB, 2²⁰ 6.42 s / 429 MB,
2²² 5.28 s / 916 MB, none 5.41 s / 1650 MB. The default 2²⁰ (8 MiB per worker) was chosen by the
user. The 20-thread cost (+24 %) is inherent to target tiles: every tile re-reads all source rows of
the line from a table that no longer fits in cache; at one thread it is not measurable. Lines whose
matrix fits (all of en-fr large) take the unchanged path.

## Step 27: trainer experiments (review performance items 7 and 6)

`benchmarks/trainer_bench.py` (interleaved `phases seq` runs of several builds/settings, each
under `quiet_run --idle-cpus 1.5` and `timed_run`; 1-thread runs pinned to P-core 2; outputs
of every run compared byte for byte with the first run of the corpus and, on en-fr large, with
the golden fwd/rev/gdfa) and `benchmarks/trainer_summary.py` (medians). Three repetitions per
cell; times are **both directions summed**, seconds; peak RSS of the whole `phases` run (MB).
Corpora (`benchmarks/trainer_corpora.py`, seed 27, `-d -o -v`):

- **en-fr large** (1,130,551 lines, default `-b`).
- **short-hv**: 1,000,000 synthetic pairs of 1–6 tokens over a Zipf vocabulary of 1M types per
  side: many tiny phase-A tasks and phase-B rows, vocabulary-sized scratch large against the work.
- **long-rep**: 4,000 synthetic pairs of 150–300 tokens over 3,000 types per side, `-b 100`.
  With the default 4,096-line groups all 40 buffers form one phase-A task, so its E-step and
  decode are serial even at 20 threads (the same in every variant).

Raw data: `reference/runs/step27/b7/runs.tsv`, `b6/runs.tsv` (other_cpus 0.6–1.7 for every
run used; the one 20-thread short-hv run disturbed by a compile, other_cpus 8.3, was dropped
and re-run).

### Item 7: the existing trainer, no extra resident memory

Builds, stacked: *base* = Step 26; *a* = prior tables kept while the tension's bits are
unchanged; *ab* = + phase-B plan cached per thread count, jobs claimed in chunks of ≈ 1/256 of
a worker's share instead of one lock + atomic + two clock reads per job, trace-only work under
`FA_TRACE_B`; *abd* = + phase-A/decode scratch (the two vocabulary-sized stamp arrays of
`LineLookup`) handed between tasks of a phase instead of allocated per 4,096-line task.

| corpus | thr | variant | prior | E-step | M-step | decode | total | peak MB |
|---|---|---|---|---|---|---|---|---|
| short-hv | 20 | base | 0.003 | 0.465 | 1.127 | 0.162 | 3.66 | 849 |
| | | a | 0.001 | 0.461 | 1.135 | 0.163 | 3.68 | 858 |
| | | ab | 0.001 | 0.459 | **0.935** | 0.158 | 3.58 | 810 |
| | | abd | 0.001 | **0.374** | 0.935 | **0.135** | **3.47** | 817 |
| short-hv | 1 | base | 0.000 | 4.450 | 3.283 | 1.537 | 16.67 | 458 |
| | | a | 0.000 | 4.421 | 3.263 | 1.541 | 16.79 | 458 |
| | | ab | 0.000 | 4.431 | **3.041** | 1.525 | 16.55 | 476 → 458 (slim) |
| | | abd | 0.000 | **4.219** | 3.044 | **1.465** | **16.23** | 476 → 458 (slim) |
| long-rep | 20 | base | 0.062 | 7.944 | 0.565 | 1.849 | 10.62 | 222 |
| | | abd | 0.054 | 7.950 | 0.566 | 1.875 | 10.62 | 224 |
| long-rep | 1 | base | 0.178 | 7.884 | 6.535 | 1.856 | 17.21 | 175 |
| | | a | **0.146** | 7.850 | 6.549 | 1.846 | 17.17 | 175 |
| | | abd | 0.148 | 7.857 | 6.568 | 1.837 | 17.17 | 175–194 (see below) |
| en-fr large | 20 | base | 0.067 | 5.209 | 4.402 | 1.787 | 12.95 | 1691 |
| | | a | **0.057** | 5.327 | 4.464 | 1.782 | 12.99 | 1707 |
| | | abd | 0.064 | 5.202 | 4.393 | 1.846 | 12.92 | 1711 |
| en-fr large | 1 | base | 0.253 | 55.334 | 26.296 | 17.314 | 110.47 | 1540 |
| | | a | **0.207** | 55.458 | 26.217 | 17.359 | 110.44 | 1540 |
| | | abd | 0.205 | 55.336 | 26.180 | 17.345 | 110.27 | 1533 |

(ab and abd rows of long-rep and en-fr large are within noise of a; all rows in the TSV.)

- **Prior reuse (kept):** −18 % on the prior stage at 1 thread (en-fr large 0.253 → 0.207 s,
  long-rep 0.178 → 0.146 s): with `-o` the first two iterations share a tension; without `-o`
  every iteration does. A small stage, so no visible change in total.
- **Phase-B plan (kept):** M-step −17 % at 20 threads and −7 % at 1 thread on short-hv (393k–431k
  jobs per direction, mostly tiny rows); no change where rows are few and long. The first version
  stored the plan as 32-byte job descriptors plus a cost per job: +18 MB peak at 1 thread on
  short-hv. Stored as one `u32` per job (row id, or a part number with the top bit), the peak is
  back to 457.9 MB (= base; three runs each) at the same speed, and the per-job costs are only
  computed for `FA_TRACE_B`.
- **Scratch reuse (kept):** short-hv E-step −19 % at 20 threads (−5 % at 1), decode −15 % (−4 %);
  the 245 tasks per iteration each zero-faulted two vocabulary-sized arrays (≈ 1.2M ids). No change
  elsewhere. A phase holds at most one scratch per running task, as before. The 175 vs 194 MB peaks
  of long-rep at 1 thread are not this change: the peak is at the very end of the run (writing),
  and base gives 192 MB too when run outside the benchmark harness (`timed_run` alone, twice).
- With all three, en-fr large is unchanged (12.92 vs 12.95 s at 20 threads, 110.3 vs 110.5 s at
  1): its rows are few and long, its tasks large, and its prior a small stage.

### Item 6: the opt-in cell-offset cache (`--offset-cache MB`, `offset_cache_mb`)

The first E-step after the first M-step stores, for each line in corpus order while the budget
lasts, the table cell (`u32`) of every distinct pair it looks up (the line's matrix, then the
NULL row's cells); later E-steps and the decode read the probabilities through those offsets
(prefetched 8 ahead) instead of the dense index / sampled search. Column-mode lines (Step 26)
are never cached. Budget per direction; 0 (default) = off, nothing is allocated. *hit* = share
of the distinct pairs looked up per pass that the cache covers. Build *abd* + the cache:

| corpus | thr | budget | hit | cache MiB/dir | setup | E-step | decode | total | peak MB |
|---|---|---|---|---|---|---|---|---|---|
| en-fr large | 20 | 0 | – | – | 0.122 | 5.270 | 1.764 | 12.95 | 1694 |
| | | 256 MiB | 0.15 | 256 | 0.144 | 5.002 | 1.762 | 12.68 | 1951 |
| | | 1 GiB | 0.60 | 1026 | 0.153 | 4.405 | 1.276 | 11.72 | 2749 |
| | | all (16 GiB) | 1.00 | 1723/1716 | 0.152 | **3.480** | **0.793** | **10.34** | 3409 |
| en-fr large | 1 | 0 | – | – | 0.350 | 55.244 | 17.351 | 110.26 | 1537 |
| | | 256 MiB | 0.15 | 256 | 0.486 | 52.849 | 15.874 | 106.50 | 1814 |
| | | 1 GiB | 0.60 | 1026 | 0.504 | 44.662 | 11.287 | 93.96 | 2580 |
| | | all | 1.00 | 1723/1716 | 0.509 | **37.530** | **7.021** | **82.41** | 3266 |
| short-hv | 20 | 0 | – | – | 0.149 | 0.369 | 0.135 | 3.33 | 816 |
| | | ≥ 256 MiB | 1.00 | 73/74 | 0.235–0.247 | 0.251–0.256 | 0.066 | 3.25–3.70 | 898–940 |
| short-hv | 1 | 0 | – | – | 0.145 | 4.236 | 1.477 | 16.27 | 476 |
| | | ≥ 256 MiB | 1.00 | 73/74 | 0.187 | 2.339–2.355 | 0.453–0.461 | 13.29–13.40 | 550 |
| long-rep | 20 | 0 | – | – | 0.038 | 7.898 | 1.860 | 10.57 | 223 |
| | | ≥ 256 MiB | 1.00 | 155 | 0.050 | 7.863–7.870 | 1.740–1.767 | 10.48–10.50 | 377–382 |
| long-rep | 1 | 0 | – | – | 0.057 | 7.852 | 1.835 | 17.14 | 194 |
| | | ≥ 256 MiB | 1.00 | 155 | 0.064 | 7.804–7.819 | 1.756–1.764 | 17.04–17.05 | 348 |

M-step and prior are unchanged in every row (phase B does not use the cache).

- **en-fr large, whole corpus cached (≈ 1.7 GB per direction): −25 % at 1 thread (110.3 → 82.4 s),
  −20 % at 20 threads (12.95 → 10.34 s); peak 3.27 / 3.41 GB** — over the default path's 2 GB
  target, under the 4 GB per 1M pairs ceiling (4.5 GB here). The gain is close to linear in the
  hit rate (1 GiB: 60 % of the pairs, −15 % / −9.5 %). It is largest where lookups dominate a line
  (at 1 thread, en-fr large: E-step −32 %, decode −60 %; short-hv: E-step −44 %, decode −69 %)
  and nil where a line's time goes into its arithmetic (long-rep: −1 %, +155 MB).
- Setup costs +0.15 s at 1 thread (counting each line's distinct words); the first cached
  iteration costs nothing extra measurable (the offsets are written as the lookups happen).
- **Kept as opt-in only**, as the plan requires: `--offset-cache MB` in both CLIs and
  `train(offset_cache_mb=...)`; the default (0) path allocates nothing and is the *abd* build.
  The parity suite was also run with the cache forced on (4 GiB, every line cached).

### End to end through the native CLI (en-fr large, both directions + gdfa)

`bench_cli.sh` (`fastalign train INPUT -o P -t T -s grow-diag-final-and`, `quiet_run
--idle-cpus 1.5`, outputs identical to golden in every run); *base* = Step 26, *final* = all
kept candidates (cache off, the default), *cache* = final with `--offset-cache 4096`.

| threads | base | final | final + cache |
|---|---|---|---|
| 20 (wall, 3 runs) | 13.16 / 13.17 / 13.29 s | 12.99 / 13.09 / 13.11 s | **10.46 / 10.48 / 10.51 s** |
| 20 (peak) | 1454–1475 MB | 1432–1455 MB | 3141–3148 MB |
| 1, P-core (wall) | 110.7 / 110.8 / 112.1 s | 110.99 / 111.0 / 111.4 s | **83.0 s** |
| 1 (peak) | 1290 MB (×3) | 1369–1370 MB (×3) | 3004 MB |

The default path stays under the 2 GB target (≤ 1.46 GB at 20 threads, ≤ 1.37 GB at 1). The
1-thread peak of *final* is 80 MB above *base*, but it is not memory any candidate holds: RSS is
lower than base's throughout training (RSS timelines in `reference/runs/9b/enfr-large/
cli-s27-*-t1.rss`), and both peaks are the last 3 s of the inverse direction's decode, while
its output text grows. The 1-thread peak of each stacked commit: prior reuse alone 1354 MB
(it allocates nothing new: the same tables are rebuilt less often), + plan 1345 MB, + scratch
1370 MB; a decode pool per output window gave 1343 MB. The size of that spike depends on where
the growing output buffer lands in the heap (whether glibc can grow it in place), not on what
is live; the change was not kept.
