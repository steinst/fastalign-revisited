# Results: fastalign revisited against fast_align (en-fr large and medium)

Step 9b's results come first; the Step 9a measurements (before any optimisation) follow unchanged.

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
