# Performance candidates (Step 9a → 9b)

## Status after Step 9b (2026-11-18, commit `070bb23`; numbers in `RESULTS.md`)

| # | candidate | outcome |
|---|---|---|
| 1 | each (e, f) pair once per line | **done**: E-step −25 %, decode −22 % at 20 threads; −33 % / −31 % at 1 |
| 2a | narrowing search over sorted targets | not tried (2c′ replaced the search for the hot rows) |
| 2b | lock-step searches with MLP | **tried, reverted**: slower at 1 thread, no gain at 20 |
| 2c′ | dense target-id index of the busiest rows (64 MB, `--no-lookup-index`) | **done**: E-step −21 %, decode −20 % at 20 threads |
| 3a | stored single source position instead of the scan | **done**: M-step −15 % (20 thr), −16 % (1 thr) |
| 3b | find why the M-step stops scaling | **investigated** (`FA_TRACE_B`): perfectly balanced, memory bound (≈ 3× per-worker slowdown at 20 threads); 3c (fused `[prob, count]` array) tried and reverted (+8 %). Line blocking across rows remains (see 9 below) |
| 4 | return the initial pass's memory | **done** (`malloc_trim`): −0.3 GB peak |
| 5 | directions one after the other | **done**: −0.45 GB peak, same speed at 20 threads, faster at 2 |
| 6 | initial pass load balance | **done** (`with_max_len(16)`; the cause was rayon's piece sizes, not the NULL row): 1.67 → 0.30 s per direction |
| 7 | `u16` hash order | not tried: the remaining route below 2 GB (≈ −130 MB per direction while training) |
| 8 | per-cell lookup cache | not tried: ≈ 0.86 GB per direction, conflicts with the 2 GB target |

## Status after Step 9b session 2 (2026-11-19, commit `b5ad759`)

| # | candidate | outcome |
|---|---|---|
| 9 | M-step line blocking of the dense-index rows | **done**, small: M-step −6.5 % at 20 threads (65,536-line blocks); +4 % at 1 thread, so off there |
| 10 | (new) the CLI frees each finished direction's model unless `--save`/`--table` | **done**: CLI peak 2.09–2.12 → 1.57–1.59 GB |
| 11 | (new) prior tables stored by source position (phase B reads them contiguously) | **done**: M-step −18 % at 20 threads, −17 % at 1 |
| 7 | `u16` hash order | not tried; now only for the Python API's 2.09 GB (≈ −70 MB per direction: 2 B × 34.8M cells, not the 130 MB estimated in 9a) |
| 2c′ budget | 128 MB dense index | not tried; the CLI has the memory now (1.59 GB), the API does not |
| 8 | per-cell lookup cache | not tried (≈ 0.86 GB per direction) |

Remaining candidates, in order (CLI at 20 threads 19.48 s against the 15 s target; scaling 9.9×):

- **2c′ budget** (speed): 128 MB per direction gave ≈ −0.3 s E-step per direction in session 1;
  fits the 2 GB target in the CLI, puts the API at ≈ 2.2 GB.
- **7** (memory, API only): ≈ 2.09 → ≈ 1.95 GB if both the trainer and the kept model use it.
- **E-step and decode** (6.1 s of ≈ 9.5 s per direction at 20 threads) are now the main cost; no
  ranked candidate is left for them that fits the memory target other than 2c′ and 8. Reaching
  15 s would need roughly −2.2 s per direction there.

## Status after Step 9b session 3 (2026-11-19; numbers in `RESULTS.md`)

| # | candidate | outcome |
|---|---|---|
| 2c | sampled key index (every 16th key) of the rows without a dense index | **done**: E-step −26 %, decode −22 % at 20 threads; −24 % / −25 % at 1 |
| 12 | (new) a line's lookups in passes with software prefetch | **done**: E-step −21 %, decode −22 % at 20 threads; −34 % / −36 % at 1 |
| 13 | (new) phase B prefetches the lines ahead in each row's line list | **done**: M-step −26 % at 1 thread, −4 % at 20 |
| 7 | `u16` hash order | **done**: −70 MB per table; API peak 2.07 → 1.95 GB, CLI 1.59 → 1.52 GB |
| 2c′ budget | 128 MB dense index | tried on top of 2c: E-step −4 % (< 5 %), not adopted |
| 8 | per-cell lookup cache | not tried: ≈ 0.86 GB per direction would break the 2 GB target, and after 2c/12 the lookups are ≈ 2.9 + 1.0 s per direction at 20 threads |

All targets but scaling are met (CLI 14.41 s at 20 threads, 112.3 s at 1, peak 1.52 GB; API
1.95 GB). Scaling is 7.8×: the remaining phases are memory-bandwidth bound at 20 threads (M-step
5.7×, E-step 9.8×), and this session's latency fixes helped one thread more than twenty. Left for
anyone who wants more: an M-step that touches less memory per addition (its per-line reads of
target tokens, column sums and prior columns are the traffic), and 8 if memory is allowed to grow.

---

## Original Step 9a list

Ranked by expected wall-time saving on en-fr large, both directions + grow-diag-final-and
(`RESULTS.md`: 32.0 s at 20 threads, 356 s at 1 thread; targets 15 s, 200 s, 12×, 2 GB). Evidence is from
`RESULTS.md` and `benchmarks/profiles/` (`t20-*` CLI both directions, `t20-seq-*` directions one after the
other, `t1-*` forward at 1 thread on a P-core; percentages inclusive of the phase unless marked self).

Ground rules (project plan, Step 9b): one candidate per commit; unit + property tests and `parity.sh full`
after each; re-measure the phase with `phases` / `run_scaling.sh`; revert if the phase gains < 5 % or
memory goes over target. **f32 is not allowed anywhere a posterior, count or probability lives.**

Things checked that are **not** problems: arithmetic of the E-step (prior, `Feature`, divisions, `ln`:
< 5 % of the phase; the prior cache already removes the `exp`s), tension optimisation and prior cache
(< 0.5 % together), output formatting (< 2 % of the decode), writing files (0.07 s), symmetrisation
(0.49 s at 20 threads), corpus load (0.20 s), allocation in per-line loops (malloc not visible),
idle threads (CPU utilisation 85–92 % in every parallel phase).

The arithmetic of every candidate below is untouched: each one changes only *where an operand is fetched
from*, *which thread does a whole unit of work*, or *how memory is allocated*. The values that are added,
the order of every addition, and the per-buffer grouping of the statistics stay exactly as in Step 6.

---

## 1. Look each (e, f) pair up once per line (E-step and decode)

- **Phase:** E-step + decode: 72 % of samples at 20 threads (`t20`), 82 % at 1 thread; 21.4 s of the
  34.1 s `seq` run at 20 threads, 290 s of 357 s at 1 thread.
- **Evidence:** `Ctx::prob` is 93.6 % of the E-step and 95.3 % of the decode. `table_stats`:
  **34.9 % of all cell visits (229.6M of 657.9M per iteration and direction) repeat an (e, f) pair that the
  same line has already looked up** (function words: `the`/`la`, `,`/`,` …).
- **Change:** per line, map source tokens to distinct indices and target tokens to distinct indices
  (per-thread stamp arrays over the vocabulary, as in phase B), look up each distinct pair once into a small
  per-thread `de × df` matrix (row-major by distinct source word, so one row's lookups are adjacent), and let
  `column()` read `P[de(i)][df(j)]` instead of calling `find`. Same in `decode_line`.
- **Expected gain:** −30 % of the lookup phases: ≈ −40 s per direction at 1 thread (−80 s of 356 s),
  ≈ −3 s per direction at 20 threads (≈ −5 s of 32 s).
- **Memory:** a few KB of per-thread scratch.
- **Why no bit can change:** the matrix holds the very `f64` the table holds for that cell (a copy, not a
  computation); `probs[i] = P × a(j, i)`, the NULL term, `sum` and every statistic are formed in the same
  order from the same operands. Before the first M-step the constant is used as before.

## 2. A faster search into long rows (E-step and decode)

- **Phase:** same as 1; what remains of the lookup after it.
- **Evidence:** binary search 56.8 % (E-step) / 58.0 % (decode), mostly `select_unpredictable`, i.e. the
  branchless probe waiting on memory; the load of the found probability another 35 % self. **59 % of visits
  go into rows of ≥ 16,384 entries, 43–47 % into rows ≥ 32,768** (rows of up to 86k–131k target ids): 15–17
  dependent probes, the deeper ones cache misses. `perf stat`: IPC 0.87 at 1 thread, ≤ 3.7 LLC misses per
  lookup.
- **Change (try in this order, keep what measures):** (a) the distinct targets of a line sorted once, then
  each distinct source row searched for them in ascending order, each search starting at the previous hit
  (narrowing range; gizapp found galloping alone did not help, so measure with (b)); (b) interleave the
  independent searches of one line (all distinct pairs of 1) in lock-step with software prefetch of each
  next probe, so that their misses overlap (memory-level parallelism, instead of one chain at a time);
  (c) a per-row two-level index: every 16th key of rows longer than e.g. 1024 in a separate small array, then
  one 64-byte line of 16 keys scanned with SIMD: ~log2(L/16) probes into a 16× smaller, better-cached array
  plus one line.
- **Expected gain:** 1.5–2× on the remaining lookup time: a further ≈ −30 to −45 s per direction at 1
  thread, ≈ −2 to −3 s per direction at 20 threads.
- **Memory:** (a), (b) none; (c) +1/16 of the target-id array ≈ +9 MB per direction.
- **Why no bit can change:** the search returns the same cell index (keys are unique and sorted); only how
  it is found changes.

## 3. M-step: remove the per-(row, line) source scan, then find why it stops scaling at 8 threads

- **Phase:** M-step: 24 % of samples at 20 threads; 8.3 s of the 34.1 s `seq` run at 20 threads, 54 s at
  1 thread.
- **Evidence:** (a) for every (row, line) phase B scans the whole source line for the row's positions:
  `table_stats` gives 0.71 (fwd) / 0.92 (rev) compares per count addition; `eq` + `line` are 28 % self of
  the M-step at 1 thread, 16 % at 20. (b) The M-step does not get faster beyond 8 threads: fwd 4.39 s (8) →
  3.98 (16) → 4.20 (20); 8 P-cores alone 4.30 s, 12 E-cores alone 4.41 s; **2.6× more CPU time at 20
  threads than at 1** (141.6 vs 54.2 CPU-s; E-step 1.27×) at 85–92 % utilisation; hot-row splitting makes
  it slightly slower (3.59 s without). An Amdahl fit of 1, 2 and 8 threads gives ≈ 1.1 s per direction of
  serial-equivalent time.
- **Change:** (a) store, per row, the occurrences as (line, position list) when building `RowLines`
  (positions as `u8`/`u16` runs after each line id, or one `u32` per source token), so phase B never scans a
  source line. (b) Before changing more, instrument phase B per job (wall time of the longest jobs, time
  before the first and after the last job, bytes touched) to separate a critical-path job, the serial job
  building, and memory contention. Then the matching fix: process each row's lines in blocks that are
  shared across rows (better locality of the column-sum and token reads), give the NULL row a cheaper walk
  (it reads every target token), or drop splitting.
- **Expected gain:** (a) −25 % of the M-step at 1 thread (≈ −13 s), −15 % at 20; (b) if the M-step
  scaled like the E-step (13.6×), 8.3 → ≈ 4 s at 20 threads (≈ −4 s sequential, ≈ −2–3 s with both
  directions concurrent).
- **Memory:** (a) + ≈ 1 B per source token if positions replace the scan (≈ +20–80 MB per direction,
  depending on encoding); (b) none expected.
- **Why no bit can change:** each count still receives the same additions in the same (line, j, NULL, i)
  order; (a) only replaces the search for `i` by a stored list of the same `i` in the same order.

## 4. Return the initial pass's transient memory (memory target)

- **Phase:** whole run, memory. Peak 2.71–2.87 GB at 20 threads vs a live heap of 1.94 GB.
- **Evidence:** RSS during training is ≈ 0.7 GB above the live heap with both directions (2.65 vs 1.94
  GB), ≈ 0.32 GB with one (1.38 vs 1.06 GB); peak grows with the thread count (2.11 GB at 1 thread, 2.61 at
  2, 2.86 at 20). The initial pass allocates two `Vec`s per row (130k rows), postings and the `RowLines`
  build arrays on up to 20 threads' malloc arenas, then frees them while the trainers allocate theirs.
- **Change:** build the CSR directly into its final arrays (first pass: row sizes; second pass: fill
  `cols`/`order` in place, per-thread scratch reused across rows), so no per-row `Vec` exists; or, as a
  first quick measurement, `malloc_trim(0)` after both initial passes.
- **Expected gain:** −0.5 to −0.7 GB peak; perhaps a little time (fewer page faults and copies, initial
  pass 3.2 s at 20 threads).
- **Memory:** lower.
- **Why no bit can change:** allocation only; the row contents and hash order are the same.

## 5. Train the two directions one after the other (memory, 2-thread speed)

- **Phase:** whole run.
- **Evidence:** at ≥ 4 threads concurrent directions are no faster than sequential ones (99.4 vs 96.1 s
  at 4, 32.0 vs 34.1 s at 20), at 2 threads they are slower (210.2 vs 184.8 s), and concurrent training
  holds both trainers: peak 2.71–2.87 GB vs 1.64 GB for `phases seq` at 20 threads.
- **Change:** in `pipeline::train`, train the directions sequentially, each with the whole pool (keep the
  concurrent mode as an option if 9b finds it faster after candidates 1–3; with a better-scaling M-step
  and initial pass, concurrency matters less). Free the finished direction's training-only state at once
  (already happens through `into_parts`).
- **Expected gain:** peak ≈ 1.64 GB + the finished model (546 MiB) + texts ≈ 2.2 GB, below 2 GB together
  with 4; speed equal at ≥ 4 threads (re-measure: concurrency hides the serial initial pass and M-step tail
  of each direction, worth ≈ 2 s at 20 threads today), −25 s at 2 threads.
- **Memory:** −0.6 GB or more.
- **Why no bit can change:** each direction's computation is independent of the other's and of the
  thread count (Step 6 property tests).

## 6. Initial pass: parallelise the NULL row and the hot rows

- **Phase:** initial pass: 1.9× scaling; 3.2 s of the 34.1 s `seq` run at 20 threads (9 %), ≈ 2.5–3 s of
  wall before iteration 1 in the CLI.
- **Evidence:** 3.11 s at 1 thread → 1.70 s at 20 (fwd). The NULL row (every target token, 23.6M stamp
  checks, 86k distinct keys) and the rows of the most frequent words are each one rayon task; profile:
  `ordinary_row_sequence` → `push_new` stamp writes 33 % self, `StlOrder` emulation 10 %.
- **Change:** for rows above a size threshold, compute the first-occurrence sequence over line ranges in
  parallel (each part: its locally new target ids in order) and merge in part order, keeping only ids not
  seen in earlier parts; the `StlOrder` emulation stays sequential over the distinct keys of the row.
  The length statistics (`size_counts`) run concurrently already.
- **Expected gain:** initial pass 1.7 → ≈ 0.5 s per direction at 20 threads (−2.4 s sequential,
  −1.5 s concurrent); little at 1 thread.
- **Memory:** per-part sequences of hot rows (≤ row length × parts × 4 B, transient).
- **Why no bit can change:** the first-occurrence order of a row's keys is the same, so its insertion
  sequence, bucket count and hash iteration order are the same (checked by `check_ttable.sh` and the
  stlhash tests).

## 7. Smaller per-cell structure: hash order as `u16` where a row allows it (memory)

- **Phase:** memory only (needed only if 4 + 5 do not reach 2 GB).
- **Evidence:** the hash-order permutation is 4 B per cell (139 MB per direction) and is read only by the
  M-step's normalisation and the `-p` writer; offsets within a row fit in `u16` for rows of ≤ 65,536
  entries; only a few of the longest forward rows exceed it (they take 19.5 % of the forward visits; no reverse row is that long).
- **Change:** `u16` order for short rows, `u32` for the few long ones (two arrays and a per-row flag), or
  store the order in the saved model only.
- **Expected gain:** ≈ −130 MB per direction while training; no speed change expected.
- **Memory:** lower.
- **Why no bit can change:** the same permutation, stored narrower.

## 8. Optional per-cell lookup cache (only within the memory ceiling, switchable off)

- **Phase:** E-step iterations 3–4 and the decode, if 1–2 leave the lookup dominant.
- **Evidence:** gizapp's per-pair gather cache (2 B per cell) made its Model 1 iteration 3× faster
  (`gizapp/benchmarks/PERF-TODO.md` 2b). Here, after 1, 428M distinct-pair visits per direction and
  iteration.
- **Change:** in iteration 2 (the first with lookups) store each distinct pair's row offset (as gizapp: `u16`
  row offset or a gap encoding; rows here reach 131k entries, so plain `u16` does not suffice) and reuse it
  in the later iterations and the decode; `--no-lookup-cache` turns it off.
- **Expected gain:** most of the remaining lookup time of 3 of the 4 looked-up passes: perhaps −40 % of the
  E-step + decode at 1 thread.
- **Memory:** ≈ 0.86 GB per direction (2 B × 428M), ≈ 1.7 GB for both: conflicts with the 2 GB target, fits
  the 4 GB per 1M ceiling (4.5 GB here) only after 4/5; must be switchable off (project plan, Targets).
- **Why no bit can change:** the cache returns the same cell index the search would.
