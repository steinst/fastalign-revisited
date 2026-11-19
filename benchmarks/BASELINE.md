# Baseline: fast_align

Reference: fast_align commit `cab1e9a`, built in gizapp's `reference/fast_align-src/build/`
(symlinked under `reference/`). Machine: 20 cores, 125 GB RAM, Linux. Config `-d -o -v`, 5
iterations, forward and `-r`, then `atools -c grow-diag-final-and`. Threads via `OMP_NUM_THREADS`.
Raw rows: `baseline_timings.tsv` (corpus, threads, step, wall s, peak RSS kB) and
`baseline_results.tsv` (corpus, threads, step, precision, recall, AER). Tiny and medium rows were
measured by `baseline.sh` in Step 2; **enfr-large rows are gizapp's measurements** (2026-11-16/17,
same machine and binary; not re-run, because a 1-thread run takes 10 minutes). The Step 2 golden
run of enfr-large reproduces gizapp's 1-thread AER exactly (16.68 / 16.25 / 15.29).

The directions were run one after the other. Gold sentences are the last lines of each corpus
(`aer.py --offset`). Corpora: see gizapp `benchmarks/BASELINE.md` (Hansards en-fr, shared-task
ro-en; not committed, research use only).

## Accuracy (AER %, precision %, recall %)

| corpus | thr | src→tgt (fwd) | tgt→src (rev) | grow-diag-final-and |
|---|---|---|---|---|
| enfr-tiny | 1 | 26.36 (P 70.02, R 80.31) | 25.02 (P 72.53, R 79.05) | 25.39 (P 70.30, R 83.21) |
| enfr-tiny | 20 | 26.36 (P 70.03, R 80.31) | 25.01 (P 72.55, R 79.05) | 25.40 (P 70.28, R 83.21) |
| roen-tiny | 1 | 44.95 (P 59.44, R 51.26) | 43.37 (P 62.05, R 52.08) | 42.47 (P 58.85, R 56.26) |
| roen-tiny | 20 | 45.00 (P 59.38, R 51.23) | 43.33 (P 62.09, R 52.11) | 42.49 (P 58.84, R 56.24) |
| enfr-medium | 1 | 18.19 (P 77.94, R 88.83) | 17.53 (P 79.68, R 87.12) | 16.80 (P 78.85, R 91.68) |
| enfr-medium | 20 | 18.15 (P 77.99, R 88.83) | 17.57 (P 79.64, R 87.10) | 16.76 (P 78.91, R 91.68) |
| roen-medium | 1 | 34.84 (P 69.95, R 60.99) | 34.40 (P 71.29, R 60.76) | 32.06 (P 69.50, R 66.46) |
| roen-medium | 20 | 34.84 (P 69.91, R 61.00) | 34.41 (P 71.27, R 60.75) | 32.03 (P 69.50, R 66.51) |
| enfr-large | 1 | 16.68 (P 79.78, R 89.75) | 16.25 (P 81.03, R 88.29) | 15.29 (P 80.64, R 92.62) |
| enfr-large | 20 | 16.63 (P 79.81, R 89.82) | 16.26 (P 81.02, R 88.29) | 15.30 (P 80.62, R 92.64) |

## Time (wall s) and peak RSS (MB)

| corpus | thr | fwd | rev | atools gdfa | RSS fwd / rev |
|---|---|---|---|---|---|
| enfr-tiny | 1 | 0.19 | 0.18 | 0.01 | 30 / 31 |
| enfr-tiny | 20 | 0.09 | 0.08 | 0.01 | 30 / 30 |
| roen-tiny | 1 | 0.22 | 0.23 | 0.01 | 37 / 37 |
| roen-tiny | 20 | 0.09 | 0.09 | 0.01 | 37 / 37 |
| enfr-medium | 1 | 27.7 | 30.3 | 0.61 | 674 / 673 |
| enfr-medium | 20 | 4.9 | 5.0 | 0.62 | 674 / 673 |
| roen-medium | 1 | 13.6 | 13.9 | 0.41 | 587 / 588 |
| roen-medium | 20 | 2.9 | 3.0 | 0.32 | 587 / 588 |
| enfr-large | 1 | 295.0 | 299.8 | 6.58 | 2942 / 2945 |
| enfr-large | 20 | 33.2 | 33.4 | 6.57 | 2942 / 2945 |

Both directions plus atools on enfr-large: 601 s at 1 thread, 73 s at 20 threads (the Targets'
reference figures). Timings of the medium corpora vary by a few seconds between sessions (gizapp
measured 24.2 s / 3.0 s for enfr-medium fwd at 1 / 20 threads; this session 27.7 / 4.9), so
treat them as ±15 %.

## fast_align is not deterministic with more than one thread

With `OMP_NUM_THREADS=1` fast_align is deterministic (a second run of enfr-medium fwd is
byte-identical to the golden file). At 20 threads the output differs from the 1-thread output.
Alignment lines that differ from the 1-thread golden files (default config):

| corpus | fwd | rev |
|---|---|---|
| enfr-tiny | 38 / 1447 | 31 / 1447 |
| roen-tiny | 34 / 1248 | 46 / 1248 |
| enfr-medium | 2,286 / 100,447 (2.3 %) | 1,031 / 100,447 (1.0 %) |
| roen-medium | 764 / 48,689 (1.6 %) | 514 / 48,689 (1.1 %) |

Cause (to be confirmed in Step 3): the count update is an unsynchronised data race and the OpenMP
reductions sum in thread order. This is why the parity target is defined against the
single-threaded fast_align (see the README).

## Golden outputs

`benchmarks/golden.sh` writes `reference/golden/<corpus>/<dir>.<config>/{cmd,out,err,rc[,table]}`
(1.9 GB, gitignored; takes about 4 minutes for the small corpora with 18 parallel jobs, plus about
5 minutes for enfr-large): 14 training configs plus forced-alignment runs, in both directions, for
the four small corpora; `atools` for all methods and `invert`; enfr-large default config only.
`benchmarks/parity.sh` compares against them.
