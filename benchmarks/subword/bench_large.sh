#!/usr/bin/env bash
# Time the aligner on a subword corpus against the word-level corpus (Step 13).
#
#   bench_large.sh RUN_DIR THREADS REPEATS [--pin CPU]
#
# RUN_DIR is a `large.py prepare` directory (holds pieces.txt). Each repeat runs the word-level
# corpus and the piece corpus back to back (interleaved, so drifting machine load hits both),
# both directions + grow-diag-final-and, through `quiet_run.py` (waits for a quiet machine,
# reports other_cpus) and `/usr/bin/time` (wall s, peak RSS kB). The piece run's outputs stay in
# RUN_DIR (align.*) for `large.py score`; word-level outputs go to RUN_DIR/word/. Results are
# appended to RUN_DIR/bench.tsv: kind threads wall_s peak_rss_kb cpu_s other_cpus. IDLE_CPUS
# (default 1.5) is quiet_run's threshold; set it high to run on a busy machine and judge the
# runs by other_cpus.
# Pin 1-thread runs to a P-core (NOTES Step 9a: CPUs 0-7 are P-cores): --pin 2.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN=$1 THREADS=$2 REPEATS=$3
shift 3
PIN=()
if [[ ${1:-} == --pin ]]; then PIN=(taskset -c "$2"); fi
BIN=$REPO/target/release/fastalign
CORP=$REPO/reference/data/corpora/enfr
PY=$REPO/.venv/bin/python
mkdir -p "$RUN/word"
one() {  # kind, then the train arguments
  local kind=$1 out
  shift
  out=$("$PY" "$REPO/benchmarks/quiet_run.py" --idle-cpus "${IDLE_CPUS:-1.5}" --max-wait 3600 -- \
    /usr/bin/time -f "TIME %e %M %U %S" "${PIN[@]}" "$BIN" train "$@" -s grow-diag-final-and \
    -t "$THREADS" -q 2>&1 >/dev/null)
  local t rss cpu other
  read -r t rss cpu < <(grep '^TIME' <<<"$out" | awk '{print $2, $3, $4 + $5}')
  other=$(grep -o 'other_cpus=[0-9.]*' <<<"$out" | cut -d= -f2)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$kind" "$THREADS" "$t" "$rss" "$cpu" "$other" |
    tee -a "$RUN/bench.tsv"
}
for _ in $(seq "$REPEATS"); do
  one word "$CORP/large.src" "$CORP/large.tgt" -o "$RUN/word/align"
  one subword "$RUN/pieces.txt" -o "$RUN/align"
done
