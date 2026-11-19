#!/usr/bin/env bash
# Step 9b: one `phases seq` run on a corpus, outputs checked against golden, key phases printed.
#
#   bench_phases.sh TAG THREADS [CORPUS] [--pin CPU] [--bin PHASES_BINARY]
#
# Waits for a quiet machine and reports the CPU other processes used (quiet_run.py).
#
# Writes reference/runs/9b/CORPUS/TAG-tTHREADS.{out,json,...}. Needs
# cargo build --release --example phases.
set -u
. "$(dirname "$0")/common.sh"
[ $# -ge 2 ] || die "usage: bench_phases.sh TAG THREADS [CORPUS] [--pin CPU]"
TAG=$1; T=$2; shift 2
C=enfr-large; PIN=""; FLAGS=""; PH="$REPO/target/release/examples/phases"
while [ $# -gt 0 ]; do
  case $1 in
    --pin) PIN="taskset -c $2"; shift ;;
    --bin) PH=$2; shift ;;
    --flags) FLAGS=$2; shift ;;
    *) C=$1 ;;
  esac
  shift
done
IN="$GOLDEN/_input/$C.txt"; G="$GOLDEN/$C"
OUT="$RUNS/9b/$C"; mkdir -p "$OUT"
P="$OUT/$TAG-t$T"
python3 "$BENCH_DIR/quiet_run.py" --idle-cpus "${IDLE_CPUS:-1.0}" -- $PIN python3 "$BENCH_DIR/timed_run.py" "$P" -- "$PH" "$IN" "$P" "$T" seq $FLAGS \
  > "$P.out" 2> "$P.quiet"
tail -1 "$P.quiet"
ok=1
for x in "fwd fwd.default" "rev rev.default" "gdfa atools.grow-diag-final-and"; do
  set -- $x
  cmp -s "$P.$1" "$G/$2/out" || { echo "DIFFERS: $P.$1"; ok=0; }
done
rm -f "$P".{fwd,rev,gdfa,fwd.err,rev.err}
[ $ok = 1 ] && echo "outputs identical to golden"
awk -F'\t' '$1 ~ /^(initial_pass|e_step|m_step|decode|total)$/ {printf "%s %s %s\n", $1, $2, $3}' "$P.out" \
  | paste -sd' ' | sed 's/ \(initial_pass\|e_step\|m_step\|decode\|total\) /\n\1 /g'
grep -o '"peak_rss_mb": [0-9.]*' "$P.out"
