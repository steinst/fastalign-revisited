#!/usr/bin/env bash
# Step 9b: one end-to-end CLI run (`fastalign train INPUT -o P -t T -s grow-diag-final-and`) on a
# quiet machine (quiet_run.py), outputs checked against golden, wall time and peak RSS printed.
#
#   bench_cli.sh TAG THREADS [CORPUS] [--pin CPUS] [--bin FASTALIGN_BINARY]
set -u
. "$(dirname "$0")/common.sh"
[ $# -ge 2 ] || die "usage: bench_cli.sh TAG THREADS [CORPUS] [--pin CPUS] [--bin BINARY]"
TAG=$1; T=$2; shift 2
C=enfr-large; PIN=""; BIN="$REPO/target/release/fastalign"
while [ $# -gt 0 ]; do
  case $1 in
    --pin) PIN="taskset -c $2"; shift ;;
    --bin) BIN=$2; shift ;;
    *) C=$1 ;;
  esac
  shift
done
IN="$GOLDEN/_input/$C.txt"; G="$GOLDEN/$C"
OUT="$RUNS/9b/$C"; mkdir -p "$OUT"
P="$OUT/cli-$TAG-t$T"
python3 "$BENCH_DIR/quiet_run.py" --idle-cpus "${IDLE_CPUS:-1.0}" -- $PIN python3 "$BENCH_DIR/timed_run.py" "$P" -- "$BIN" train "$IN" \
  -o "$P" -t "$T" -s grow-diag-final-and > "$P.out" 2> "$P.quiet"
tail -1 "$P.quiet"
ok=1
for x in "fwd fwd.default" "inv rev.default" "grow-diag-final-and atools.grow-diag-final-and"; do
  set -- $x
  cmp -s "$P.$1.pharaoh" "$G/$2/out" || { echo "DIFFERS: $P.$1.pharaoh"; ok=0; }
done
rm -f "$P".*.pharaoh
[ $ok = 1 ] && echo "outputs identical to golden"
grep -o '"wall_s": [0-9.]*\|"peak_rss_mb": [0-9.]*' "$P.out" | paste -sd' '
