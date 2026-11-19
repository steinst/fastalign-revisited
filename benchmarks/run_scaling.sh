#!/usr/bin/env bash
# Step 9a measurements: per-phase times, end-to-end CLI times and peak RSS at several thread
# counts, plus fast_align with timestamped stderr for its own per-phase breakdown.
#
#   run_scaling.sh CORPUS "THREADS..." [--no-fast-align] [--fast-align-threads "1 20"]
#
# CORPUS is a corpus id (enfr-large, enfr-medium, ...); input reference/golden/_input/CORPUS.txt.
# Writes reference/runs/9a/CORPUS/{phases,cli}-tN.{out,log,rss,json} and fast_align-{fwd,rev}-tN.*,
# and checks every alignment file against the golden output (default config).
# Needs: cargo build --release --bin fastalign --example phases
set -u
. "$(dirname "$0")/common.sh"

[ $# -ge 2 ] || die "usage: run_scaling.sh CORPUS \"THREADS...\" [--no-fast-align] [--fast-align-threads \"T...\"]"
C=$1; THREADS=$2; shift 2
FA=1; FA_THREADS="1 20"
while [ $# -gt 0 ]; do
  case $1 in
    --no-fast-align) FA=0 ;;
    --fast-align-threads) FA_THREADS=$2; shift ;;
    *) die "unknown option $1" ;;
  esac
  shift
done
IN="$GOLDEN/_input/$C.txt"; G="$GOLDEN/$C"
[ -f "$IN" ] || die "missing $IN"
OUT="$RUNS/9a/$C"; mkdir -p "$OUT"
BIN="$REPO/target/release/fastalign"; PH="$REPO/target/release/examples/phases"
TR="python3 $BENCH_DIR/timed_run.py"

check() { # FILE GOLDEN_FILE
  if cmp -s "$1" "$2"; then echo "  identical: $(basename "$1")"; else echo "  DIFFERS: $1 vs $2"; fi
}

for t in $THREADS; do
  echo "== $C phases seq, $t threads"
  $TR "$OUT/phases-t$t" -- "$PH" "$IN" "$OUT/phases-t$t" "$t" seq > "$OUT/phases-t$t.out.tmp"
  # timed_run prints its JSON summary last; the phase table is the program's stdout before it.
  mv "$OUT/phases-t$t.out.tmp" "$OUT/phases-t$t.out"
  check "$OUT/phases-t$t.fwd" "$G/fwd.default/out"
  check "$OUT/phases-t$t.rev" "$G/rev.default/out"
  check "$OUT/phases-t$t.gdfa" "$G/atools.grow-diag-final-and/out"
  rm -f "$OUT/phases-t$t".{fwd,rev,gdfa,fwd.err,rev.err}

  echo "== $C fastalign train (CLI, both directions + gdfa), $t threads"
  $TR "$OUT/cli-t$t" -- "$BIN" train "$IN" -o "$OUT/cli-t$t" -t "$t" -s grow-diag-final-and \
    > "$OUT/cli-t$t.out"
  check "$OUT/cli-t$t.fwd.pharaoh" "$G/fwd.default/out"
  check "$OUT/cli-t$t.inv.pharaoh" "$G/rev.default/out"
  check "$OUT/cli-t$t.grow-diag-final-and.pharaoh" "$G/atools.grow-diag-final-and/out"
  rm -f "$OUT/cli-t$t".*.pharaoh
done

if [ "$FA" = 1 ]; then
  for t in $FA_THREADS; do
    for d in fwd rev; do
      r=""; [ $d = rev ] && r="-r"
      echo "== $C fast_align $d, $t threads"
      OMP_NUM_THREADS=$t $TR "$OUT/fast_align-$d-t$t" -- "$FAST_ALIGN/fast_align" -i "$IN" -d -o -v $r \
        > "$OUT/fast_align-$d-t$t.pharaoh"
      rm -f "$OUT/fast_align-$d-t$t.pharaoh"
    done
  done
fi
echo "== done: $OUT"
