#!/usr/bin/env bash
# fast_align baseline: time, peak RSS and AER per corpus, direction and thread count.
#
#   baseline.sh CORPUS THREADS      e.g. baseline.sh enfr-medium 20
#
# Options -d -o -v (default config), both directions, then atools grow-diag-final-and.
# Output: reference/runs/CORPUS/fast_align-tTHREADS/{timings.tsv,results.tsv,*.pharaoh}
# Rows are also printed as TSV: corpus<TAB>threads<TAB>label<TAB>wall_s<TAB>maxrss_kb.
set -u
. "$(dirname "$0")/common.sh"
[ $# -eq 2 ] || die "usage: baseline.sh CORPUS THREADS"
C=$1; T=$2
corpus_files "$C"
OUT="$RUNS/$C/fast_align-t$T"; mkdir -p "$OUT"
export OMP_NUM_THREADS=$T
IN="$GOLDEN/_input/$C.txt"; [ -f "$IN" ] || { mkdir -p "$GOLDEN/_input"; make_input "$C" "$IN"; }
: > "$OUT/timings.tsv"
timed fwd "$OUT/timings.tsv" bash -c "'$FAST_ALIGN/fast_align' -i '$IN' -d -o -v > '$OUT/fwd.pharaoh'" || die fwd
timed rev "$OUT/timings.tsv" bash -c "'$FAST_ALIGN/fast_align' -i '$IN' -d -o -v -r > '$OUT/rev.pharaoh'" || die rev
timed sym "$OUT/timings.tsv" bash -c "'$FAST_ALIGN/atools' -i '$OUT/fwd.pharaoh' -j '$OUT/rev.pharaoh' -c grow-diag-final-and > '$OUT/sym.pharaoh'" || die atools
# AER on the gold sentences, which are the last NT lines of every corpus.
N=$(n_lines "$IN")
: > "$OUT/results.tsv"
for d in fwd rev sym; do
  r=$(python3 "$BENCH_DIR/aer.py" "$OUT/$d.pharaoh" "$GOLD" --offset $((N - NT)) --format tsv) || die aer
  printf '%s\t%s\t%s\t%s\n' "$C" "$T" "$d" "$r" >> "$OUT/results.tsv"
done
awk -v c="$C" -v t="$T" -F'\t' '{print c "\t" t "\t" $0}' "$OUT/timings.tsv"
cat "$OUT/results.tsv"
