#!/usr/bin/env bash
# Compare our corpus/vocabulary/table construction (Step 4) with the instrumented fast_align:
# dictionary ids, size_counts order, and every row's hash iteration order and bucket count.
#
#   check_ttable.sh CORPUS [FLAGS...]     e.g.  check_ttable.sh enfr-medium -r -b 1000
#
# FLAGS are fast_align flags among -r -N -b N. fast_align runs with -I 0 (initial pass only).
# Needs benchmarks/build_instrumented.sh and `cargo build --release --example dump_tables`.
set -u
. "$(dirname "$0")/common.sh"
CORPUS=$1; shift
INPUT="$GOLDEN/_input/$CORPUS.txt"
FA="$REF/fast_align-instrumented/build/fast_align"
OURS="${DUMP_TABLES:-$REPO/target/release/examples/dump_tables}"
[ -f "$INPUT" ] || die "no input $INPUT"
[ -x "$FA" ] || die "run benchmarks/build_instrumented.sh"
[ -x "$OURS" ] || die "cargo build --release --example dump_tables"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
OMP_NUM_THREADS=1 FA_DUMP="$TMP/fa" FA_DUMP_ROWS=all FA_DUMP_DICT=1 "$FA" -i "$INPUT" -I 0 "$@" > /dev/null 2>&1
"$OURS" -i "$INPUT" "$@" ${THREADS:+--threads $THREADS} > "$TMP/ours"
# fast_align's initial line has one more field (freeze_copy_same_order); compare the shared ones.
sed -n 's/^\(initial .* rows [0-9]*\).*/\1/p; /^\(dict\|size_counts\|row\) /p' "$TMP/fa" > "$TMP/fa.cmp"
fail=0
for kind in initial dict size_counts row; do
  a=$(grep -c "^$kind " "$TMP/fa.cmp"); b=$(grep -c "^$kind " "$TMP/ours")
  if cmp -s <(grep -a "^$kind " "$TMP/fa.cmp") <(grep -a "^$kind " "$TMP/ours"); then
    echo "  $kind: $a lines identical"
  else
    fail=1; echo "  $kind: DIFFER ($a fast_align vs $b ours); first difference:"
    diff <(grep -a "^$kind " "$TMP/fa.cmp") <(grep -a "^$kind " "$TMP/ours") | head -4 | cut -c1-200
  fi
done
cells=$(awk '$1=="row"{s+=$4} END{print s}' "$TMP/ours")
if [ $fail = 0 ]; then echo "check_ttable $CORPUS $*: OK ($cells cells)"; else echo "check_ttable $CORPUS $*: FAIL"; fi
exit $fail
