#!/usr/bin/env bash
# Check that the instrumented fast_align (built by build_instrumented.sh) is byte-identical to the
# golden outputs of the plain build: stdout, stderr (the -p path echoed on stderr is substituted)
# and the -p table, for every golden fwd/rev run of the given corpora. Dumps are enabled
# (FA_DUMP, all rows, dictionary, per-buffer sums) so the instrumentation code paths run.
#
#   check_instrumented.sh [CORPUS...]      default: enfr-tiny roen-tiny enfr-medium roen-medium
# DUMP_ROWS=1,2,3 limits the row dump (default all; use a short list for enfr-large).
set -u
. "$(dirname "$0")/common.sh"
BIN="$REF/fast_align-instrumented/build/fast_align"
[ -x "$BIN" ] || die "run benchmarks/build_instrumented.sh first"
CORP=${*:-$SMALL_CORPORA}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
for c in $CORP; do
  for g in "$GOLDEN/$c"/{fwd,rev}.*; do
    [ -f "$g/done" ] || continue
    read -r _ _ input rest < "$g/cmd"
    args=${rest//$g\/table/$TMP\/table}
    rm -f "$TMP/table"
    # shellcheck disable=SC2086
    OMP_NUM_THREADS=1 FA_DUMP="$TMP/dump" FA_DUMP_ROWS=${DUMP_ROWS:-all} FA_DUMP_DICT=1 FA_DUMP_BUFFERS=1 \
      "$BIN" -i "$input" $args > "$TMP/out" 2> "$TMP/err.raw"
    rc=$?
    sed "s#$TMP/table#$g/table#" "$TMP/err.raw" > "$TMP/err"
    ok=1
    cmp -s "$g/out" "$TMP/out" || { ok=0; echo "FAIL $c/${g##*/} stdout"; }
    cmp -s "$g/err" "$TMP/err" || { ok=0; echo "FAIL $c/${g##*/} stderr"; }
    [ "$rc" = "$(cat "$g/rc")" ] || { ok=0; echo "FAIL $c/${g##*/} rc $rc"; }
    if [[ $rest == *"-p "* ]]; then cmp -s "$g/table" "$TMP/table" || { ok=0; echo "FAIL $c/${g##*/} table"; }; fi
    if [ $ok = 1 ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
  done
done
echo "instrumented vs golden: $PASS identical, $FAIL differ"
[ $FAIL = 0 ]
