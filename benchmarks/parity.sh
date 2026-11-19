#!/usr/bin/env bash
# Byte-for-byte parity check of our binary against the golden fast_align outputs.
#
#   parity.sh {quick|full} [--threads N...] [--bin PATH]
#
# quick: tiny corpora, every config, threads 1 and 4 (override with --threads).
# full:  all corpora incl. en-fr large (default config only), threads 1 2 4 8 20.
# Compared per run: stdout (alignments, -s scores), stderr statistics (progress dots ignored),
# the -p table. Then atools parity for the symmetrisation methods, once `symmetrize` exists.
# Generate the golden files first with benchmarks/golden.sh.
#
# Our binary is driven through its own CLI (`train`, `align --table`, `symmetrize`); `run_ours`
# translates the fast_align flags recorded in each golden run's `cmd` file.
set -u
. "$(dirname "$0")/common.sh"

MODE=${1:-}; shift || true
case $MODE in quick|full) ;; *) die "usage: parity.sh {quick|full} [--threads N...] [--bin PATH]";; esac
THREADS=""; BIN="${FASTALIGN_BIN:-$REPO/target/release/fastalign}"
while [ $# -gt 0 ]; do
  case $1 in
    --bin) shift; BIN=$1;;
    --threads) shift; while [ $# -gt 0 ] && [[ $1 =~ ^[0-9]+$ ]]; do THREADS="$THREADS $1"; shift; done; continue;;
    *) die "unknown option $1";;
  esac; shift
done
if [ -z "$THREADS" ]; then [ "$MODE" = quick ] && THREADS="1 4" || THREADS="1 2 4 8 20"; fi

[ -x "$BIN" ] || { echo "parity: binary not built ($BIN)"; exit 1; }
[ -d "$GOLDEN/enfr-tiny" ] || die "golden outputs missing; run benchmarks/golden.sh"

if [ "$MODE" = quick ]; then CORPUS_LIST="enfr-tiny roen-tiny"; else CORPUS_LIST="$SMALL_CORPORA $LARGE_CORPORA"; fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0; SKIP=0

# run_ours INPUT OUTDIR THREADS FAST_ALIGN_FLAGS... : writes OUTDIR/out and OUTDIR/err.
# The recorded fast_align flags are translated to the `fastalign` CLI (see the README's flag table):
#   training: `train INPUT -o OUTDIR/p --directions fwd|inv --log [--table]`; the statistics log
#             (OUTDIR/p.<dir>.log) plus fast_align's `ARG=<c>` option echo become `err`;
#   -f TABLE: `align --table TABLE INPUT`, whose stderr gets the same echo.
run_ours() {
  local input=$1 outdir=$2 threads=$3; shift 3
  mkdir -p "$outdir"
  local dir=fwd diag=0 opt=0 vb=0 table="" forced="" echo_args="" c
  local -a extra=() fextra=()
  while [ $# -gt 0 ]; do
    local f=$1; shift
    case $f in
      -i) shift;;
      -r) dir=inv;;
      -d) diag=1;;
      -o) opt=1;;
      -v) vb=1;;
      -N) extra+=(--no-null); fextra+=(--no-null);;
      -s) extra+=(--scores);;
      -I) extra+=(-I "$1"); shift;;
      -q) extra+=(--p0 "$1"); fextra+=(--p0 "$1"); shift;;
      -T) diag=1; extra+=(-T "$1"); fextra+=(--tension "$1"); shift;;
      -a) extra+=(-a "$1"); shift;;
      -m) fextra+=(-m "$1"); shift;;
      -b) extra+=(-b "$1"); shift;;
      -t) extra+=(--beam-threshold "$1"); shift;;
      -p) table=$1; extra+=(--table); shift;;
      -f) forced=$1; shift;;
      *) die "parity: unhandled fast_align flag $f";;
    esac
    case $f in -i|-r|-d|-o|-v|-N|-s|-I|-q|-T|-a|-m|-b|-t|-p|-f) echo_args="$echo_args ${f#-}";; esac
  done
  # fast_align echoes every option letter on stderr, input option first.
  { echo "ARG=i"; for c in $echo_args; do echo "ARG=$c"; done; } > "$outdir/echo"
  if [ -n "$forced" ]; then
    [ $diag = 0 ] && fextra+=(--no-diagonal)
    "$BIN" align --table "$forced" "$input" --direction "$dir" "${fextra[@]}" -t "$threads" \
      > "$outdir/out" 2> "$outdir/errtail"
    cat "$outdir/echo" "$outdir/errtail" > "$outdir/err"
  else
    [ $diag = 0 ] && extra+=(--no-diagonal)
    [ $opt = 0 ] && extra+=(--no-optimize-tension)
    [ $vb = 0 ] && extra+=(--no-variational-bayes)
    "$BIN" train "$input" -o "$outdir/p" --directions "$dir" --log --quiet -t "$threads" "${extra[@]}" \
      2> "$outdir/errtail"
    cat "$outdir/p.$dir.pharaoh" > "$outdir/out"
    cat "$outdir/echo" "$outdir/p.$dir.log" > "$outdir/err"
    [ -n "$table" ] && cp "$outdir/p.$dir.table" "$table"
  fi
  return 0
}

check() { # KIND GOLDENFILE OURFILE LABEL
  if msg=$(python3 "$BENCH_DIR/parity_compare.py" "$1" "$2" "$3"); then PASS=$((PASS+1))
  else FAIL=$((FAIL+1)); echo "FAIL $4 [$1]"; echo "$msg"; fi
}

for c in $CORPUS_LIST; do
  for g in "$GOLDEN/$c"/{fwd,rev}.*; do
    [ -f "$g/done" ] || continue
    cfg=$(basename "$g")
    # Recorded command: fast_align -i INPUT [-r] FLAGS...  -> keep everything after the input.
    read -r _ _ input rest < "$g/cmd"
    args=${rest//$g\/table/$TMP\/table}
    for t in $THREADS; do
      label="$c/$cfg t=$t"
      out="$TMP/run"; rm -rf "$out"
      # shellcheck disable=SC2086
      run_ours "$input" "$out" "$t" $args
      check out "$g/out" "$out/out" "$label"
      check err "$g/err" "$out/err" "$label"
      if [ -f "$g/table" ] && [[ $rest == *"-p "* ]]; then check table "$g/table" "$TMP/table" "$label"; fi
    done
  done
  # atools
  if "$BIN" symmetrize --help > /dev/null 2>&1; then
    for a in "$GOLDEN/$c"/atools.*; do
      [ -d "$a" ] || continue
      m=${a##*/atools.}
      f="$GOLDEN/$c/fwd.default/out"; r="$GOLDEN/$c/rev.default/out"
      if [ "$m" = invert ]; then "$BIN" symmetrize "$f" -m invert > "$TMP/sym" 2>/dev/null
      else "$BIN" symmetrize "$f" "$r" -m "$m" > "$TMP/sym" 2>/dev/null; fi
      check out "$a/out" "$TMP/sym" "$c/atools.$m"
    done
  else
    SKIP=$((SKIP+1)); echo "SKIP $c atools: '$BIN symmetrize' not available yet (Step 7)"
  fi
done
echo "parity $MODE (threads:$THREADS): $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" = 0 ]
