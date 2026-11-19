#!/usr/bin/env bash
# Generate golden outputs from the reference fast_align, single-threaded (OMP_NUM_THREADS=1).
#
#   golden.sh [--no-large] [--jobs N] [--force]
#
# Layout: reference/golden/<corpus>/<dir>.<config>/{cmd,out,err,rc[,table]}
#   corpus  enfr-tiny roen-tiny enfr-medium roen-medium (+ enfr-large unless --no-large)
#   dir     fwd | rev  (rev = fast_align -r)
#   config  see CONFIGS below
# Forced alignment: <dir>.fa-train trains on all but the last HELDOUT lines and writes `table`;
#   <dir>.fa-plain / fa-diag / fa-diag-s align the last HELDOUT lines with `-f table`
#   (fa-diag = `-d -T <final tension> -m <expected length ratio>`, read from fa-train's stderr, as
#   fast_align's force_align.py does).
# atools: reference/golden/<corpus>/atools.<method>/out from fwd.default and rev.default.
# A finished job has a `done` marker and is skipped unless --force.
set -u
. "$(dirname "$0")/common.sh"
HELDOUT=100
JOBS=8; LARGE=1; FORCE=0
while [ $# -gt 0 ]; do
  case $1 in
    --no-large) LARGE=0;;
    --jobs) shift; JOBS=$1;;
    --force) FORCE=1;;
    --job) shift; MODE=job; break;;   # internal: run one job (see run_job)
    *) die "unknown option $1";;
  esac; shift
done

# name|flags   (the -p table path is substituted for TABLE)
CONFIGS=(
  "default|-d -o -v"
  "plain|"
  "d|-d"
  "do|-d -o"
  "v|-v"
  "N|-N"
  "dovN|-d -o -v -N"
  "dov-s|-d -o -v -s"
  "dov-p|-d -o -v -p TABLE"
  "I1|-I 1"
  "I10|-I 10"
  "misc|-q 0.2 -T 2 -a 0.1 -m 1.5"
  "b1000|-d -o -v -b 1000"
  "b1|-d -o -v -b 1"
)

# run_job DIR INPUT FLAGS... : run fast_align once into DIR. dir name ends .fwd/.rev prefix decides -r
run_job() {
  local dir=$1 input=$2; shift 2
  [ "$FORCE" = 1 ] || [ ! -f "$dir/done" ] || return 0
  mkdir -p "$dir"; rm -f "$dir/done"
  local base; base=$(basename "$dir")
  local rflag=""; case $base in rev.*) rflag="-r";; esac
  local args=()
  for a in "$@"; do args+=("${a//TABLE/$dir/table}"); done
  echo "fast_align -i $input $rflag ${args[*]}" > "$dir/cmd"
  OMP_NUM_THREADS=1 "$FAST_ALIGN/fast_align" -i "$input" $rflag "${args[@]}" > "$dir/out" 2> "$dir/err"
  echo $? > "$dir/rc"
  touch "$dir/done"
  echo "done $dir"
}

if [ "${MODE:-}" = job ]; then
  FORCE=${FORCE_ENV:-0}; run_job "$@"; exit 0
fi

CORPORA_LIST="$SMALL_CORPORA"; [ $LARGE = 1 ] && CORPORA_LIST="$CORPORA_LIST $LARGE_CORPORA"
mkdir -p "$GOLDEN/_input"
for c in $CORPORA_LIST; do
  [ -f "$GOLDEN/_input/$c.txt" ] || make_input "$c" "$GOLDEN/_input/$c.txt"
  [ "$c" = enfr-large ] && continue   # large: default config only, no forced-alignment slice
  n=$(n_lines "$GOLDEN/_input/$c.txt")
  head -n $((n - HELDOUT)) "$GOLDEN/_input/$c.txt" > "$GOLDEN/_input/$c.train"
  tail -n $HELDOUT "$GOLDEN/_input/$c.txt" > "$GOLDEN/_input/$c.heldout"
done

# Run a list of "dir<TAB>input<TAB>flags" lines in parallel.
run_list() {
  FORCE_ENV=$FORCE xargs -d '\n' -P "$JOBS" -I{} bash -c '
    IFS=$'"'"'\t'"'"' read -r dir input flags <<< "$1"
    # shellcheck disable=SC2086
    "'"$0"'" --job "$dir" "$input" $flags' _ {}
}

# Phase 1: training matrix (largest corpora first so the long jobs start early) + fa-train.
{
  for c in $CORPORA_LIST; do
    in="$GOLDEN/_input/$c.txt"
    if [ "$c" = enfr-large ]; then
      for d in fwd rev; do printf '%s\t%s\t%s\n' "$GOLDEN/$c/$d.default" "$in" "-d -o -v"; done
      continue
    fi
    for d in fwd rev; do
      for cfg in "${CONFIGS[@]}"; do
        name=${cfg%%|*}; flags=${cfg#*|}
        printf '%s\t%s\t%s\n' "$GOLDEN/$c/$d.$name" "$in" "$flags"
      done
      printf '%s\t%s\t%s\n' "$GOLDEN/$c/$d.fa-train" "$GOLDEN/_input/$c.train" "-d -o -v -p TABLE"
    done
  done
} | tac | run_list

# Phase 2: forced alignment of the held-out slice.
{
  for c in $SMALL_CORPORA; do
    for d in fwd rev; do
      t="$GOLDEN/$c/$d.fa-train"
      [ -f "$t/done" ] || die "missing $t"
      T=$(awk '/final tension/ {v=$NF} END {print v}' "$t/err")
      M=$(awk '/expected target length/ {v=$NF} END {print v}' "$t/err")
      h="$GOLDEN/_input/$c.heldout"
      printf '%s\t%s\t%s\n' "$GOLDEN/$c/$d.fa-plain" "$h" "-f $t/table"
      printf '%s\t%s\t%s\n' "$GOLDEN/$c/$d.fa-diag" "$h" "-d -T $T -m $M -f $t/table"
      printf '%s\t%s\t%s\n' "$GOLDEN/$c/$d.fa-diag-s" "$h" "-d -T $T -m $M -s -f $t/table"
    done
  done
} | run_list

# Phase 3: atools on the default config's two files.
for c in $CORPORA_LIST; do
  f="$GOLDEN/$c/fwd.default/out"; r="$GOLDEN/$c/rev.default/out"
  for m in intersect union grow-diag grow-diag-final grow-diag-final-and; do
    d="$GOLDEN/$c/atools.$m"; mkdir -p "$d"
    "$FAST_ALIGN/atools" -i "$f" -j "$r" -c "$m" > "$d/out" 2> "$d/err"; echo $? > "$d/rc"
  done
  d="$GOLDEN/$c/atools.invert"; mkdir -p "$d"
  "$FAST_ALIGN/atools" -i "$f" -c invert > "$d/out" 2> "$d/err"; echo $? > "$d/rc"
done
echo "golden: all done"
