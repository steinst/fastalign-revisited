# Shared helpers for the benchmark scripts. Source this file; do not execute it.
# shellcheck shell=bash
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$BENCH_DIR")"
REF="$REPO/reference"
FAST_ALIGN="$REF/fast_align-src/build"
CORPORA="$REF/data/corpora"
GOLDEN="$REF/golden"
RUNS="$REF/runs"

# Corpus ids are PAIR-SIZE (e.g. enfr-medium). The pairs and sizes in the test matrix:
SMALL_CORPORA="enfr-tiny roen-tiny enfr-medium roen-medium"
LARGE_CORPORA="enfr-large"

die() { echo "error: $*" >&2; exit 1; }

# timed LABEL TSV CMD... : run CMD under /usr/bin/time -v, append "LABEL<TAB>wall_s<TAB>maxrss_kb"
# to TSV, and return CMD's exit status. CMD's stdout/stderr go to LABEL.log next to TSV.
timed() {
  local label="$1" tsv="$2"; shift 2
  local dir; dir="$(dirname "$tsv")"
  local tfile="$dir/$label.time"
  /usr/bin/time -v -o "$tfile" "$@" > "$dir/$label.log" 2>&1
  local rc=$?
  local wall rss
  wall="$(awk -F': ' '/Elapsed \(wall clock\)/ {n=split($NF,a,":"); s=0; for(i=1;i<=n;i++) s=s*60+a[i]; print s}' "$tfile")"
  rss="$(awk -F': ' '/Maximum resident set size/ {print $2}' "$tfile")"
  printf '%s\t%s\t%s\n' "$label" "$wall" "$rss" >> "$tsv"
  return $rc
}

n_lines() { wc -l < "$1" | tr -d ' '; }

# corpus_files CORPUS : sets SRC TGT GOLD NT for a corpus id like enfr-medium
corpus_files() {
  local pair="${1%-*}" size="${1#*-}"
  SRC="$CORPORA/$pair/$size.src"; TGT="$CORPORA/$pair/$size.tgt"; GOLD="$CORPORA/$pair/gold.txt"
  [ -f "$SRC" ] && [ -f "$TGT" ] || die "missing corpus $1"
  NT=$(python3 -c "import json; print(json.load(open('$CORPORA/$pair/meta.json'))['test_sentences'])")
}

# make_input CORPUS OUTFILE : fast_align input "src ||| tgt". Tab is the joiner because '|' can
# occur in the text (Romanian corpus) but the corpora have no tabs.
make_input() {
  corpus_files "$1"
  paste "$SRC" "$TGT" | sed 's/\t/ ||| /' > "$2"
}
