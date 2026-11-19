#!/usr/bin/env bash
# Bounded fuzzing of the readers, the model loader, the -p table reader, the pharaoh/atools code
# and a tiny end-to-end training (Step 10). Needs nightly + cargo-fuzz.
#
#   benchmarks/fuzz.sh [SECONDS_PER_TARGET=60]
#
# Seeds are generated from the synthetic data below into crates/fastalign-core/fuzz/corpus
# (gitignored). Exit status 1 if any target crashes (artifacts in fuzz/artifacts).
set -euo pipefail
. ~/.cargo/env
cd "$(dirname "$0")/.."
SECS=${1:-60}
FUZZ=crates/fastalign-core/fuzz
BIN=target/release/fastalign
[ -x $BIN ] || cargo build --release -p fastalign-core
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

printf 'the house ||| la maison\nthe blue house ||| la maison bleue\na book ||| un livre\n\n  ||| x\nb |||\n' > "$T/c.txt"
printf 'the house\na book\n' > "$T/s.txt"; printf 'la maison\nun livre\n' > "$T/t.txt"
$BIN train "$T/c.txt" -o "$T/m" --save "$T/m.fastalign" --table --log -q -t 1 -s grow-diag-final-and
mkdir -p $FUZZ/corpus/{corpus_joined,corpus_pair,train_small,model_read,text_table,pharaoh,subword}
{ printf '\0'; cat "$T/c.txt"; } > $FUZZ/corpus/corpus_joined/seed
{ printf '\x80'; cat "$T/s.txt" "$T/t.txt"; } > $FUZZ/corpus/corpus_pair/seed
{ printf '\x1f\x03'; head -c 600 "$T/c.txt"; } > $FUZZ/corpus/train_small/seed
cp "$T/m.fastalign" $FUZZ/corpus/model_read/seed
{ cat "$T/m.fwd.table"; printf '\xff'; cat "$T/c.txt"; } > $FUZZ/corpus/text_table/seed
{ cat "$T/m.fwd.pharaoh"; printf '\xff'; cat "$T/m.inv.pharaoh"; } > $FUZZ/corpus/pharaoh/seed
printf '\xe2\x96\x81the \xe2\x96\x81ho use ||| \xe2\x96\x81la \xe2\x96\x81mai son\n' > "$T/sw.txt"
$BIN train "$T/sw.txt" -o "$T/sw" --save "$T/sw.fastalign" --subword-marker $'\u2581' -q -t 1
cp "$T/sw.fastalign" $FUZZ/corpus/model_read/seed-subword
{ cat "$T/sw.txt"; printf '\xff0-0 1-1 2-0\xff\xe2\x96\x81'; } > $FUZZ/corpus/subword/seed

rc=0
for t in corpus_joined corpus_pair train_small model_read text_table pharaoh subword; do
  echo "== $t ($SECS s)"
  cargo +nightly fuzz run --fuzz-dir $FUZZ "$t" -- -max_total_time="$SECS" -rss_limit_mb=4096 \
    -max_len=65536 2>&1 | tail -4 || rc=1
done
exit $rc
