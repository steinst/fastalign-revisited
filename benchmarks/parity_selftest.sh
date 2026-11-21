#!/usr/bin/env bash
# Self-test of parity.sh: every injected fault must make `parity.sh quick` fail.
#
#   parity_selftest.sh [--bin PATH]
#
# Faults: (1) a wrapper that produces the right output and then exits nonzero, (2) a missing
# `done` marker in a golden run, (3) a missing atools directory, (4) a binary without a working
# `symmetrize`, (5) a wrapper that drops the table file. The unmodified run must pass.
set -u
. "$(dirname "$0")/common.sh"
BIN="${FASTALIGN_BIN:-$REPO/target/release/fastalign}"
[ "${1:-}" = --bin ] && BIN=$2
[ -x "$BIN" ] || die "binary not built ($BIN)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
BAD=0

expect() { # WANT(pass|fail) LABEL ENV... -- extra parity args
  local want=$1 label=$2; shift 2
  if env "$@" bash "$BENCH_DIR/parity.sh" quick --threads 1 > "$T/log" 2>&1; then got=pass; else got=fail; fi
  if [ "$got" = "$want" ]; then echo "ok   $label ($got)"; else echo "BAD  $label: wanted $want, got $got"; tail -3 "$T/log"; BAD=1; fi
}

# a golden copy for the fixture faults (symlinks to the files; only the markers change)
mkgold() {
  local d=$1; mkdir -p "$d"
  for c in enfr-tiny roen-tiny; do cp -rs "$GOLDEN/$c" "$d/$c"; done
}
# Wrapper binaries.
cat > "$T/nonzero" <<WRAP
#!/usr/bin/env bash
"$BIN" "\$@"; rc=\$?
[ \$rc = 0 ] && [ "\${1:-}" = train ] && exit 3
exit \$rc
WRAP
cat > "$T/nosym" <<WRAP
#!/usr/bin/env bash
[ "\${1:-}" = symmetrize ] && exit 2
exec "$BIN" "\$@"
WRAP
cat > "$T/notable" <<WRAP
#!/usr/bin/env bash
"$BIN" "\$@"; rc=\$?
rm -f "\${4:-}".fwd.table "\${4:-}".inv.table
exit \$rc
WRAP
chmod +x "$T/nonzero" "$T/nosym" "$T/notable"

expect pass "unmodified" FASTALIGN_BIN="$BIN"
expect fail "exit status ignored" FASTALIGN_BIN="$T/nonzero"
expect fail "symmetrize unavailable" FASTALIGN_BIN="$T/nosym"
expect fail "table output missing" FASTALIGN_BIN="$T/notable"

mkgold "$T/g1"; rm "$T/g1/enfr-tiny/fwd.default/done"
expect fail "golden run without done" FASTALIGN_BIN="$BIN" PARITY_GOLDEN="$T/g1"
mkgold "$T/g2"; rm -r "$T/g2/roen-tiny/atools.union"
expect fail "missing atools fixture" FASTALIGN_BIN="$BIN" PARITY_GOLDEN="$T/g2"
mkgold "$T/g3"; rm -r "$T/g3/enfr-tiny/rev.misc"
expect fail "missing golden run directory" FASTALIGN_BIN="$BIN" PARITY_GOLDEN="$T/g3"
mkgold "$T/g4"; rm "$T/g4/enfr-tiny/fwd.default/out"
expect fail "missing golden output file" FASTALIGN_BIN="$BIN" PARITY_GOLDEN="$T/g4"
[ $BAD = 0 ] && echo "parity self-test: all faults detected" || { echo "parity self-test FAILED"; exit 1; }
