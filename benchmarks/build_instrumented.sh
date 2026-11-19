#!/usr/bin/env bash
# Build the instrumented fast_align and the stlhash probe under reference/ (gitignored).
#
#   build_instrumented.sh
#
# reference/fast_align-instrumented/  copy of reference/fast_align-src + benchmarks/instrument.patch,
#                                     built with the reference build's flags (CMake Release:
#                                     -Wall -std=c++11 -O3 -g -fopenmp -O3 -DNDEBUG)
# reference/bin/stlhash_probe         from benchmarks/stlhash_probe.cc
#
# The instrumented binary behaves exactly like the plain one on stdout, stderr and -p; when
# FA_DUMP=PATH is set it also writes a dump to PATH (hex floats, see docs/SPEC-fast-align.md,
# "Instrumented build"):
#   FA_DUMP_ROWS=all|ID,ID,...   rows whose hash iteration order is dumped (default 1..10)
#   FA_DUMP_DICT=1               also dump the dictionary (id, word)
#   FA_DUMP_BUFFERS=1            also dump the per-buffer partial sums of every iteration
set -eu
. "$(dirname "$0")/common.sh"
SRC="$REF/fast_align-src"
DST="$REF/fast_align-instrumented"
rm -rf "$DST"
mkdir -p "$DST"
cp -r "$SRC/src" "$SRC/cmake" "$SRC/CMakeLists.txt" "$DST/"
# The patch was made with `diff -ru fast_align-orig fast_align-instrumented` from reference/.
(cd "$DST" && patch -s -p1 < "$BENCH_DIR/instrument.patch")
mkdir -p "$DST/build"
(cd "$DST/build" && cmake -DCMAKE_BUILD_TYPE=Release .. > cmake.log && make -j4 fast_align > make.log)
echo "built $DST/build/fast_align"

mkdir -p "$REF/bin"
g++ -std=c++11 -O2 -Wall -o "$REF/bin/stlhash_probe" "$BENCH_DIR/stlhash_probe.cc"
echo "built $REF/bin/stlhash_probe"
