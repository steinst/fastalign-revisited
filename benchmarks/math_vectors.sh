#!/usr/bin/env bash
# Build benchmarks/math_probe.cc against fast_align's own headers with the reference build's
# flags and write its output as test vectors for math.rs:
#   reference/bin/math_probe
#   crates/fastalign-core/tests/data/math/vectors.txt
set -eu
. "$(dirname "$0")/common.sh"
mkdir -p "$REF/bin" "$REPO/crates/fastalign-core/tests/data/math"
# Reference flags (CMake Release): -Wall -std=c++11 -O3 -g -fopenmp -O3 -DNDEBUG
g++ -Wall -std=c++11 -O3 -g -fopenmp -O3 -DNDEBUG -I "$REF/fast_align-src" \
  -o "$REF/bin/math_probe" "$BENCH_DIR/math_probe.cc"
if objdump -d "$REF/bin/math_probe" | grep -Eq 'vfn?madd|vfn?msub'; then die "probe contains FMA"; fi
OMP_NUM_THREADS=1 "$REF/bin/math_probe" > "$REPO/crates/fastalign-core/tests/data/math/vectors.txt"
wc -l "$REPO/crates/fastalign-core/tests/data/math/vectors.txt"
