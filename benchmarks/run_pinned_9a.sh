#!/usr/bin/env bash
# Step 9a (en-fr large): 1-thread runs pinned to a P-core (CPU 2), repeated 20-thread CLI runs, 1-thread profile.
# Run after run_scaling.sh enfr-large; writes reference/runs/9a/enfr-large/{cli-t20-repN,cli-t1-pin,phases-t1-pin,fast_align-*-t1-pin}.* and reference/runs/9a/prof-t1/perf.data.
# Needs the release binary, the phases example and the profiling build (see profile_phases.py).
set -u
cd "$(dirname "$0")/.."; R=$(pwd)
. benchmarks/common.sh
IN=$GOLDEN/_input/enfr-large.txt; G=$GOLDEN/enfr-large; O=$RUNS/9a/enfr-large
TR="python3 benchmarks/timed_run.py"
for r in 1 2 3; do
  $TR $O/cli-t20-rep$r -- target/release/fastalign train $IN -o $O/rep -t 20 -s grow-diag-final-and > /dev/null
  cmp -s $O/rep.grow-diag-final-and.pharaoh $G/atools.grow-diag-final-and/out && echo "rep$r identical" || echo "rep$r DIFFERS"
done
rm -f $O/rep.*
$TR $O/cli-t1-pin -- taskset -c 2 target/release/fastalign train $IN -o $O/pin -t 1 -s grow-diag-final-and > /dev/null
cmp -s $O/pin.grow-diag-final-and.pharaoh $G/atools.grow-diag-final-and/out && echo "cli pin identical" || echo "cli pin DIFFERS"
rm -f $O/pin.*
$TR $O/phases-t1-pin -- taskset -c 2 target/release/examples/phases $IN $O/pin 1 seq > $O/phases-t1-pin.out
cmp -s $O/pin.gdfa $G/atools.grow-diag-final-and/out && echo "phases pin identical" || echo "phases pin DIFFERS"
rm -f $O/pin.*
for d in fwd rev; do
  r=""; [ $d = rev ] && r="-r"
  OMP_NUM_THREADS=1 $TR $O/fast_align-$d-t1-pin -- taskset -c 2 $FAST_ALIGN/fast_align -i $IN -d -o -v $r > /dev/null
done
mkdir -p $RUNS/9a/prof-t1 && cd $RUNS/9a/prof-t1
taskset -c 2 perf record -F 499 -g -o perf.data -- $R/target/prof/profiling/fastalign train $IN -o out -t 1 --directions fwd > /dev/null 2> perf.err
cmp -s out.fwd.pharaoh $G/fwd.default/out && echo "prof t1 identical" || echo "prof t1 DIFFERS"
rm -f out.*
echo done
