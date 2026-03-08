#!/bin/bash
# Compare performance across versions at a range of matrix sizes.
#
# Usage:
#   ./scripts/compare_versions.sh
#   ./scripts/compare_versions.sh 500 1000 2000 4000    # custom sizes
#
# Output: results/comparison.csv

SIZES="${@:-500 1000 2000 4000}"
REPS=3
OUT=results/comparison.csv
VERSIONS="v1_baseline v2_serial_opt"

mkdir -p results
echo "version,n,threads,rep,time_s,gflops" > $OUT

for VERSION in $VERSIONS; do
    echo ">>> Building $VERSION ..."
    make bench VERSION=$VERSION -s
    for n in $SIZES; do
        echo "    n=$n ..."
        ./test/benchmark $n $REPS | sed "s/^/${VERSION},/" >> $OUT
    done
done

echo ""
echo "Results written to $OUT"
echo ""
column -t -s, $OUT
