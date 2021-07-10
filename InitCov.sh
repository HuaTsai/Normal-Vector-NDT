#!/bin/sh
echo "Start initialize lcov coverage"
lcov -d build -z
lcov -d build -b . --no-external --initial -c -o InitialCoverage.info
