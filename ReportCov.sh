#!/bin/sh
echo "Start report lcov coverage"
lcov -d build -b . --no-external -c -o ReportCoverage.info
genhtml -o CoverageReport --prefix='pwd' InitialCoverage.info ReportCoverage.info
