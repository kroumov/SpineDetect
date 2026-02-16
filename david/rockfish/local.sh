#!/usr/bin/env bash
set -euo pipefail

JOBS=5

parallel --halt soon,fail=1 --joblog parallel.log \
  -j "$JOBS" --colsep , \
  python app/downsample.py ./example {1} {2} {3} -s \
  :::: ./example/jobs.csv
