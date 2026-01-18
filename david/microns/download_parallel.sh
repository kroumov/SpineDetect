#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="./vessels"
NX=2
NY=2
NZ=32
JOBS=24

parallel -j "$JOBS" --halt soon,fail=1 --joblog parallel.log \
  python memoize_vessels.py "$OUT_DIR" {1} {2} {3} -s \
  ::: $(seq 0 "$NX") ::: $(seq 0 "$NY") ::: $(seq 0 "$NZ")
