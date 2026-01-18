#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="./svx"
NX=1
NY=1
NZ=4
JOBS=16

parallel -j "$JOBS" --halt soon,fail=1 --joblog parallel.log \
  python memoize_svx.py "$OUT_DIR" {1} {2} {3} -s \
  ::: $(seq 0 "$NX") ::: $(seq 0 "$NY") ::: $(seq 0 "$NZ")
