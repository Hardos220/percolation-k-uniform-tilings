#!/usr/bin/env bash
# Coupled slip-pair runs on the GPU: build nz_pair_gpu once, run every task of the list
# (cellE cellS mode Lx Ly twist seconds seed name groups); outputs $OUT/<name>__run<g>_{E,S}.txt.
set -u
LIST=$1
OUT=${GPUQ_OUT:-out}
mkdir -p "$OUT"
/usr/local/cuda/bin/nvcc -O3 -arch=sm_80 -std=c++17 -o nz_pair_gpu nz_pair_gpu.cu || exit 3
n=$(grep -c . "$LIST"); i=0
while read -r ce cs mode Lx Ly tw secs seed name groups; do
  groups=${groups%$'\r'}
  [ -z "${ce:-}" ] && continue
  i=$((i + 1))
  [ -s "$OUT/${name}__run000_E.txt" ] && continue
  for attempt in 1 2 3; do
    echo "[$i/$n] $name ${secs}s (attempt $attempt)"
    if ./nz_pair_gpu "cells/$ce" "cells/$cs" "$mode" "$Lx" "$Ly" "$tw" "$secs" "$seed" "$OUT/$name" "$groups" >> "$OUT/perf.txt" 2>> "$OUT/errors.txt" < /dev/null; then
      break
    fi
    rm -f "$OUT/${name}__run"*
    echo "FAILED $name attempt $attempt" | tee -a "$OUT/failed.txt"
    [ "$attempt" -lt 3 ] && sleep 120
  done
done < "$LIST"
echo "batch done: $i tasks"
