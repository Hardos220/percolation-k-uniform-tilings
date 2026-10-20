#!/usr/bin/env bash
# compute-sanitizer memcheck of nz_pair_gpu on the production configurations (few threads, short runs)
set -u
OUT=${GPUQ_OUT:-out}; mkdir -p "$OUT"
/usr/local/cuda/bin/nvcc -O3 -lineinfo -arch=sm_80 -std=c++17 -o nz_pair_gpu nz_pair_gpu.cu || exit 3
cat tasks/pairs_A.txt tasks/pairs_B.txt tasks/pairs_C.txt tasks/test.txt | while read -r ce cs mode Lx Ly tw secs seed name groups; do
  [ -z "${ce:-}" ] && continue
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --print-limit 20 \
    ./nz_pair_gpu "cells/$ce" "cells/$cs" "$mode" "$Lx" "$Ly" "$tw" 25 "$seed" "$OUT/mc_$name" 4 2048 > "$OUT/memcheck_$name.log" 2>&1
  echo "memcheck $name exit $? : $(grep -c 'Invalid\|ERROR SUMMARY: [1-9]' "$OUT/memcheck_$name.log") error lines; $(grep 'ERROR SUMMARY' "$OUT/memcheck_$name.log")" | tee -a "$OUT/memcheck_summary.txt"
done
echo done
