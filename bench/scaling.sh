#!/usr/bin/env bash
# Search throughput at several thread counts on a saved index.
#   bench/scaling.sh <build dir> <sift dir> <index file> [thread counts...]
set -euo pipefail

build=$1
sift=$2
index=$3
shift 3
counts=("$@")
if [ ${#counts[@]} -eq 0 ]; then counts=(2 4 8 14); fi

for t in "${counts[@]}"; do
  echo "== $t threads =="
  "$build/needle_bench" --base "$sift/sift_base.fvecs" --query "$sift/sift_query.fvecs" \
    --gt "$sift/sift_groundtruth.ivecs" --load "$index" --qthreads "$t" | sed -n '/QPS (/,$p'
done
