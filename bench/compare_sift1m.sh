#!/usr/bin/env bash
# Runs hnswlib and needle back to back on SIFT1M so both sets of timings share one machine state.
#   bench/compare_sift1m.sh <build dir> <dir holding sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs>
#                           [extra needle_bench flags, e.g. --threads 0 --L 128 or --load index.idx]
set -euo pipefail
build=${1:?build dir}
sift=${2:?sift dir}
shift 2

echo "== hnswlib =="
"$build/needle_hnsw_baseline" --base "$sift/sift_base.fvecs" --query "$sift/sift_query.fvecs" \
  --gt "$sift/sift_groundtruth.ivecs"

echo
echo "== needle =="
"$build/needle_bench" --base "$sift/sift_base.fvecs" --query "$sift/sift_query.fvecs" \
  --gt "$sift/sift_groundtruth.ivecs" "$@"
