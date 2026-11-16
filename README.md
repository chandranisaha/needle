# needle

[![CI](https://github.com/chandranisaha/needle/actions/workflows/ci.yml/badge.svg)](https://github.com/chandranisaha/needle/actions/workflows/ci.yml)

A vector search engine written from scratch in C++20. It builds a Vamana graph index (the algorithm
from the DiskANN paper) and answers nearest-neighbour queries without scanning every vector. No
dependencies beyond the standard library.

![The demo searching 20,000 word vectors as the query is steered](docs/demo-type.jpg)

- **Fast and accurate.** On SIFT1M (1,000,000 vectors, 128 dimensions) one thread answers 5,777
  queries per second at 98.5% recall@10, 145 times faster than brute force.
- **Level with hnswlib.** Run back to back on the same machine, the two recall-versus-speed curves
  sit within run-to-run noise of each other.
- **Parallel build.** One spinlock per node; 302 s on one thread, 74 s on 14. The tests pass under
  ThreadSanitizer.
- **Two real bugs found and explained.** A pruning rule that collapsed recall on clustered data, and
  nodes no search could reach. Both are written up below with the measurements that found them.
- **Interactive demo.** Type words, steer them with dials, and watch the index being searched and
  built in the browser.

What is here: the index, AVX2 distance code, a multi-threaded build, index save and load, an exact
brute-force baseline, a benchmark against hnswlib, tests, and the demo page.

## Build and run

Needs CMake 3.20+ and a C++20 compiler (tested with g++ 15.2 and clang 21.1 on Ubuntu under WSL2).

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/needle_tests
./build/needle_bench --n 100000 --dim 64 --threads 0
```

To benchmark a dataset in `.fvecs` format, and keep the index for next time:

```
./build/needle_bench --base sift_base.fvecs --query sift_query.fvecs --gt sift_groundtruth.ivecs \
    --threads 0 --qthreads 14 --save sift.idx
./build/needle_bench --base sift_base.fvecs --query sift_query.fvecs --gt sift_groundtruth.ivecs \
    --load sift.idx
```

To compare against hnswlib on the same data:

```
git clone https://github.com/nmslib/hnswlib third_party/hnswlib
cmake -S . -B build && cmake --build build -j
bench/compare_sift1m.sh build path/to/sift --threads 1
```

## Demo

`demo/index.html` has four views:

- **Type**: type words and the page searches 20,000 real word vectors on every keystroke. Words add
  up, a leading minus subtracts (`king - man + woman` finds `queen`), and three dials push the query
  along a direction (masculine to feminine, nature to technology, gloomy to cheerful) while the
  results change live. Each search compares about 350 of the 20,000 words and is checked against a
  full scan in the page. **Flow** keeps the query moving: several times a second the closest unused
  word is written out and pulls the query toward itself, so the dials bend a running stream of words
  ("ocean" drifts through sea, coastline, reef, wildlife; turn the technology dial and it becomes
  software, computers, internet). It is a walk through the index, not a language model.
- **Search**: click anywhere and watch the beam search hop from the entry point to the nearest
  neighbours, with a count of how many points it compared. "Brute force it" replays the same query
  by scanning every point.
- **Build**: watch a 700-point index being built, one insertion at a time. Sliders change the
  pruning parameters and restart the build.
- **Speed**: the SIFT1M numbers below as a chart, needle against hnswlib.

```
./build/needle_export_demo demo/graph.js     # rebuild the 2-D index with the C++ engine
python3 -m http.server --directory demo 8000 # then open http://localhost:8000
```

The graphs in `demo/graph.js` and `demo/words.js` come from the C++ engine. The search and the live
build on the page are JavaScript ports of `VamanaIndex::beam_search`, `insert_point` and
`robust_prune`, not the C++ code itself.

The word data is GloVe 6B (Stanford), cut to the 20,000 most common words, reduced from 300 to 64
dimensions with PCA and rounded to int8. That reduction keeps about 55% of each word's original top
10 neighbours, so results are sensible but not what full-size GloVe would return. To rebuild it
(needs numpy; the second download takes only the first 40 MB, which holds the most common words):

```
mkdir -p data/glove
curl -L -o data/glove/vocab.json https://huggingface.co/NeuML/glove-6B/resolve/main/vocab.json
curl -L -r 0-40000999 -o data/glove/model.head.safetensors \
    https://huggingface.co/NeuML/glove-6B/resolve/main/model.safetensors
python3 tools/prep_words.py data/glove data/words
./build/needle_export_words data/words/words.fvecs data/words/words.txt demo/words.js
```

On those vectors the index (R = 16) gets 0.990 recall@10 at L = 16 while comparing 246 of 20,000
words per query. The 2-D word map is laid out by `export_words.cpp` from each word's 15 exact
nearest neighbours; it is only for display, the search runs in 64 dimensions.

## Results

SIFT1M: 1,000,000 vectors, 128 dimensions, first 1,000 of the standard queries, k = 10, recall
measured against the dataset's own ground truth. Laptop CPU (Intel Core Ultra 5 225H) under WSL2,
g++ 15.2, `-O3 -march=native`. hnswlib and needle were run back to back in one session, one search
thread each.

needle: R = 32, build L = 64, alpha = 1.2. hnswlib (commit ca42672): M = 16, efConstruction = 200.

| needle L | recall@10 | queries/s | dist. comps | | hnswlib ef | recall@10 | queries/s | dist. comps |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.7738 | 25,856 | 472 | | 10 | 0.6859 | 27,838 | 400 |
| 20 | 0.8850 | 16,567 | 685 | | 20 | 0.8193 | 19,691 | 579 |
| 40 | 0.9492 | 9,387 | 1,076 | | 40 | 0.9153 | 11,895 | 907 |
| 80 | 0.9851 | 5,777 | 1,802 | | 80 | 0.9720 | 7,202 | 1,525 |
| 160 | 0.9955 | 3,151 | 3,102 | | 160 | 0.9925 | 3,900 | 2,650 |
| 320 | 0.9988 | 1,595 | 5,361 | | 320 | 0.9981 | 2,210 | 4,646 |

The same L and ef do not give the same recall, so compare at equal recall. Interpolating needle's
curve: at hnswlib's 0.972 needle does about 7,000 queries/s against 7,202, and at 0.9925 about 3,900
against 3,900. At 0.915 needle is roughly 10% ahead. Timings on this laptop move by more than that
between runs, so the fair reading is that the two are level, not that either wins.

| | needle | hnswlib |
|---|---|---|
| Build, 1 thread | 302 s | 288 s |
| Build, 14 threads | 74 s | not measured |
| Brute force, 1 thread | 39.8 queries/s | |

Throughput with 14 threads searching one shared index:

| L | recall@10 | queries/s | vs 1 thread |
|---|---|---|---|
| 40 | 0.9469 | 35,829 | 4.0x |
| 80 | 0.9852 | 20,470 | 3.8x |
| 160 | 0.9951 | 11,226 | 3.7x |

Scaling stops at about 4x on 14 threads, and the cause is not established. This CPU has 4
performance cores and 10 slower efficiency cores, so 14 threads were never going to give 14x, but
how much of the gap that explains is unmeasured. `bench/scaling.sh` runs the same search at each
thread count; it needs an otherwise idle machine to give usable numbers.

### What each change bought

Search at L = 80 on SIFT1M, one thread:

| version | recall@10 | queries/s | 1-thread build |
|---|---|---|---|
| scalar distance, one-sweep prune | 0.9517 | 2,565 | 909 s |
| AVX2 distance | 0.9517 | 6,422 | 316 s |
| AVX2 and two-sweep prune | 0.9851 | 5,777 | 302 s |

The prune change costs speed at a fixed L because the graph spends more comparisons per hop, but it
reaches a given recall with a narrower search: 0.95 recall took L = 80 before (6,422 queries/s) and
takes L = 40 now (9,387 queries/s).

## How it works

- **Graph.** Every vector is a node with at most R outgoing edges, stored as one flat `n * R` array.
- **Search.** Start at the medoid, keep the L closest candidates seen so far in a sorted list, and
  keep expanding the closest one not yet expanded until none is left.
- **Build.** Insert points in a random order, twice. For each point, search for it, then prune the
  nodes that search expanded down to R neighbours, and add the reverse edges.
- **Pruning.** A candidate is dropped when a neighbour already kept is closer to it than the point
  itself is, scaled by alpha. Alpha above 1 keeps some longer edges, which is what lets a search
  cross the dataset in few hops.
- **Threads.** The build takes one spinlock per node and never holds two at once. Searches after the
  build take no locks; each thread has its own scratch memory.

## Bugs found

**Prune kept only the nearest candidates.** I implemented the prune as one sweep with alpha = 1.2,
as the paper's pseudocode reads. With R = 32 that fills every slot with the closest candidates that
pass, and the sweep stops before it reaches farther ones in new directions. On tight Gaussian
clusters (256 clusters, 64 dimensions) recall@10 stuck at 0.73 even at L = 320, because almost no
edges left a cluster. Making the build beam wider (L = 128) did not help, which ruled out the search
and pointed at the prune. Fix: sweep twice, first keeping only candidates no kept neighbour covers
(the alpha = 1 rule), then filling the remaining slots with what alpha allows. The reference DiskANN
code does a version of this. After the fix the same clustered data reaches 0.9997 at L = 40, and on
SIFT1M recall at L = 80 went from 0.9517 to 0.9851. `test_tight_clusters_stay_connected` guards it.

**Orphaned nodes.** The first SIFT1M build left 376 vectors unreachable from the entry point, so no
query could return them. On the first 100,000 vectors there were 6, all with zero incoming edges:
every node that could point at them had dropped the edge while pruning, and nothing in the build
guarantees a node keeps one. Exact duplicates were not the cause (SIFT1M has 14,538 duplicate pairs
and no larger groups, and none of the 6 was a duplicate). Fix: a repair pass after the build attaches
each unreachable node to its nearest reachable node, evicting that node's farthest neighbour only if
the neighbour has another incoming edge. Since the prune fix above, a SIFT1M build has no orphans
even with the repair turned off (`--repair 0`), so the old prune was the main cause. The repair pass
stays because nothing else guarantees reachability.

## Tests

`needle_tests` covers: distance (AVX2 against scalar on odd sizes), exact search against a full
sort, file round trips, the recall metric, graph invariants (degree limit, no self or duplicate
edges, everything reachable), a deterministic single-threaded build, an 8-thread build, save and
load including refusing a truncated file or one from another dataset, and recall on three kinds of
data. The same tests pass under ThreadSanitizer (`-DCMAKE_CXX_FLAGS=-fsanitize=thread`).

`.github/workflows/ci.yml` runs them on every push with gcc, clang, the scalar distance kernel
forced on (`-DNEEDLE_FORCE_SCALAR=ON`), and ThreadSanitizer.

## Layout

- `include/needle/distance.h`: squared L2 distance, scalar and AVX2
- `src/vamana.cpp`: graph build, pruning, beam search, repair pass, save and load
- `src/brute_force.cpp`: exact top-k and the recall metric
- `bench/bench_main.cpp`: recall against queries per second
- `bench/hnsw_baseline.cpp`, `bench/compare_sift1m.sh`: the hnswlib comparison
- `bench/scaling.sh`: search throughput at several thread counts
- `tests/test_main.cpp`: the tests above
- `tools/unreachable.cpp`: lists unreachable nodes and checks whether they are duplicates
- `tools/export_demo.cpp`: builds the 2-D index the demo page loads
- `tools/prep_words.py`, `tools/export_words.cpp`: build the word index and word map for the Type tab

## Data and references

- SIFT1M is from the TEXMEX corpus (Jégou, Douze, Schmid; INRIA). It is not in this repository.
- The word vectors are GloVe 6B (Pennington, Socher, Manning, 2014), which Stanford releases under
  the Public Domain Dedication and License. `demo/words.js` is derived from them.
- hnswlib (Apache-2.0) is used only as the benchmark baseline and is not vendored.
- Vamana and RobustPrune: Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor
  Search on a Single Node", NeurIPS 2019.
- HNSW: Malkov and Yashunin, "Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs", 2018.

## Limits

- Only squared L2 distance. No cosine or inner product.
- The index is static: no inserts or deletes after the build.
- Vectors stay uncompressed in memory (512 MB for SIFT1M, plus 128 MB of graph).
- A threaded build is not reproducible run to run; a single-threaded one is.
- The benchmark is one dataset on one laptop.

## Next

1. Find out why search throughput stops scaling at 4x: run `bench/scaling.sh` on an idle machine,
   then pin threads to the performance cores to separate core speed from memory bandwidth
2. Product quantization to cut memory
3. Inserts and deletes
4. Compile the engine to WebAssembly so the demo runs the C++ itself
