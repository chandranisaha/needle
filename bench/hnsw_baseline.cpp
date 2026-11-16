// hnswlib on the same data, queries and metric as needle_bench, as the baseline to compare against.
//   needle_hnsw_baseline --base base.fvecs --query query.fvecs [--gt truth.ivecs]
//                        [--n ROWS] [--queries Q] [--k K] [--M M] [--efc EFC] [--seed S]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "needle/brute_force.h"
#include "needle/dataset.h"

using namespace needle;
using Clock = std::chrono::steady_clock;

static double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

static double percentile(const std::vector<double>& sorted_us, double p) {
  return sorted_us[static_cast<std::size_t>(p * static_cast<double>(sorted_us.size() - 1))];
}

// hnswlib does not count distance computations in its fast search path, so a second, untimed
// sweep swaps in this wrapper around the library's own distance function.
static hnswlib::DISTFUNC<float> g_real_dist = nullptr;
static std::uint64_t g_dist_comps = 0;
static float counting_dist(const void* a, const void* b, const void* param) {
  ++g_dist_comps;
  return g_real_dist(a, b, param);
}

static std::vector<std::uint32_t> top_ids(hnswlib::HierarchicalNSW<float>& index, const float* query,
                                          std::size_t k) {
  auto heap = index.searchKnn(query, k);
  std::vector<std::uint32_t> ids(heap.size());
  for (std::size_t i = heap.size(); i-- > 0;) {
    ids[i] = static_cast<std::uint32_t>(heap.top().second);
    heap.pop();
  }
  return ids;
}

int main(int argc, char** argv) {
  std::string base_path, query_path, gt_path;
  std::size_t rows = 0, num_queries = 1000, k = 10, M = 16, efc = 200, seed = 42;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const char* value = argv[i + 1];
    if (flag == "--base") base_path = value;
    else if (flag == "--query") query_path = value;
    else if (flag == "--gt") gt_path = value;
    else if (flag == "--n") rows = std::strtoull(value, nullptr, 10);
    else if (flag == "--queries") num_queries = std::strtoull(value, nullptr, 10);
    else if (flag == "--k") k = std::strtoull(value, nullptr, 10);
    else if (flag == "--M") M = std::strtoull(value, nullptr, 10);
    else if (flag == "--efc") efc = std::strtoull(value, nullptr, 10);
    else if (flag == "--seed") seed = std::strtoull(value, nullptr, 10);
    else {
      std::fprintf(stderr, "unknown flag %s\n", flag.c_str());
      return 2;
    }
  }
  if (base_path.empty() || query_path.empty()) {
    std::fprintf(stderr, "--base and --query are required\n");
    return 2;
  }

  const Dataset base = read_fvecs(base_path, rows);
  const Dataset queries = read_fvecs(query_path, num_queries);
  std::printf("base: %zu x %zu (%s)\n", base.n, base.dim, base_path.c_str());
  std::printf("queries: %zu, k = %zu\n", queries.n, k);

  std::vector<std::vector<std::uint32_t>> truth;
  if (!gt_path.empty() && rows == 0) {
    truth = read_ivecs(gt_path);
    truth.resize(queries.n);
  } else {
    truth = exact_knn_batch(base, queries, k);
  }

  auto start = Clock::now();
  for (std::size_t q = 0; q < queries.n; ++q) (void)exact_knn(base, queries.row(q), k);
  const double brute_qps = static_cast<double>(queries.n) / seconds_since(start);
  std::printf("brute force (needle, scalar): %.1f QPS (1 thread)\n", brute_qps);

  hnswlib::L2Space space(base.dim);
  hnswlib::HierarchicalNSW<float> index(&space, base.n, M, efc, seed);
  start = Clock::now();
  for (std::size_t i = 0; i < base.n; ++i) index.addPoint(base.row(i), i);
  std::printf("build: hnswlib M=%zu efConstruction=%zu -> %.1f s (1 thread)\n\n", M, efc,
              seconds_since(start));

  std::printf("| ef | recall@%zu | QPS | speedup | p50 us | p99 us | dist comps |\n", k);
  std::printf("|---|---|---|---|---|---|---|\n");
  for (const std::size_t ef : {10, 20, 40, 80, 160, 320}) {
    if (ef < k) continue;
    index.setEf(ef);
    std::vector<std::vector<std::uint32_t>> found(queries.n);
    std::vector<double> latency_us(queries.n);

    const auto sweep_start = Clock::now();
    for (std::size_t q = 0; q < queries.n; ++q) {
      const auto t0 = Clock::now();
      found[q] = top_ids(index, queries.row(q), k);
      latency_us[q] = seconds_since(t0) * 1e6;
    }
    const double qps = static_cast<double>(queries.n) / seconds_since(sweep_start);

    g_real_dist = index.fstdistfunc_;
    g_dist_comps = 0;
    index.fstdistfunc_ = counting_dist;
    for (std::size_t q = 0; q < queries.n; ++q) (void)top_ids(index, queries.row(q), k);
    index.fstdistfunc_ = g_real_dist;

    std::sort(latency_us.begin(), latency_us.end());
    std::printf("| %zu | %.4f | %.0f | %.1fx | %.1f | %.1f | %.0f |\n", ef,
                recall_at_k(truth, found, k), qps, qps / brute_qps, percentile(latency_us, 0.50),
                percentile(latency_us, 0.99),
                static_cast<double>(g_dist_comps) / static_cast<double>(queries.n));
  }
  return 0;
}
