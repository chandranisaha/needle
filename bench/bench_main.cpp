// Recall-versus-speed benchmark.
//   needle_bench [--n N] [--dim D] [--latent I | --latent 0 --clusters C] [--queries Q] [--k K]
//                [--R R] [--L L] [--alpha A] [--seed S]
//   needle_bench --base base.fvecs --query query.fvecs [--gt truth.ivecs] [--rows N] ...
//   Build options: [--threads T] (0 = all cores) [--save index.idx] [--load index.idx]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "needle/brute_force.h"
#include "needle/dataset.h"
#include "needle/distance.h"
#include "needle/vamana.h"

using namespace needle;
using Clock = std::chrono::steady_clock;

static double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Options {
  std::size_t n = 100'000;
  std::size_t dim = 64;
  std::size_t clusters = 256;
  std::size_t latent = 12;  // 0 switches to tight Gaussian blobs, a known hard case
  std::size_t queries = 1000;
  std::size_t k = 10;
  std::size_t rows = 0;  // read only the first rows of --base; ground truth is then recomputed
  unsigned query_threads = 1;  // >1 adds a throughput table with that many searching threads
  std::string base_path, query_path, gt_path, save_path, load_path;
  VamanaParams index;
};

static Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const char* value = argv[i + 1];
    if (flag == "--n") o.n = std::strtoull(value, nullptr, 10);
    else if (flag == "--dim") o.dim = std::strtoull(value, nullptr, 10);
    else if (flag == "--clusters") o.clusters = std::strtoull(value, nullptr, 10);
    else if (flag == "--latent") o.latent = std::strtoull(value, nullptr, 10);
    else if (flag == "--queries") o.queries = std::strtoull(value, nullptr, 10);
    else if (flag == "--k") o.k = std::strtoull(value, nullptr, 10);
    else if (flag == "--R") o.index.R = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
    else if (flag == "--L") o.index.L = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
    else if (flag == "--alpha") o.index.alpha = std::strtof(value, nullptr);
    else if (flag == "--seed") o.index.seed = std::strtoull(value, nullptr, 10);
    else if (flag == "--repair") o.index.repair_orphans = std::strtoul(value, nullptr, 10) != 0;
    else if (flag == "--base") o.base_path = value;
    else if (flag == "--query") o.query_path = value;
    else if (flag == "--gt") o.gt_path = value;
    else if (flag == "--rows") o.rows = std::strtoull(value, nullptr, 10);
    else if (flag == "--threads") o.index.threads = static_cast<unsigned>(std::strtoul(value, nullptr, 10));
    else if (flag == "--qthreads") o.query_threads = static_cast<unsigned>(std::strtoul(value, nullptr, 10));
    else if (flag == "--save") o.save_path = value;
    else if (flag == "--load") o.load_path = value;
    else {
      std::fprintf(stderr, "unknown flag %s\n", flag.c_str());
      std::exit(2);
    }
  }
  return o;
}

static double percentile(std::vector<double> sorted_us, double p) {
  const auto idx = static_cast<std::size_t>(p * static_cast<double>(sorted_us.size() - 1));
  return sorted_us[idx];
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);  // show progress even when output goes to a file
  Options o = parse(argc, argv);
  if (o.index.threads == 0) o.index.threads = std::max(1u, std::thread::hardware_concurrency());

  Dataset base, queries;
  if (!o.base_path.empty()) {
    base = read_fvecs(o.base_path, o.rows);
    queries = read_fvecs(o.query_path, o.queries);
    std::printf("base: %zu x %zu (%s)\n", base.n, base.dim, o.base_path.c_str());
  } else {
    const Dataset all = o.latent != 0 ? make_low_rank(o.n + o.queries, o.dim, o.latent, 2026)
                                      : make_clustered(o.n + o.queries, o.dim, o.clusters, 2026);
    base = slice(all, 0, o.n);
    queries = slice(all, o.n, o.n + o.queries);
    if (o.latent != 0) {
      std::printf("base: %zu x %zu (synthetic, intrinsic dimension %zu)\n", base.n, base.dim, o.latent);
    } else {
      std::printf("base: %zu x %zu (synthetic, %zu tight clusters)\n", base.n, base.dim, o.clusters);
    }
  }
  std::printf("queries: %zu, k = %zu\n", queries.n, o.k);

  std::vector<std::vector<std::uint32_t>> truth;
  if (!o.gt_path.empty() && o.rows == 0) {
    truth = read_ivecs(o.gt_path);
    truth.resize(queries.n);
  } else {
    truth = exact_knn_batch(base, queries, o.k);
  }

  auto start = Clock::now();
  for (std::size_t q = 0; q < queries.n; ++q) (void)exact_knn(base, queries.row(q), o.k);
  const double brute_qps = static_cast<double>(queries.n) / seconds_since(start);
  std::printf("brute force: %.1f QPS (1 thread, %s distance)\n", brute_qps, distance_kernel());

  VamanaIndex index(base, o.index);
  if (!o.load_path.empty()) {
    index.load(o.load_path);
    std::printf("loaded %s, mean degree %.1f\n", o.load_path.c_str(), index.mean_degree());
  } else {
    start = Clock::now();
    index.build();
    std::printf("build: R=%u L=%u alpha=%.2f -> %.1f s (%u thread%s), mean degree %.1f\n", o.index.R,
                o.index.L, static_cast<double>(o.index.alpha), seconds_since(start), o.index.threads,
                o.index.threads == 1 ? "" : "s", index.mean_degree());
  }
  if (!o.save_path.empty()) index.save(o.save_path);
  std::printf("reachable from entry point: %zu / %zu\n\n", index.reachable_from_medoid(), base.n);

  std::printf("| L | recall@%zu | QPS | speedup | p50 us | p99 us | dist comps |\n", o.k);
  std::printf("|---|---|---|---|---|---|---|\n");
  std::vector<double> single_qps;
  for (const std::size_t L : {10, 20, 40, 80, 160, 320}) {
    if (L < o.k) continue;
    std::vector<std::vector<std::uint32_t>> found(queries.n);
    std::vector<double> latency_us(queries.n);
    std::uint64_t dist_comps = 0;

    const auto sweep_start = Clock::now();
    for (std::size_t q = 0; q < queries.n; ++q) {
      SearchStats stats;
      const auto t0 = Clock::now();
      const auto hits = index.search(queries.row(q), o.k, L, &stats);
      latency_us[q] = seconds_since(t0) * 1e6;
      dist_comps += stats.dist_comps;
      for (const Neighbor& h : hits) found[q].push_back(h.id);
    }
    const double qps = static_cast<double>(queries.n) / seconds_since(sweep_start);

    std::sort(latency_us.begin(), latency_us.end());
    std::printf("| %zu | %.4f | %.0f | %.1fx | %.1f | %.1f | %.0f |\n", L,
                recall_at_k(truth, found, o.k), qps, qps / brute_qps, percentile(latency_us, 0.50),
                percentile(latency_us, 0.99),
                static_cast<double>(dist_comps) / static_cast<double>(queries.n));
    single_qps.push_back(qps);
  }

  if (o.query_threads > 1) {
    // Throughput: the same queries over and over from several threads sharing one index.
    std::printf("\n| L | QPS (%u threads) | scaling |\n|---|---|---|\n", o.query_threads);
    std::size_t row = 0;
    for (const std::size_t L : {10, 20, 40, 80, 160, 320}) {
      if (L < o.k) continue;
      const std::size_t total = queries.n * 20;
      std::atomic<std::size_t> next{0};
      auto work = [&] {
        for (;;) {
          const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= total) return;
          (void)index.search(queries.row(i % queries.n), o.k, L);
        }
      };
      const auto t0 = Clock::now();
      std::vector<std::thread> pool;
      for (unsigned t = 1; t < o.query_threads; ++t) pool.emplace_back(work);
      work();
      for (std::thread& t : pool) t.join();
      const double qps = static_cast<double>(total) / seconds_since(t0);
      std::printf("| %zu | %.0f | %.1fx |\n", L, qps, qps / single_qps[row++]);
    }
  }
  return 0;
}
