#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <vector>

#include "needle/brute_force.h"
#include "needle/dataset.h"
#include "needle/distance.h"
#include "needle/vamana.h"

using namespace needle;

static int failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

static void test_distance() {
  const float a[] = {1.0f, 2.0f, 3.0f};
  const float b[] = {4.0f, 6.0f, 3.0f};
  CHECK(l2_sq(a, b, 3) == 25.0f);
  CHECK(l2_sq(a, a, 3) == 0.0f);
  CHECK(l2_sq(a, b, 3) == l2_sq(b, a, 3));
}

static void test_simd_matches_scalar() {
  // Odd sizes exercise the 16-wide loop, the 8-wide step and the scalar tail.
  for (const std::size_t dim : {1, 7, 8, 15, 16, 17, 31, 128, 131}) {
    const Dataset ds = make_clustered(40, dim, 4, 100 + dim);
    for (std::size_t i = 0; i + 1 < ds.n; ++i) {
      const float want = l2_sq_scalar(ds.row(i), ds.row(i + 1), dim);
      const float got = l2_sq(ds.row(i), ds.row(i + 1), dim);
      CHECK(std::fabs(got - want) <= 1e-4f * std::max(1.0f, want));
    }
  }
  std::printf("distance kernel: %s\n", distance_kernel());
}

static void test_parallel_build() {
  const Dataset all = make_low_rank(20'200, 32, 10, 77);
  const Dataset base = slice(all, 0, 20'000);
  const Dataset queries = slice(all, 20'000, 20'200);
  const auto truth = exact_knn_batch(base, queries, 10);

  VamanaIndex index(base, {.R = 24, .L = 64, .threads = 8});
  index.build();

  CHECK(index.reachable_from_medoid() == base.n);
  for (std::uint32_t i = 0; i < base.n; ++i) {
    const auto nbrs = index.neighbors(i);
    CHECK(nbrs.size() <= 24);
    const std::set<std::uint32_t> unique(nbrs.begin(), nbrs.end());
    CHECK(unique.size() == nbrs.size());
    CHECK(unique.count(i) == 0);
    for (const std::uint32_t nb : nbrs) CHECK(nb < base.n);
  }

  std::vector<std::vector<std::uint32_t>> found(queries.n);
  for (std::size_t q = 0; q < queries.n; ++q) {
    for (const Neighbor& h : index.search(queries.row(q), 10, 64)) found[q].push_back(h.id);
  }
  const double recall = recall_at_k(truth, found, 10);
  std::printf("recall@10 after an 8-thread build, L=64: %.4f\n", recall);
  CHECK(recall >= 0.95);
}

static void test_tight_clusters_stay_connected() {
  // Dense, well-separated blobs: a prune that only keeps the nearest candidates leaves no edges
  // between blobs, and a search that starts in the wrong one never gets out.
  const Dataset all = make_clustered(50'200, 64, 256, 2026);
  const Dataset base = slice(all, 0, 50'000);
  const Dataset queries = slice(all, 50'000, 50'200);
  const auto truth = exact_knn_batch(base, queries, 10);

  VamanaIndex index(base, {.threads = 0});
  index.build();
  std::vector<std::vector<std::uint32_t>> found(queries.n);
  for (std::size_t q = 0; q < queries.n; ++q) {
    for (const Neighbor& h : index.search(queries.row(q), 10, 80)) found[q].push_back(h.id);
  }
  const double recall = recall_at_k(truth, found, 10);
  std::printf("recall@10 on 50000x64 tight clusters, L=80: %.4f\n", recall);
  CHECK(recall >= 0.95);
}

static void test_save_load_round_trip() {
  const Dataset base = make_clustered(3000, 16, 20, 8);
  VamanaIndex built(base, {.R = 16, .L = 40});
  built.build();
  const auto path = (std::filesystem::temp_directory_path() / "needle_round_trip.idx").string();
  built.save(path);

  VamanaIndex loaded(base, {.R = 16, .L = 40});
  loaded.load(path);
  CHECK(loaded.medoid() == built.medoid());
  for (std::uint32_t i = 0; i < base.n; ++i) {
    const auto a = built.neighbors(i);
    const auto b = loaded.neighbors(i);
    CHECK(std::equal(a.begin(), a.end(), b.begin(), b.end()));
  }
  const auto want = built.search(base.row(7), 10, 40);
  const auto got = loaded.search(base.row(7), 10, 40);
  CHECK(want.size() == got.size());
  for (std::size_t i = 0; i < want.size(); ++i) CHECK(want[i].id == got[i].id);

  // A file for another dataset, or a damaged one, must be refused instead of read.
  const Dataset other = make_clustered(2999, 16, 20, 8);
  VamanaIndex wrong(other, {.R = 16, .L = 40});
  bool refused = false;
  try { wrong.load(path); } catch (const std::runtime_error&) { refused = true; }
  CHECK(refused);

  std::filesystem::resize_file(path, std::filesystem::file_size(path) / 2);
  refused = false;
  try { loaded.load(path); } catch (const std::runtime_error&) { refused = true; }
  CHECK(refused);
  std::filesystem::remove(path);
}

static void test_exact_knn_matches_full_sort() {
  const Dataset base = make_clustered(500, 8, 5, 1);
  const Dataset queries = make_clustered(20, 8, 5, 2);
  for (std::size_t q = 0; q < queries.n; ++q) {
    std::vector<Neighbor> all;
    for (std::size_t i = 0; i < base.n; ++i) {
      all.push_back({l2_sq(queries.row(q), base.row(i), base.dim), static_cast<std::uint32_t>(i)});
    }
    std::sort(all.begin(), all.end());
    const auto got = exact_knn(base, queries.row(q), 10);
    CHECK(got.size() == 10);
    for (std::size_t i = 0; i < got.size(); ++i) CHECK(got[i].id == all[i].id);
  }
  CHECK(exact_knn(base, queries.row(0), 0).empty());
  CHECK(exact_knn(base, queries.row(0), 10'000).size() == base.n);
}

static void test_fvecs_round_trip() {
  const Dataset ds = make_clustered(37, 5, 3, 7);
  const auto path = (std::filesystem::temp_directory_path() / "needle_round_trip.fvecs").string();
  write_fvecs(path, ds);
  const Dataset back = read_fvecs(path);
  CHECK(back.n == ds.n);
  CHECK(back.dim == ds.dim);
  CHECK(back.data == ds.data);
  CHECK(read_fvecs(path, 10).n == 10);
  std::filesystem::remove(path);
}

static void test_recall_metric() {
  const std::vector<std::vector<std::uint32_t>> truth = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const std::vector<std::vector<std::uint32_t>> found = {{4, 3, 2, 1}, {5, 6, 0, 0}};
  CHECK(recall_at_k(truth, truth, 4) == 1.0);
  CHECK(recall_at_k(truth, found, 4) == 0.75);
  CHECK(recall_at_k(truth, found, 2) == 0.5);
}

static void test_graph_invariants() {
  const Dataset base = make_clustered(3000, 16, 20, 3);
  VamanaIndex index(base, {.R = 16, .L = 40, .alpha = 1.2f, .seed = 9});
  index.build();

  CHECK(index.reachable_from_medoid() == base.n);
  for (std::uint32_t i = 0; i < base.n; ++i) {
    const auto nbrs = index.neighbors(i);
    CHECK(nbrs.size() <= 16);
    CHECK(!nbrs.empty());
    const std::set<std::uint32_t> unique(nbrs.begin(), nbrs.end());
    CHECK(unique.size() == nbrs.size());
    CHECK(unique.count(i) == 0);
    for (const std::uint32_t nb : nbrs) CHECK(nb < base.n);
  }
}

static void test_repair_leaves_no_orphans() {
  const Dataset base = make_clustered(20'000, 64, 256, 2026);
  VamanaIndex raw(base, {.repair_orphans = false});
  VamanaIndex repaired(base, {});
  raw.build();
  repaired.build();

  std::printf("reachable without repair: %zu / %zu\n", raw.reachable_from_medoid(), base.n);
  CHECK(repaired.reachable_from_medoid() == base.n);
  for (std::uint32_t i = 0; i < base.n; ++i) {
    const auto nbrs = repaired.neighbors(i);
    const std::set<std::uint32_t> unique(nbrs.begin(), nbrs.end());
    CHECK(unique.size() == nbrs.size());
    CHECK(unique.count(i) == 0);
  }
}

static void test_build_is_deterministic() {
  const Dataset base = make_clustered(2000, 16, 20, 4);
  const VamanaParams params{.R = 16, .L = 40, .alpha = 1.2f, .seed = 5};
  VamanaIndex a(base, params);
  VamanaIndex b(base, params);
  a.build();
  b.build();

  CHECK(a.medoid() == b.medoid());
  for (std::uint32_t i = 0; i < base.n; ++i) {
    const auto na = a.neighbors(i);
    const auto nb = b.neighbors(i);
    CHECK(std::equal(na.begin(), na.end(), nb.begin(), nb.end()));
  }
}

static void test_recall_on_clustered_data() {
  const Dataset all = make_clustered(5200, 16, 30, 11);
  const Dataset base = slice(all, 0, 5000);
  const Dataset queries = slice(all, 5000, 5200);
  const auto truth = exact_knn_batch(base, queries, 10);

  VamanaIndex index(base, {.R = 24, .L = 64, .alpha = 1.2f, .seed = 1});
  index.build();

  std::vector<std::vector<std::uint32_t>> found(queries.n);
  for (std::size_t q = 0; q < queries.n; ++q) {
    const auto hits = index.search(queries.row(q), 10, 64);
    CHECK(hits.size() == 10);
    CHECK(std::is_sorted(hits.begin(), hits.end()));
    for (const Neighbor& h : hits) found[q].push_back(h.id);
  }
  const double recall = recall_at_k(truth, found, 10);
  std::printf("recall@10 on 5000x16 clustered, L=64: %.4f\n", recall);
  CHECK(recall >= 0.95);
}

static void test_base_point_finds_itself() {
  const Dataset base = make_clustered(2000, 16, 20, 6);
  VamanaIndex index(base, {.R = 16, .L = 40, .alpha = 1.2f, .seed = 2});
  index.build();

  std::size_t found_self = 0;
  for (std::uint32_t i = 0; i < base.n; ++i) {
    SearchStats stats;
    const auto hits = index.search(base.row(i), 1, 40, &stats);
    if (!hits.empty() && hits[0].dist == 0.0f) ++found_self;
    CHECK(stats.dist_comps < base.n);
  }
  CHECK(found_self >= base.n * 99 / 100);
}

int main() {
  test_distance();
  test_simd_matches_scalar();
  test_parallel_build();
  test_tight_clusters_stay_connected();
  test_save_load_round_trip();
  test_exact_knn_matches_full_sort();
  test_fvecs_round_trip();
  test_recall_metric();
  test_graph_invariants();
  test_repair_leaves_no_orphans();
  test_build_is_deterministic();
  test_recall_on_clustered_data();
  test_base_point_finds_itself();

  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::puts("all tests passed");
  return 0;
}
