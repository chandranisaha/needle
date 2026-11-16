#include "needle/brute_force.h"

#include <algorithm>
#include <atomic>
#include <thread>

#include "needle/distance.h"

namespace needle {

std::vector<Neighbor> exact_knn(const Dataset& base, const float* query, std::size_t k) {
  k = std::min(k, base.n);
  if (k == 0) return {};

  // Max-heap of the k best so far; the root is the one to beat.
  std::vector<Neighbor> heap;
  heap.reserve(k);
  for (std::size_t i = 0; i < base.n; ++i) {
    const Neighbor cand{l2_sq(query, base.row(i), base.dim), static_cast<std::uint32_t>(i)};
    if (heap.size() < k) {
      heap.push_back(cand);
      std::push_heap(heap.begin(), heap.end());
    } else if (cand < heap.front()) {
      std::pop_heap(heap.begin(), heap.end());
      heap.back() = cand;
      std::push_heap(heap.begin(), heap.end());
    }
  }
  std::sort_heap(heap.begin(), heap.end());
  return heap;
}

std::vector<std::vector<std::uint32_t>> exact_knn_batch(const Dataset& base, const Dataset& queries,
                                                        std::size_t k, unsigned threads) {
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());

  std::vector<std::vector<std::uint32_t>> out(queries.n);
  std::atomic<std::size_t> next{0};
  auto work = [&] {
    for (std::size_t q = next.fetch_add(1); q < queries.n; q = next.fetch_add(1)) {
      const auto hits = exact_knn(base, queries.row(q), k);
      out[q].reserve(hits.size());
      for (const Neighbor& h : hits) out[q].push_back(h.id);
    }
  };

  std::vector<std::thread> pool;
  for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
  work();
  for (std::thread& t : pool) t.join();
  return out;
}

double recall_at_k(const std::vector<std::vector<std::uint32_t>>& truth,
                   const std::vector<std::vector<std::uint32_t>>& found, std::size_t k) {
  if (truth.empty() || k == 0) return 0.0;

  std::size_t hits = 0;
  std::size_t total = 0;
  for (std::size_t q = 0; q < truth.size(); ++q) {
    const std::size_t kt = std::min(k, truth[q].size());
    const std::size_t kf = std::min(k, found[q].size());
    for (std::size_t i = 0; i < kt; ++i) {
      const auto end = found[q].begin() + static_cast<std::ptrdiff_t>(kf);
      if (std::find(found[q].begin(), end, truth[q][i]) != end) ++hits;
    }
    total += kt;
  }
  return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
}

}  // namespace needle
