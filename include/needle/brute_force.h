#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "needle/dataset.h"

namespace needle {

/// A search hit, ordered by distance and then id so results are deterministic.
struct Neighbor {
  float dist;
  std::uint32_t id;

  friend bool operator<(const Neighbor& a, const Neighbor& b) noexcept {
    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
  }
};

/// Exact k nearest neighbours of one query by scanning every base vector, closest first.
std::vector<Neighbor> exact_knn(const Dataset& base, const float* query, std::size_t k);

/// Exact top-k ids for every query, split across threads (0 = hardware concurrency).
std::vector<std::vector<std::uint32_t>> exact_knn_batch(const Dataset& base, const Dataset& queries,
                                                        std::size_t k, unsigned threads = 0);

/// Mean fraction of the true top-k ids that appear in the first k found ids.
double recall_at_k(const std::vector<std::vector<std::uint32_t>>& truth,
                   const std::vector<std::vector<std::uint32_t>>& found, std::size_t k);

}  // namespace needle
