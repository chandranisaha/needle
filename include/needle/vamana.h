#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "needle/brute_force.h"
#include "needle/dataset.h"

namespace needle {

struct VamanaParams {
  std::uint32_t R = 32;     // max out-degree
  std::uint32_t L = 64;     // beam width while building
  float alpha = 1.2f;       // >1 keeps longer edges, which shortens search paths
  std::uint64_t seed = 42;  // insertion order
  bool repair_orphans = true;  // reconnect nodes that pruning left with no way in
  unsigned threads = 1;     // build threads; 0 = all cores. Only 1 gives a reproducible graph.
};

struct SearchStats {
  std::uint32_t dist_comps = 0;
  std::uint32_t hops = 0;
};

/// Single-layer proximity graph from the DiskANN paper (Subramanya et al., NeurIPS 2019).
class VamanaIndex {
 public:
  /// Keeps a reference to base; it must outlive the index.
  VamanaIndex(const Dataset& base, VamanaParams params);

  /// Two insertion passes over a random order: alpha = 1, then the configured alpha.
  void build();

  /// Approximate k nearest neighbours, closest first. Larger L trades speed for recall.
  /// Safe to call from many threads once the index is built.
  std::vector<Neighbor> search(const float* query, std::size_t k, std::size_t L,
                               SearchStats* stats = nullptr) const;

  /// Writes the graph (not the vectors) to a file.
  void save(const std::string& path) const;
  /// Reads a graph written by save(). The index must wrap the same vectors it was built from.
  void load(const std::string& path);

  std::uint32_t medoid() const noexcept { return medoid_; }
  std::span<const std::uint32_t> neighbors(std::uint32_t id) const noexcept;
  double mean_degree() const noexcept;
  /// Number of nodes a search can reach at all; anything below n means lost recall.
  std::size_t reachable_from_medoid() const;
  const VamanaParams& params() const noexcept { return params_; }

 private:
  void beam_search(const float* query, std::size_t L, std::vector<Neighbor>* expanded,
                   SearchStats* stats) const;
  void robust_prune(std::vector<Neighbor>& pool, float alpha, std::vector<std::uint32_t>& kept) const;
  void insert_point(std::uint32_t p, float alpha);
  std::size_t repair_orphans();
  bool attach_orphan(std::uint32_t p, std::vector<std::uint32_t>& in_degree);
  std::uint32_t find_medoid() const;
  std::atomic_flag* lock_of(std::uint32_t id) const noexcept { return locks_ ? &locks_[id] : nullptr; }

  const Dataset& base_;
  VamanaParams params_;
  std::uint32_t medoid_ = 0;
  std::vector<std::uint32_t> neighbors_;  // n * R slots, first degree_[i] of each row are valid
  std::vector<std::uint32_t> degree_;
  std::unique_ptr<std::atomic_flag[]> locks_;  // one spinlock per node, alive only in a threaded build
};

}  // namespace needle
