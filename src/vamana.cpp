#include "needle/vamana.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>

#include "needle/distance.h"

namespace needle {
namespace {

struct Candidate {
  float dist;
  std::uint32_t id;
  bool expanded;
};

/// Per-thread working memory, reused across calls so searches and inserts allocate nothing.
struct Scratch {
  std::vector<std::uint32_t> stamp;
  std::uint32_t epoch = 0;
  std::vector<Candidate> beam;
  std::vector<std::uint32_t> nbrs;  // a neighbour list copied out from under its lock
  std::vector<Neighbor> pool, back_pool;
  std::vector<std::uint32_t> kept, back_kept;
  std::vector<char> pruned;
  std::vector<float> occlusion;

  void begin(std::size_t n) {
    if (stamp.size() < n) stamp.resize(n, 0);
    if (++epoch == 0) {
      std::fill(stamp.begin(), stamp.end(), 0u);
      epoch = 1;
    }
    beam.clear();
  }

  /// True the first time id is seen in the current search.
  bool visit(std::uint32_t id) noexcept {
    if (stamp[id] == epoch) return false;
    stamp[id] = epoch;
    return true;
  }
};

Scratch& scratch() {
  thread_local Scratch s;
  return s;
}

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#else
  std::this_thread::yield();
#endif
}

/// Holds one node's spinlock for a scope. A null flag (single-threaded build, or any search after
/// the build) makes it a no-op. A thread never holds two of these at once, so they cannot deadlock.
class NodeLock {
 public:
  explicit NodeLock(std::atomic_flag* flag) noexcept : flag_(flag) {
    if (flag_ == nullptr) return;
    while (flag_->test_and_set(std::memory_order_acquire)) {
      while (flag_->test(std::memory_order_relaxed)) cpu_relax();
    }
  }
  ~NodeLock() {
    if (flag_ != nullptr) flag_->clear(std::memory_order_release);
  }
  NodeLock(const NodeLock&) = delete;
  NodeLock& operator=(const NodeLock&) = delete;

 private:
  std::atomic_flag* flag_;
};

inline void prefetch(const float* row) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(row);
  __builtin_prefetch(row + 16);
#else
  (void)row;
#endif
}

/// Runs fn(i) for i in [0, count) on the given number of threads, handing out work in small blocks.
template <typename Fn>
void parallel_for(std::size_t count, unsigned threads, Fn fn) {
  if (threads <= 1) {
    for (std::size_t i = 0; i < count; ++i) fn(i);
    return;
  }
  constexpr std::size_t block = 64;
  std::atomic<std::size_t> next{0};
  auto work = [&] {
    for (;;) {
      const std::size_t begin = next.fetch_add(block, std::memory_order_relaxed);
      if (begin >= count) return;
      const std::size_t end = std::min(count, begin + block);
      for (std::size_t i = begin; i < end; ++i) fn(i);
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
  work();
  for (std::thread& t : pool) t.join();
}

constexpr std::uint32_t file_magic = 0x314C444Eu;  // "NDL1"

struct FileHeader {
  std::uint32_t magic;
  std::uint32_t R;
  std::uint64_t n;
  std::uint64_t dim;
  std::uint32_t medoid;
  std::uint32_t reserved;
};

}  // namespace

VamanaIndex::VamanaIndex(const Dataset& base, VamanaParams params) : base_(base), params_(params) {
  if (params_.R == 0 || params_.L == 0) throw std::invalid_argument("R and L must be positive");
  if (params_.alpha < 1.0f) throw std::invalid_argument("alpha must be >= 1");
}

std::span<const std::uint32_t> VamanaIndex::neighbors(std::uint32_t id) const noexcept {
  return {neighbors_.data() + static_cast<std::size_t>(id) * params_.R, degree_[id]};
}

double VamanaIndex::mean_degree() const noexcept {
  if (degree_.empty()) return 0.0;
  std::uint64_t total = 0;
  for (std::uint32_t d : degree_) total += d;
  return static_cast<double>(total) / static_cast<double>(degree_.size());
}

std::size_t VamanaIndex::reachable_from_medoid() const {
  if (degree_.empty()) return 0;

  std::vector<char> seen(base_.n, 0);
  std::vector<std::uint32_t> frontier{medoid_};
  seen[medoid_] = 1;
  std::size_t count = 1;
  while (!frontier.empty()) {
    const std::uint32_t v = frontier.back();
    frontier.pop_back();
    for (const std::uint32_t nb : neighbors(v)) {
      if (seen[nb]) continue;
      seen[nb] = 1;
      ++count;
      frontier.push_back(nb);
    }
  }
  return count;
}

std::uint32_t VamanaIndex::find_medoid() const {
  std::vector<double> sum(base_.dim, 0.0);
  for (std::size_t i = 0; i < base_.n; ++i) {
    const float* v = base_.row(i);
    for (std::size_t d = 0; d < base_.dim; ++d) sum[d] += v[d];
  }
  std::vector<float> centroid(base_.dim);
  for (std::size_t d = 0; d < base_.dim; ++d) {
    centroid[d] = static_cast<float>(sum[d] / static_cast<double>(base_.n));
  }

  std::uint32_t best = 0;
  float best_dist = l2_sq(centroid.data(), base_.row(0), base_.dim);
  for (std::size_t i = 1; i < base_.n; ++i) {
    const float d = l2_sq(centroid.data(), base_.row(i), base_.dim);
    if (d < best_dist) {
      best_dist = d;
      best = static_cast<std::uint32_t>(i);
    }
  }
  return best;
}

void VamanaIndex::build() {
  if (base_.n == 0) throw std::invalid_argument("cannot index an empty dataset");

  neighbors_.assign(base_.n * params_.R, 0u);
  degree_.assign(base_.n, 0u);
  medoid_ = find_medoid();

  std::vector<std::uint32_t> order(base_.n);
  for (std::size_t i = 0; i < base_.n; ++i) order[i] = static_cast<std::uint32_t>(i);
  std::mt19937_64 rng(params_.seed);
  for (std::size_t i = base_.n - 1; i > 0; --i) std::swap(order[i], order[rng() % (i + 1)]);

  unsigned threads = params_.threads != 0 ? params_.threads : std::thread::hardware_concurrency();
  threads = std::max(1u, threads);
  if (threads > 1) locks_ = std::make_unique<std::atomic_flag[]>(base_.n);

  for (const float alpha : {1.0f, params_.alpha}) {
    parallel_for(base_.n, threads, [&](std::size_t i) { insert_point(order[i], alpha); });
  }
  locks_.reset();
  if (params_.repair_orphans) repair_orphans();
}

bool VamanaIndex::attach_orphan(std::uint32_t p, std::vector<std::uint32_t>& in_degree) {
  const std::size_t R = params_.R;
  beam_search(base_.row(p), params_.L, nullptr, nullptr);

  // Everything in the beam is reachable. Try the nearest first.
  for (const Candidate& c : scratch().beam) {
    const std::uint32_t v = c.id;
    std::uint32_t* nv = neighbors_.data() + static_cast<std::size_t>(v) * R;
    const std::uint32_t dv = degree_[v];
    if (dv < R) {
      nv[dv] = p;
      degree_[v] = dv + 1;
      ++in_degree[p];
      return true;
    }

    // v is full: give up its farthest neighbour, but only one that has another way in.
    const float* vv = base_.row(v);
    std::uint32_t slot = dv;
    float farthest = -1.0f;
    for (std::uint32_t t = 0; t < dv; ++t) {
      if (in_degree[nv[t]] < 2) continue;
      const float d = l2_sq(vv, base_.row(nv[t]), base_.dim);
      if (d > farthest) {
        farthest = d;
        slot = t;
      }
    }
    if (slot == dv) continue;
    --in_degree[nv[slot]];
    nv[slot] = p;
    ++in_degree[p];
    return true;
  }
  return false;
}

std::size_t VamanaIndex::repair_orphans() {
  const std::size_t n = base_.n;
  std::vector<std::uint32_t> in_degree(n, 0);
  for (std::uint32_t v = 0; v < n; ++v) {
    for (const std::uint32_t nb : neighbors(v)) ++in_degree[nb];
  }

  std::vector<char> seen;
  std::vector<std::uint32_t> frontier;
  auto flood = [&](std::uint32_t start) {
    seen[start] = 1;
    frontier.push_back(start);
    while (!frontier.empty()) {
      const std::uint32_t v = frontier.back();
      frontier.pop_back();
      for (const std::uint32_t nb : neighbors(v)) {
        if (seen[nb]) continue;
        seen[nb] = 1;
        frontier.push_back(nb);
      }
    }
  };

  // An eviction can cut off the evicted node, so re-check until a round changes nothing.
  std::size_t repaired = 0;
  for (int round = 0; round < 8; ++round) {
    seen.assign(n, 0);
    flood(medoid_);
    bool changed = false;
    for (std::uint32_t p = 0; p < n; ++p) {
      if (seen[p] || !attach_orphan(p, in_degree)) continue;
      flood(p);
      ++repaired;
      changed = true;
    }
    if (!changed) break;
  }
  return repaired;
}

void VamanaIndex::beam_search(const float* query, std::size_t L, std::vector<Neighbor>* expanded,
                              SearchStats* stats) const {
  Scratch& s = scratch();
  s.begin(base_.n);
  std::vector<Candidate>& beam = s.beam;
  SearchStats local;
  const bool building = locks_ != nullptr;

  s.visit(medoid_);
  beam.push_back({l2_sq(query, base_.row(medoid_), base_.dim), medoid_, false});
  ++local.dist_comps;

  // Invariant: every beam entry before cursor has been expanded.
  std::size_t cursor = 0;
  while (cursor < beam.size()) {
    if (beam[cursor].expanded) {
      ++cursor;
      continue;
    }
    beam[cursor].expanded = true;
    const Candidate cur = beam[cursor];
    if (expanded != nullptr) expanded->push_back({cur.dist, cur.id});
    ++local.hops;

    // Other threads rewrite neighbour lists during a threaded build, so read a private copy then.
    std::span<const std::uint32_t> nbrs;
    if (building) {
      {
        const NodeLock guard(lock_of(cur.id));
        const auto live = neighbors(cur.id);
        s.nbrs.assign(live.begin(), live.end());
      }
      nbrs = s.nbrs;
    } else {
      nbrs = neighbors(cur.id);
    }
    for (const std::uint32_t nb : nbrs) prefetch(base_.row(nb));

    std::size_t next = cursor + 1;
    for (const std::uint32_t nb : nbrs) {
      if (!s.visit(nb)) continue;
      const float d = l2_sq(query, base_.row(nb), base_.dim);
      ++local.dist_comps;
      if (beam.size() == L && d >= beam.back().dist) continue;

      const auto it = std::upper_bound(beam.begin(), beam.end(), d,
                                       [](float v, const Candidate& c) { return v < c.dist; });
      const auto pos = static_cast<std::size_t>(it - beam.begin());
      beam.insert(it, {d, nb, false});
      if (beam.size() > L) beam.pop_back();
      next = std::min(next, pos);
    }
    cursor = next;
  }

  if (stats != nullptr) *stats = local;
}

std::vector<Neighbor> VamanaIndex::search(const float* query, std::size_t k, std::size_t L,
                                          SearchStats* stats) const {
  if (degree_.empty()) throw std::logic_error("search before build");

  beam_search(query, std::max(L, k), nullptr, stats);
  const std::vector<Candidate>& beam = scratch().beam;
  std::vector<Neighbor> out;
  out.reserve(std::min(k, beam.size()));
  for (std::size_t i = 0; i < beam.size() && i < k; ++i) out.push_back({beam[i].dist, beam[i].id});
  return out;
}

void VamanaIndex::robust_prune(std::vector<Neighbor>& pool, float alpha,
                               std::vector<std::uint32_t>& kept) const {
  std::sort(pool.begin(), pool.end());
  pool.erase(std::unique(pool.begin(), pool.end(),
                         [](const Neighbor& a, const Neighbor& b) { return a.id == b.id; }),
             pool.end());

  // occlusion[j] is the largest d(p, j) / d(i, j) over kept neighbours i: how strongly some kept
  // neighbour already covers j's direction. Distances are squared, so it is compared with alpha^2.
  //
  // The paper keeps j whenever occlusion < alpha in one sweep. With alpha > 1 that fills all R slots
  // with the nearest candidates and never reaches the farther, more spread-out ones. So sweep twice:
  // first take only candidates nothing covers (the alpha = 1 rule), then spend whatever slots are
  // left on the ones alpha lets through.
  Scratch& s = scratch();
  s.occlusion.assign(pool.size(), 0.0f);
  s.pruned.assign(pool.size(), 0);  // here: already kept
  kept.clear();
  const float alpha_sq = alpha * alpha;

  for (const float limit : {1.0f, alpha_sq}) {
    for (std::size_t i = 0; i < pool.size() && kept.size() < params_.R; ++i) {
      if (s.pruned[i] || s.occlusion[i] >= limit) continue;
      s.pruned[i] = 1;
      kept.push_back(pool[i].id);
      const float* keeper = base_.row(pool[i].id);
      for (std::size_t j = i + 1; j < pool.size(); ++j) {
        if (s.pruned[j] || s.occlusion[j] >= alpha_sq) continue;
        const float between = l2_sq(keeper, base_.row(pool[j].id), base_.dim);
        const float ratio = between > 0.0f ? pool[j].dist / between : std::numeric_limits<float>::max();
        s.occlusion[j] = std::max(s.occlusion[j], ratio);
      }
    }
    if (alpha_sq <= 1.0f) break;
  }
}

void VamanaIndex::insert_point(std::uint32_t p, float alpha) {
  Scratch& s = scratch();
  const float* vp = base_.row(p);
  const std::size_t R = params_.R;
  std::uint32_t* np = neighbors_.data() + static_cast<std::size_t>(p) * R;

  s.pool.clear();
  beam_search(vp, params_.L, &s.pool, nullptr);
  {
    const NodeLock guard(lock_of(p));
    s.nbrs.assign(np, np + degree_[p]);
  }
  for (const std::uint32_t nb : s.nbrs) s.pool.push_back({l2_sq(vp, base_.row(nb), base_.dim), nb});
  std::erase_if(s.pool, [p](const Neighbor& c) { return c.id == p; });
  robust_prune(s.pool, alpha, s.kept);
  {
    const NodeLock guard(lock_of(p));
    std::copy(s.kept.begin(), s.kept.end(), np);
    degree_[p] = static_cast<std::uint32_t>(s.kept.size());
  }

  // Add the reverse edge j -> p, re-pruning j when it is already full.
  for (const std::uint32_t j : s.kept) {
    const NodeLock guard(lock_of(j));
    std::uint32_t* nj = neighbors_.data() + static_cast<std::size_t>(j) * R;
    const std::uint32_t dj = degree_[j];
    if (std::find(nj, nj + dj, p) != nj + dj) continue;

    if (dj < R) {
      nj[dj] = p;
      degree_[j] = dj + 1;
      continue;
    }
    const float* vj = base_.row(j);
    s.back_pool.clear();
    for (std::uint32_t t = 0; t < dj; ++t) s.back_pool.push_back({l2_sq(vj, base_.row(nj[t]), base_.dim), nj[t]});
    s.back_pool.push_back({l2_sq(vj, vp, base_.dim), p});
    robust_prune(s.back_pool, alpha, s.back_kept);
    std::copy(s.back_kept.begin(), s.back_kept.end(), nj);
    degree_[j] = static_cast<std::uint32_t>(s.back_kept.size());
  }
}

void VamanaIndex::save(const std::string& path) const {
  if (degree_.empty()) throw std::logic_error("save before build");
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open " + path + " for writing");

  const FileHeader header{file_magic, params_.R, base_.n, base_.dim, medoid_, 0};
  out.write(reinterpret_cast<const char*>(&header), sizeof header);
  out.write(reinterpret_cast<const char*>(degree_.data()),
            static_cast<std::streamsize>(degree_.size() * sizeof(std::uint32_t)));
  out.write(reinterpret_cast<const char*>(neighbors_.data()),
            static_cast<std::streamsize>(neighbors_.size() * sizeof(std::uint32_t)));
  if (!out) throw std::runtime_error("write failed: " + path);
}

void VamanaIndex::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);

  FileHeader header{};
  in.read(reinterpret_cast<char*>(&header), sizeof header);
  if (!in || header.magic != file_magic) throw std::runtime_error(path + " is not a needle index");
  if (header.n != base_.n || header.dim != base_.dim) {
    throw std::runtime_error(path + " was built from a different dataset");
  }
  if (header.R == 0 || header.medoid >= header.n) throw std::runtime_error(path + " is corrupt");

  std::vector<std::uint32_t> degree(header.n);
  std::vector<std::uint32_t> neighbors(header.n * header.R);
  in.read(reinterpret_cast<char*>(degree.data()),
          static_cast<std::streamsize>(degree.size() * sizeof(std::uint32_t)));
  in.read(reinterpret_cast<char*>(neighbors.data()),
          static_cast<std::streamsize>(neighbors.size() * sizeof(std::uint32_t)));
  if (!in) throw std::runtime_error(path + " is truncated");

  // A bad id would send a search out of bounds, so check every edge before accepting the file.
  for (std::size_t i = 0; i < header.n; ++i) {
    if (degree[i] > header.R) throw std::runtime_error(path + " is corrupt");
    for (std::uint32_t t = 0; t < degree[i]; ++t) {
      if (neighbors[i * header.R + t] >= header.n) throw std::runtime_error(path + " is corrupt");
    }
  }

  params_.R = header.R;
  medoid_ = header.medoid;
  degree_ = std::move(degree);
  neighbors_ = std::move(neighbors);
}

}  // namespace needle
