#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace needle {

/// Runs fn(i) for i in [0, count) on the given number of threads, handing out work in blocks.
/// threads == 0 means one per core.
template <typename Fn>
void parallel_for(std::size_t count, unsigned threads, std::size_t block, Fn fn) {
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  if (threads <= 1 || count <= block) {
    for (std::size_t i = 0; i < count; ++i) fn(i);
    return;
  }
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

}  // namespace needle
