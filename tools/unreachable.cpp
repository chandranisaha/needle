// Tests one explanation for unreachable nodes: they are surplus members of exact-duplicate groups.
//   needle_unreachable base.fvecs [rows] [nobuild]
// rows == 0 loads everything; "nobuild" only reports duplicate statistics.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "needle/dataset.h"
#include "needle/vamana.h"

using namespace needle;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: needle_unreachable base.fvecs [rows] [nobuild]\n");
    return 2;
  }
  const std::size_t rows = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 0;
  const bool build = !(argc > 3 && std::string(argv[3]) == "nobuild");

  const Dataset base = read_fvecs(argv[1], rows);
  const std::size_t n = base.n;
  const std::size_t bytes = base.dim * sizeof(float);
  std::printf("loaded %zu x %zu\n", n, base.dim);

  // Sort ids by raw vector bytes so identical vectors end up adjacent, lowest id first.
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
    const int c = std::memcmp(base.row(a), base.row(b), bytes);
    return c < 0 || (c == 0 && a < b);
  });

  std::vector<std::uint32_t> rank(n, 0);        // position inside its duplicate group
  std::vector<std::uint32_t> group_size(n, 1);
  std::map<std::size_t, std::size_t> histogram;  // group size -> number of groups
  std::size_t surplus = 0;                       // members beyond the first two of each group
  for (std::size_t i = 0; i < n;) {
    std::size_t j = i + 1;
    while (j < n && std::memcmp(base.row(order[i]), base.row(order[j]), bytes) == 0) ++j;
    const std::size_t g = j - i;
    if (g > 1) ++histogram[g];
    if (g > 2) surplus += g - 2;
    for (std::size_t t = i; t < j; ++t) {
      rank[order[t]] = static_cast<std::uint32_t>(t - i);
      group_size[order[t]] = static_cast<std::uint32_t>(g);
    }
    i = j;
  }

  std::printf("duplicate groups by size:");
  for (const auto& [size, count] : histogram) std::printf(" %zux%zu", count, size);
  std::printf("\nmembers beyond the first two of their group: %zu\n", surplus);
  if (!build) return 0;

  VamanaIndex index(base, {.repair_orphans = false});
  index.build();

  std::vector<char> seen(n, 0);
  std::vector<std::uint32_t> in_degree(n, 0);
  for (std::uint32_t v = 0; v < n; ++v) {
    for (const std::uint32_t nb : index.neighbors(v)) ++in_degree[nb];
  }
  std::vector<std::uint32_t> frontier{index.medoid()};
  seen[index.medoid()] = 1;
  while (!frontier.empty()) {
    const std::uint32_t v = frontier.back();
    frontier.pop_back();
    for (const std::uint32_t nb : index.neighbors(v)) {
      if (seen[nb]) continue;
      seen[nb] = 1;
      frontier.push_back(nb);
    }
  }

  std::size_t unreachable = 0, third_or_later = 0, first_two = 0, not_duplicate = 0, no_in_edges = 0;
  std::size_t surplus_but_reachable = 0;
  for (std::uint32_t v = 0; v < n; ++v) {
    if (seen[v]) {
      if (rank[v] >= 2) ++surplus_but_reachable;
      continue;
    }
    ++unreachable;
    if (in_degree[v] == 0) ++no_in_edges;
    if (group_size[v] == 1) ++not_duplicate;
    else if (rank[v] >= 2) ++third_or_later;
    else ++first_two;
  }

  std::printf("unreachable nodes: %zu (%zu have no in-edges)\n", unreachable, no_in_edges);
  std::printf("  third or later member of a duplicate group: %zu\n", third_or_later);
  std::printf("  first or second member of a duplicate group: %zu\n", first_two);
  std::printf("  not a duplicate of anything: %zu\n", not_duplicate);
  std::printf("third-or-later duplicates that are still reachable: %zu\n", surplus_but_reachable);

  VamanaIndex repaired(base, {});
  repaired.build();
  std::printf("with the repair pass: %zu / %zu reachable\n", repaired.reachable_from_medoid(), n);
  return 0;
}
