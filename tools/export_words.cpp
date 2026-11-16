// Builds an index over word vectors and writes it, with a 2-D map of the words, as a JS file for
// the demo's Type tab. Input comes from tools/prep_words.py.
//   needle_export_words words.fvecs words.txt out.js [R] [L]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "needle/brute_force.h"
#include "needle/dataset.h"
#include "needle/vamana.h"

using namespace needle;

namespace {

std::string base64(const std::uint8_t* bytes, std::size_t n) {
  static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((n + 2) / 3 * 4);
  for (std::size_t i = 0; i < n; i += 3) {
    const std::uint32_t b0 = bytes[i];
    const std::uint32_t b1 = i + 1 < n ? bytes[i + 1] : 0;
    const std::uint32_t b2 = i + 2 < n ? bytes[i + 2] : 0;
    const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
    out.push_back(table[(triple >> 18) & 63]);
    out.push_back(table[(triple >> 12) & 63]);
    out.push_back(i + 1 < n ? table[(triple >> 6) & 63] : '=');
    out.push_back(i + 2 < n ? table[triple & 63] : '=');
  }
  return out;
}

template <typename T>
std::string base64(const std::vector<T>& values) {
  return base64(reinterpret_cast<const std::uint8_t*>(values.data()), values.size() * sizeof(T));
}

/// 2-D layout that keeps each word near its k nearest neighbours: neighbours attract, random
/// pairs repel, with the 1 / (1 + d^2) similarity curve t-SNE and UMAP use.
std::vector<float> layout(const Dataset& base, const std::vector<std::vector<std::uint32_t>>& knn,
                          std::size_t epochs, std::uint64_t seed) {
  const std::size_t n = base.n;
  std::vector<float> xy(n * 2);
  float extent = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    extent = std::max({extent, std::fabs(base.row(i)[0]), std::fabs(base.row(i)[1])});
  }
  for (std::size_t i = 0; i < n; ++i) {  // start from the two strongest PCA directions
    xy[i * 2] = 10.0f * base.row(i)[0] / extent;
    xy[i * 2 + 1] = 10.0f * base.row(i)[1] / extent;
  }

  std::mt19937_64 rng(seed);
  const auto clip = [](float v) { return std::clamp(v, -4.0f, 4.0f); };
  const std::size_t negatives = 5;
  for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
    const float rate = 1.0f - static_cast<float>(epoch) / static_cast<float>(epochs);
    for (std::size_t i = 0; i < n; ++i) {
      float* a = &xy[i * 2];
      for (const std::uint32_t j : knn[i]) {
        if (j == i) continue;
        float* b = &xy[j * 2];
        const float dx = a[0] - b[0], dy = a[1] - b[1];
        const float pull = -2.0f / (1.0f + dx * dx + dy * dy);
        const float mx = rate * clip(pull * dx), my = rate * clip(pull * dy);
        a[0] += mx, a[1] += my;
        b[0] -= mx, b[1] -= my;
        for (std::size_t s = 0; s < negatives; ++s) {
          const std::size_t k = rng() % n;
          if (k == i) continue;
          const float ex = a[0] - xy[k * 2], ey = a[1] - xy[k * 2 + 1];
          const float d2 = ex * ex + ey * ey;
          const float push = 2.0f / ((0.001f + d2) * (1.0f + d2));
          a[0] += rate * clip(push * ex);
          a[1] += rate * clip(push * ey);
        }
      }
    }
  }
  return xy;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: needle_export_words words.fvecs words.txt out.js [R] [L]\n");
    return 2;
  }
  const Dataset base = read_fvecs(argv[1]);
  std::vector<std::string> words;
  {
    std::ifstream in(argv[2]);
    for (std::string line; std::getline(in, line);) {
      if (!line.empty()) words.push_back(line);
    }
  }
  if (words.size() != base.n || base.n > 65535) {
    std::fprintf(stderr, "%zu words for %zu vectors (limit 65535)\n", words.size(), base.n);
    return 1;
  }

  VamanaParams params;
  params.R = argc > 4 ? static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10)) : 16;
  params.L = argc > 5 ? static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 64;
  VamanaIndex index(base, params);
  index.build();

  const std::size_t k = 15;
  const auto truth = exact_knn_batch(base, base, k + 1);

  std::printf("%zu words x %zu dims, R=%u, mean degree %.1f, %zu reachable\n", base.n, base.dim,
              params.R, index.mean_degree(), index.reachable_from_medoid());
  for (const std::size_t L : {16, 32, 64}) {  // every 40th word as a query
    std::vector<std::vector<std::uint32_t>> want, found;
    double comps = 0.0;
    for (std::size_t q = 0; q < base.n; q += 40) {
      SearchStats stats;
      std::vector<std::uint32_t> ids;
      for (const Neighbor& hit : index.search(base.row(q), 10, L, &stats)) ids.push_back(hit.id);
      comps += static_cast<double>(stats.dist_comps);
      found.push_back(ids);
      want.emplace_back(truth[q].begin(), truth[q].begin() + 10);
    }
    std::printf("L=%zu: recall@10 %.4f, %.0f of %zu words compared per query\n", L,
                recall_at_k(want, found, 10), comps / static_cast<double>(want.size()), base.n);
  }

  const std::vector<float> xy = layout(base, truth, 300, 7);
  float lo_x = xy[0], hi_x = xy[0], lo_y = xy[1], hi_y = xy[1];
  for (std::size_t i = 0; i < base.n; ++i) {
    lo_x = std::min(lo_x, xy[i * 2]), hi_x = std::max(hi_x, xy[i * 2]);
    lo_y = std::min(lo_y, xy[i * 2 + 1]), hi_y = std::max(hi_y, xy[i * 2 + 1]);
  }
  const float span = std::max(hi_x - lo_x, hi_y - lo_y);
  std::vector<std::uint16_t> map(base.n * 2);
  for (std::size_t i = 0; i < base.n; ++i) {  // one scale for both axes, centred
    const float x = (xy[i * 2] - (lo_x + hi_x) / 2) / span + 0.5f;
    const float y = (xy[i * 2 + 1] - (lo_y + hi_y) / 2) / span + 0.5f;
    map[i * 2] = static_cast<std::uint16_t>(std::lround(x * 65535.0f));
    map[i * 2 + 1] = static_cast<std::uint16_t>(std::lround(y * 65535.0f));
  }

  std::vector<std::int8_t> vectors(base.n * base.dim);
  for (std::size_t i = 0; i < vectors.size(); ++i) {
    vectors[i] = static_cast<std::int8_t>(std::lround(base.data[i]));
  }
  std::vector<std::uint8_t> degree(base.n);
  std::vector<std::uint16_t> edges;
  for (std::uint32_t v = 0; v < base.n; ++v) {
    const auto nbrs = index.neighbors(v);
    degree[v] = static_cast<std::uint8_t>(nbrs.size());
    for (const std::uint32_t nb : nbrs) edges.push_back(static_cast<std::uint16_t>(nb));
  }

  std::FILE* out = std::fopen(argv[3], "w");
  if (out == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }
  std::fprintf(out, "window.NEEDLE_WORDS={\"n\":%zu,\"dim\":%zu,\"R\":%u,\"medoid\":%u,\"words\":\"", base.n,
               base.dim, params.R, index.medoid());
  for (std::size_t i = 0; i < words.size(); ++i) {
    std::fprintf(out, i == 0 ? "%s" : " %s", words[i].c_str());
  }
  std::fprintf(out, "\",\"vec\":\"%s\",\"xy\":\"%s\",\"deg\":\"%s\",\"adj\":\"%s\"};\n",
               base64(vectors).c_str(), base64(map).c_str(), base64(degree).c_str(),
               base64(edges).c_str());
  std::fclose(out);
  std::printf("%zu edges -> %s\n", edges.size(), argv[3]);
  return 0;
}
