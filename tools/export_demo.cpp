// Builds an index over 2-D points shaped like a spiral galaxy and writes it as a JS file that
// demo/index.html loads. The page only walks the graph; the graph itself comes from this engine.
//   needle_export_demo out.js [points] [R] [L]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "needle/dataset.h"
#include "needle/vamana.h"

using namespace needle;

namespace {

struct Rng {
  std::mt19937_64 bits;

  double uniform() { return static_cast<double>(bits() >> 11) * (1.0 / 9007199254740992.0); }

  double gauss() {
    const double u1 = 1.0 - uniform();
    const double u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
};

/// Three spiral arms, a dense core and a thin halo.
Dataset make_galaxy(std::size_t n, std::uint64_t seed) {
  Rng rng{std::mt19937_64(seed)};
  Dataset ds;
  ds.n = n;
  ds.dim = 2;
  ds.data.resize(n * 2);
  for (std::size_t i = 0; i < n; ++i) {
    const double pick = rng.uniform();
    double x = 0.0, y = 0.0;
    if (pick < 0.16) {
      x = 0.11 * rng.gauss();
      y = 0.11 * rng.gauss();
    } else if (pick < 0.24) {
      const double r = std::sqrt(rng.uniform()) * 1.15;
      const double a = 6.283185307179586 * rng.uniform();
      x = r * std::cos(a);
      y = r * std::sin(a);
    } else {
      const double t = std::pow(rng.uniform(), 0.75);
      const double arm = static_cast<double>(rng.bits() % 3) * 2.0943951023931953;
      const double angle = arm + 3.6 * t + 0.16 * rng.gauss();
      const double r = 0.08 + t + 0.035 * rng.gauss();
      x = r * std::cos(angle);
      y = r * std::sin(angle);
    }
    ds.data[i * 2] = static_cast<float>(x);
    ds.data[i * 2 + 1] = static_cast<float>(y);
  }
  return ds;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: needle_export_demo out.js [points] [R] [L]\n");
    return 2;
  }
  const std::size_t n = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 6000;
  VamanaParams params;
  params.R = argc > 3 ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 10;
  params.L = argc > 4 ? static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10)) : 40;

  const Dataset base = make_galaxy(n, 7);
  VamanaIndex index(base, params);
  index.build();

  std::FILE* out = std::fopen(argv[1], "w");
  if (out == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", argv[1]);
    return 1;
  }
  std::fprintf(out, "window.NEEDLE_GRAPH={\"n\":%zu,\"R\":%u,\"buildL\":%u,\"alpha\":%.2f,\"medoid\":%u,", n,
               params.R, params.L, static_cast<double>(params.alpha), index.medoid());

  std::fprintf(out, "\"points\":[");
  for (std::size_t i = 0; i < n * 2; ++i) {
    std::fprintf(out, i == 0 ? "%.4f" : ",%.4f", static_cast<double>(base.data[i]));
  }

  std::fprintf(out, "],\"offsets\":[0");
  std::size_t total = 0;
  for (std::uint32_t v = 0; v < n; ++v) {
    total += index.neighbors(v).size();
    std::fprintf(out, ",%zu", total);
  }

  std::fprintf(out, "],\"edges\":[");
  bool first = true;
  for (std::uint32_t v = 0; v < n; ++v) {
    for (const std::uint32_t nb : index.neighbors(v)) {
      std::fprintf(out, first ? "%u" : ",%u", nb);
      first = false;
    }
  }
  std::fprintf(out, "]};\n");
  std::fclose(out);

  std::printf("%zu points, %zu edges, mean degree %.1f, %zu reachable -> %s\n", n, total,
              index.mean_degree(), index.reachable_from_medoid(), argv[1]);
  return 0;
}
