#include "needle/dataset.h"

#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace needle {

Dataset read_fvecs(const std::string& path, std::size_t max_rows) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);

  Dataset ds;
  std::int32_t dim = 0;
  while (in.read(reinterpret_cast<char*>(&dim), sizeof dim)) {
    if (dim <= 0) throw std::runtime_error("bad dimension in " + path);
    if (ds.dim == 0) ds.dim = static_cast<std::size_t>(dim);
    if (ds.dim != static_cast<std::size_t>(dim)) throw std::runtime_error("mixed dimensions in " + path);

    const std::size_t old = ds.data.size();
    ds.data.resize(old + ds.dim);
    if (!in.read(reinterpret_cast<char*>(ds.data.data() + old),
                 static_cast<std::streamsize>(ds.dim * sizeof(float)))) {
      throw std::runtime_error("truncated vector in " + path);
    }
    ++ds.n;
    if (max_rows != 0 && ds.n == max_rows) break;
  }
  return ds;
}

void write_fvecs(const std::string& path, const Dataset& ds) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);

  const auto dim = static_cast<std::int32_t>(ds.dim);
  for (std::size_t i = 0; i < ds.n; ++i) {
    out.write(reinterpret_cast<const char*>(&dim), sizeof dim);
    out.write(reinterpret_cast<const char*>(ds.row(i)),
              static_cast<std::streamsize>(ds.dim * sizeof(float)));
  }
}

std::vector<std::vector<std::uint32_t>> read_ivecs(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);

  std::vector<std::vector<std::uint32_t>> rows;
  std::int32_t k = 0;
  while (in.read(reinterpret_cast<char*>(&k), sizeof k)) {
    if (k < 0) throw std::runtime_error("bad row length in " + path);
    std::vector<std::uint32_t> row(static_cast<std::size_t>(k));
    if (!in.read(reinterpret_cast<char*>(row.data()),
                 static_cast<std::streamsize>(row.size() * sizeof(std::uint32_t)))) {
      throw std::runtime_error("truncated row in " + path);
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

namespace {

/// Hand-rolled uniform and Box-Muller: std:: distributions differ between standard libraries.
struct Rng {
  std::mt19937_64 bits;

  explicit Rng(std::uint64_t seed) : bits(seed) {}

  double uniform() { return static_cast<double>(bits() >> 11) * (1.0 / 9007199254740992.0); }

  float gauss() {
    const double u1 = 1.0 - uniform();
    const double u2 = uniform();
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
  }
};

}  // namespace

Dataset make_clustered(std::size_t n, std::size_t dim, std::size_t clusters, std::uint64_t seed,
                       float spread) {
  if (clusters == 0) throw std::invalid_argument("clusters must be positive");

  Rng rng(seed);
  std::vector<float> centres(clusters * dim);
  for (float& c : centres) c = static_cast<float>(rng.uniform());

  Dataset ds;
  ds.n = n;
  ds.dim = dim;
  ds.data.resize(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const float* centre = centres.data() + (rng.bits() % clusters) * dim;
    float* out = ds.data.data() + i * dim;
    for (std::size_t d = 0; d < dim; ++d) out[d] = centre[d] + spread * rng.gauss();
  }
  return ds;
}

Dataset make_low_rank(std::size_t n, std::size_t dim, std::size_t latent, std::uint64_t seed,
                      float noise) {
  if (latent == 0) throw std::invalid_argument("latent must be positive");

  Rng rng(seed);
  std::vector<float> basis(dim * latent);
  for (float& b : basis) b = rng.gauss();

  Dataset ds;
  ds.n = n;
  ds.dim = dim;
  ds.data.resize(n * dim);
  std::vector<float> z(latent);
  for (std::size_t i = 0; i < n; ++i) {
    for (float& v : z) v = rng.gauss();
    float* out = ds.data.data() + i * dim;
    for (std::size_t d = 0; d < dim; ++d) {
      float acc = noise * rng.gauss();
      for (std::size_t l = 0; l < latent; ++l) acc += basis[d * latent + l] * z[l];
      out[d] = acc;
    }
  }
  return ds;
}

Dataset slice(const Dataset& ds, std::size_t begin, std::size_t end) {
  if (begin > end || end > ds.n) throw std::out_of_range("slice outside dataset");
  Dataset out;
  out.n = end - begin;
  out.dim = ds.dim;
  out.data.assign(ds.data.begin() + static_cast<std::ptrdiff_t>(begin * ds.dim),
                  ds.data.begin() + static_cast<std::ptrdiff_t>(end * ds.dim));
  return out;
}

}  // namespace needle
