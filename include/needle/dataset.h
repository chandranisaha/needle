#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace needle {

/// Row-major matrix of n vectors, each dim floats.
struct Dataset {
  std::size_t n = 0;
  std::size_t dim = 0;
  std::vector<float> data;

  const float* row(std::size_t i) const noexcept { return data.data() + i * dim; }
};

/// Reads a .fvecs file (per vector: int32 dim, then dim floats). max_rows == 0 reads everything.
Dataset read_fvecs(const std::string& path, std::size_t max_rows = 0);

/// Writes a dataset in .fvecs format.
void write_fvecs(const std::string& path, const Dataset& ds);

/// Reads a .ivecs file (per row: int32 k, then k int32 ids), the usual ground-truth format.
std::vector<std::vector<std::uint32_t>> read_ivecs(const std::string& path);

/// Gaussian blobs around uniformly placed centres; same seed gives the same data.
Dataset make_clustered(std::size_t n, std::size_t dim, std::size_t clusters,
                       std::uint64_t seed, float spread = 0.08f);

/// Points on a random latent-dimensional subspace of R^dim plus small noise, like real embeddings.
Dataset make_low_rank(std::size_t n, std::size_t dim, std::size_t latent, std::uint64_t seed,
                      float noise = 0.02f);

/// Copies rows [begin, end) into a new dataset.
Dataset slice(const Dataset& ds, std::size_t begin, std::size_t end);

}  // namespace needle
