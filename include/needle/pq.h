#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "needle/dataset.h"

namespace needle {

struct PqParams {
  std::size_t m = 32;                // parts per vector, which is also bytes per code
  std::size_t train_size = 100'000;  // vectors sampled for k-means
  std::size_t iters = 12;            // k-means iterations per part
  std::uint64_t seed = 42;
  unsigned threads = 0;              // 0 = all cores; the result does not depend on it
};

/// Product quantizer (Jégou, Douze, Schmid, 2011). Each vector is cut into m equal parts and each
/// part is replaced by the id of the nearest of 256 centroids learned for that part, so a vector
/// of any dimension is stored in m bytes.
class ProductQuantizer {
 public:
  static constexpr std::size_t kCentroids = 256;

  /// Learns the centroids by k-means on a sample of data. data.dim must be a multiple of params.m.
  void train(const Dataset& data, const PqParams& params);

  /// The m-byte code of every row, concatenated.
  std::vector<std::uint8_t> encode(const Dataset& data, unsigned threads = 0) const;
  void encode_one(const float* v, std::uint8_t* code) const;
  /// Writes the dim floats a code stands for.
  void decode(const std::uint8_t* code, float* out) const;

  /// Fills table (table_size() floats) with the squared distance from each part of query to every
  /// centroid of that part. Built once per query.
  void distance_table(const float* query, float* table) const;

  /// Squared distance from the table's query to a coded vector: m lookups, no arithmetic on the
  /// vector itself. Equals the exact distance from the query to decode(code).
  float distance(const float* table, const std::uint8_t* code) const noexcept {
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
    std::size_t s = 0;
    for (; s + 4 <= m_; s += 4) {
      a += table[s * kCentroids + code[s]];
      b += table[(s + 1) * kCentroids + code[s + 1]];
      c += table[(s + 2) * kCentroids + code[s + 2]];
      d += table[(s + 3) * kCentroids + code[s + 3]];
    }
    for (; s < m_; ++s) a += table[s * kCentroids + code[s]];
    return (a + b) + (c + d);
  }

  std::size_t m() const noexcept { return m_; }
  std::size_t dim() const noexcept { return m_ * sub_; }
  std::size_t table_size() const noexcept { return m_ * kCentroids; }
  bool trained() const noexcept { return m_ != 0; }

 private:
  const float* centroid(std::size_t part, std::size_t id) const noexcept {
    return centroids_.data() + (part * kCentroids + id) * sub_;
  }

  std::size_t m_ = 0;
  std::size_t sub_ = 0;           // dimensions per part
  std::vector<float> centroids_;  // m * 256 * sub
};

}  // namespace needle
