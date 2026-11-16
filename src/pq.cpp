#include "needle/pq.h"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

#include "needle/distance.h"
#include "parallel.h"

namespace needle {
namespace {

std::size_t nearest(const float* v, const float* centroids, std::size_t k, std::size_t dim) noexcept {
  std::size_t best = 0;
  float best_dist = std::numeric_limits<float>::max();
  for (std::size_t c = 0; c < k; ++c) {
    const float d = l2_sq(v, centroids + c * dim, dim);
    if (d < best_dist) {
      best_dist = d;
      best = c;
    }
  }
  return best;
}

/// Lloyd's k-means over n points of the given dimension. Starts from k distinct random points;
/// a centroid that ends an iteration with no points is moved onto a random point.
void kmeans(const std::vector<float>& points, std::size_t n, std::size_t dim, std::size_t k,
            std::size_t iters, std::uint64_t seed, float* centroids) {
  std::mt19937_64 rng(seed);
  std::vector<std::size_t> pick(n);
  for (std::size_t i = 0; i < n; ++i) pick[i] = i;
  for (std::size_t c = 0; c < k; ++c) {
    if (c < n) std::swap(pick[c], pick[c + rng() % (n - c)]);
    const float* from = points.data() + pick[c % n] * dim;
    std::copy(from, from + dim, centroids + c * dim);
  }

  std::vector<double> sum(k * dim);
  std::vector<std::size_t> count(k);
  for (std::size_t iter = 0; iter < iters; ++iter) {
    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(count.begin(), count.end(), 0);
    for (std::size_t i = 0; i < n; ++i) {
      const float* p = points.data() + i * dim;
      const std::size_t c = nearest(p, centroids, k, dim);
      ++count[c];
      for (std::size_t d = 0; d < dim; ++d) sum[c * dim + d] += p[d];
    }
    for (std::size_t c = 0; c < k; ++c) {
      float* to = centroids + c * dim;
      if (count[c] == 0) {
        const float* from = points.data() + (rng() % n) * dim;
        std::copy(from, from + dim, to);
        continue;
      }
      for (std::size_t d = 0; d < dim; ++d) {
        to[d] = static_cast<float>(sum[c * dim + d] / static_cast<double>(count[c]));
      }
    }
  }
}

}  // namespace

void ProductQuantizer::train(const Dataset& data, const PqParams& params) {
  if (data.n == 0) throw std::invalid_argument("cannot train a quantizer on an empty dataset");
  if (params.m == 0 || data.dim % params.m != 0) {
    throw std::invalid_argument("the dimension must be a multiple of the number of parts");
  }
  m_ = params.m;
  sub_ = data.dim / params.m;
  centroids_.assign(m_ * kCentroids * sub_, 0.0f);

  // One sample of rows, shared by every part.
  std::vector<std::size_t> rows(data.n);
  for (std::size_t i = 0; i < data.n; ++i) rows[i] = i;
  const std::size_t sample = std::min(data.n, std::max<std::size_t>(params.train_size, 1));
  std::mt19937_64 rng(params.seed);
  for (std::size_t i = 0; i < sample; ++i) std::swap(rows[i], rows[i + rng() % (data.n - i)]);

  // Parts are independent, and each has its own seed, so the thread count does not change the result.
  parallel_for(m_, params.threads, 1, [&](std::size_t part) {
    std::vector<float> points(sample * sub_);
    for (std::size_t i = 0; i < sample; ++i) {
      const float* from = data.row(rows[i]) + part * sub_;
      std::copy(from, from + sub_, points.data() + i * sub_);
    }
    kmeans(points, sample, sub_, kCentroids, params.iters, params.seed + 1 + part,
           centroids_.data() + part * kCentroids * sub_);
  });
}

void ProductQuantizer::encode_one(const float* v, std::uint8_t* code) const {
  for (std::size_t part = 0; part < m_; ++part) {
    code[part] = static_cast<std::uint8_t>(nearest(v + part * sub_, centroid(part, 0), kCentroids, sub_));
  }
}

std::vector<std::uint8_t> ProductQuantizer::encode(const Dataset& data, unsigned threads) const {
  if (!trained()) throw std::logic_error("encode before train");
  if (data.dim != dim()) throw std::invalid_argument("the quantizer was trained on another dimension");
  std::vector<std::uint8_t> codes(data.n * m_);
  parallel_for(data.n, threads, 256, [&](std::size_t i) { encode_one(data.row(i), codes.data() + i * m_); });
  return codes;
}

void ProductQuantizer::decode(const std::uint8_t* code, float* out) const {
  for (std::size_t part = 0; part < m_; ++part) {
    const float* from = centroid(part, code[part]);
    std::copy(from, from + sub_, out + part * sub_);
  }
}

void ProductQuantizer::distance_table(const float* query, float* table) const {
  for (std::size_t part = 0; part < m_; ++part) {
    const float* q = query + part * sub_;
    for (std::size_t c = 0; c < kCentroids; ++c) {
      table[part * kCentroids + c] = l2_sq(q, centroid(part, c), sub_);
    }
  }
}

}  // namespace needle
