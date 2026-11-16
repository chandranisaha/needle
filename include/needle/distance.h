#pragma once

#include <cstddef>

#if defined(__AVX2__) && defined(__FMA__) && !defined(NEEDLE_FORCE_SCALAR)
#define NEEDLE_HAVE_AVX2 1
#include <immintrin.h>
#endif

namespace needle {

/// Squared Euclidean distance, one float at a time. Kept as the reference and the fallback.
inline float l2_sq_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

#ifdef NEEDLE_HAVE_AVX2
/// Same distance, 16 floats per step. Two accumulators so consecutive FMAs do not wait on each
/// other; the compiler cannot do this itself because it changes the order of float additions.
inline float l2_sq_avx2(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
  }
  if (i + 8 <= dim) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    i += 8;
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  float sum = _mm_cvtss_f32(s);
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}
#endif

/// Squared Euclidean distance. Squaring keeps the ordering, so no search needs the square root.
inline float l2_sq(const float* a, const float* b, std::size_t dim) noexcept {
#ifdef NEEDLE_HAVE_AVX2
  return l2_sq_avx2(a, b, dim);
#else
  return l2_sq_scalar(a, b, dim);
#endif
}

/// Name of the kernel l2_sq compiles to, for benchmark output.
constexpr const char* distance_kernel() noexcept {
#ifdef NEEDLE_HAVE_AVX2
  return "avx2+fma";
#else
  return "scalar";
#endif
}

}  // namespace needle
