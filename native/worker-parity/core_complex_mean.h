#pragma once
#include "ni26/math.h"
#include <array>

namespace ni26 {
namespace {

using ComplexFloat = c10::complex<float>;

ComplexFloat core_pairwise_sum(const ComplexFloat* values, int64_t length) {
  if (length < 4) {
    ComplexFloat sum(-0.f, -0.f);
    for (int64_t i = 0; i < length; ++i) sum += values[i];
    return sum;
  }
  if (length <= 64) {
    std::array<ComplexFloat, 4> lanes{values[0], values[1], values[2], values[3]};
    const auto block_end = length - length % 4;
    for (int64_t i = 4; i < block_end; i += 4) {
      for (int64_t lane = 0; lane < 4; ++lane) lanes[lane] += values[i + lane];
    }
    auto sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (int64_t i = block_end; i < length; ++i) sum += values[i];
    return sum;
  }
  const auto half = (length / 8) * 4;
  return core_pairwise_sum(values, half) + core_pairwise_sum(values + half, length - half);
}

Tensor core_complex_mean(const Tensor& segment) {
  require(segment.dim() == 3 && segment.device().is_cpu() && segment.scalar_type() == torch::kComplexFloat &&
          segment.stride(2) == 1 && segment.size(2) > 0, "core mean expects CPU complex64 time rows");
  const auto bands = segment.size(0), channels = segment.size(1), length = segment.size(2);
  auto output = torch::empty({bands, channels, 1}, segment.options());
  const auto* values = segment.data_ptr<ComplexFloat>();
  auto* means = output.data_ptr<ComplexFloat>();
  for (int64_t band = 0; band < bands; ++band) {
    for (int64_t channel = 0; channel < channels; ++channel) {
      const auto* row = values + band * segment.stride(0) + channel * segment.stride(1);
      // NumPy sums complex64 in float32 lanes and divides in complex128.
      const auto sum = ComplexFloat(0.f, 0.f) + core_pairwise_sum(row, length);
      *means++ = ComplexFloat(float(double(sum.real()) / length), float(double(sum.imag()) / length));
    }
  }
  return output;
}

}
}
