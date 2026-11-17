#include "ni26/math.h"
#include <torch/torch.h>
#include <algorithm>
#include <vector>

namespace ni26 {
SosFilter::SosFilter(const Tensor& sections) {
  coefficients_ = sections.to(torch::kCPU, torch::kFloat64).contiguous().clone();
  require(coefficients_.dim() == 2 && coefficients_.size(1) == 6, "invalid SOS coefficients");
  finite(coefficients_, "SOS coefficients");
  initial_ = torch::empty({coefficients_.size(0), 2}, torch::kFloat64);
  const auto* a = coefficients_.data_ptr<double>();
  auto* zi = initial_.data_ptr<double>();
  int zero_b = 0, zero_a = 0;
  double scale = 1.;
  for (int64_t s = 0; s < coefficients_.size(0); ++s) {
    const auto* c = a + 6 * s;
    require(c[3] == 1., "SOS denominator must be normalized");
    const double denominator = 1. + c[4] + c[5];
    require(denominator != 0., "singular SOS initial state");
    const double gain = (c[0] + c[1] + c[2]) / denominator;
    zi[2*s] = scale * (gain - c[0]);
    zi[2*s+1] = scale * (c[2] - c[5] * gain);
    scale *= gain;
    zero_b += c[2] == 0.; zero_a += c[5] == 0.;
  }
  pad_ = 3 * (2 * coefficients_.size(0) + 1 - std::min(zero_a, zero_b));
}
Tensor SosFilter::operator()(const Tensor& input) const {
  require(input.dim() >= 1 && input.size(-1) > pad_, "window shorter than SOS padding");
  auto x = input.to(torch::kCPU, torch::kFloat64).contiguous();
  auto output = torch::empty_like(x);
  const auto n = x.size(-1), rows = x.numel() / n;
  const auto* coefficients = coefficients_.data_ptr<double>();
  const auto* initial = initial_.data_ptr<double>();
  const auto* raw = x.data_ptr<double>(); auto* result = output.data_ptr<double>();
  std::vector<double> padded(n + 2 * pad_);
  auto forward = [&] {
    const double first = padded.front();
    for (int64_t s = 0; s < coefficients_.size(0); ++s) {
      const auto* c = coefficients + 6*s;
      double z0 = initial[2*s] * first, z1 = initial[2*s+1] * first;
      for (double& v : padded) {
        double y = c[0] * v + z0;
        z0 = c[1] * v - c[4] * y + z1; z1 = c[2] * v - c[5] * y; v = y;
      }
    }
  };
  for (int64_t row = 0; row < rows; ++row) {
    const auto* source = raw + row*n;
    for (int64_t j = 0; j < pad_; ++j) padded[j] = 2*source[0] - source[pad_-j];
    std::copy(source, source+n, padded.begin()+pad_);
    for (int64_t j = 0; j < pad_; ++j) padded[pad_+n+j] = 2*source[n-1] - source[n-2-j];
    forward(); std::reverse(padded.begin(), padded.end());
    forward(); std::reverse(padded.begin(), padded.end());
    std::copy(padded.begin()+pad_, padded.begin()+pad_+n, result+row*n);
  }
  return output;
}
std::vector<SosFilter> filters(const Params& z, const std::string& key) {
  auto bank = z.get(key);
  require(bank.dim() == 3, "expected SOS bank: " + key);
  std::vector<SosFilter> result;
  for (const auto& band : bank.unbind(0)) result.emplace_back(band);
  return result;
}
}
