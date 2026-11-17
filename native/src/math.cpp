#include "ni26/math.h"
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ni26 {
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void finite(const Tensor& value, const std::string& name) {
  require(value.defined() && torch::isfinite(value).all().item<bool>(), "nonfinite " + name);
}
Tensor spectral(const Tensor& x, SpectralFunction fn, double floor) {
  auto [e, v] = torch::linalg_eigh(x, "L");
  if (fn == SpectralFunction::Log) e = e.clamp_min(floor).log();
  else if (fn == SpectralFunction::Exp) e = e.exp();
  else if (fn == SpectralFunction::NegativeHalfExp) e = (-.5 * e).exp();
  else e = e.clamp_min(floor).rsqrt();
  return torch::matmul(v * e.unsqueeze(-2), v.transpose(-2, -1).conj());
}
Tensor common_average_reference_float32(const Tensor& raw) {
  require(raw.device().is_cpu() && raw.scalar_type()==torch::kFloat32 && raw.is_contiguous() &&
          raw.sizes()==torch::IntArrayRef({41,480}), "CAR requires contiguous CPU float32(41,480)");
  auto output=torch::empty_like(raw);
  const auto* input=raw.data_ptr<float>();auto* values=output.data_ptr<float>();
  for (int64_t time=0;time<480;++time) {
    float sum=0;
    for (int64_t channel=0;channel<41;++channel) sum+=input[channel*480+time];
    const float mean=sum/41.0f;
    for (int64_t channel=0;channel<41;++channel)
      values[channel*480+time]=input[channel*480+time]-mean;
  }
  return output;
}
Tensor normalize(const Tensor& p) { return p / p.sum(-1, true); }
Tensor posterior(const Tensor& z) { return torch::softmax(z, -1); }
Tensor lda_posterior(const Tensor& z, const Tensor& means, const Tensor& inverse) {
  auto weights = torch::matmul(inverse, means.t());
  return posterior(torch::matmul(z, weights) - .5 * (means * weights.t()).sum(1));
}
Tensor covariance(const Tensor& x, bool unbiased) {
  require(x.dim() == 2 && x.size(0) > int(unbiased), "invalid covariance rows");
  auto centered = x - x.mean(0);
  return torch::matmul(centered.t(), centered) / (x.size(0) - int(unbiased));
}
Tensor median(const Tensor& x, int64_t dim) {
  dim = dim < 0 ? x.dim() + dim : dim;
  require(dim >= 0 && dim < x.dim() && x.size(dim) > 0, "invalid median dimension");
  auto sorted = std::get<0>(x.sort(dim));
  auto n = x.size(dim);
  return (sorted.select(dim, (n - 1) / 2) + sorted.select(dim, n / 2)) * .5;
}
Tensor upper_features(const Tensor& x) {
  auto n = x.size(-1);
  auto ids = torch::triu_indices(n, n, 0, torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
  auto flat = ids[0] * n + ids[1];
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(x.device());
  auto weights = torch::where(ids[0] == ids[1], torch::ones({flat.numel()},options),
                              torch::full({flat.numel()},std::sqrt(2.),options));
  auto real = torch::real(x.flatten(-2).index_select(-1, flat)) * weights;
  if (!x.is_complex()) return real;
  auto off = torch::triu_indices(n, n, 1, ids.options());
  auto imag = torch::imag(x.flatten(-2).index_select(-1, off[0] * n + off[1])) * std::sqrt(2.);
  return torch::cat({real, imag}, -1);
}
double ramp(double score, double threshold, double width) {
  require(width > 0, "nonpositive ramp width");
  return std::clamp((score - threshold) / width, 0., 1.);
}
}
