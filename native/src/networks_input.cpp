#include "networks_internal.h"
#include <array>
#include <torch/torch.h>

namespace ni26::neural {
namespace {
// Preserve NumPy2.5.4's contiguous-axis float32 reduction order.
// https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/umath/loops_utils.h.src
float pairwise_sum(const float *values, int64_t count) {
  if (count < 8) {
    float sum = -0.0f;
    for (int64_t i = 0; i < count; ++i)
      sum += values[i];
    return sum;
  }
  if (count <= 128) {
    std::array<float, 8> lanes;
    for (int j = 0; j < 8; ++j)
      lanes[j] = values[j];
    int64_t i = 8;
    for (; i < count - count % 8; i += 8)
      for (int j = 0; j < 8; ++j)
        lanes[j] += values[i + j];
    float sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (; i < count; ++i)
      sum += values[i];
    return sum;
  }
  auto half = (count / 2) / 8 * 8;
  return pairwise_sum(values, half) + pairwise_sum(values + half, count - half);
}
std::string bytes(const Tensor &value) {
  auto x = value.detach().cpu().contiguous();
  return std::string(static_cast<const char *>(x.const_data_ptr()), x.nbytes());
}
} // namespace
Tensor temporal_row_mean_float32(const Tensor &x) {
  require(x.device().is_cpu() && x.scalar_type() == torch::kFloat32 &&
              x.sizes() == torch::IntArrayRef({41, 480}),
          "temporal row mean requires CPU float32(41,480)");
  auto contiguous = x.contiguous();
  auto result = torch::empty({41, 1}, x.options());
  const auto *values = contiguous.const_data_ptr<float>();
  auto *means = result.data_ptr<float>();
  for (int channel = 0; channel < 41; ++channel)
    means[channel] = (0.0f + pairwise_sum(values + channel * 480, 480)) / 480.0f;
  return result;
}
std::string input_key(const Params &p) {
  std::string result;
  for (const auto *key : {"align", "k", "G", "scale", "shrink"})
    result += bytes(p.get(key));
  return result;
}
InputStream::InputStream(const Params &p)
    : align_(p.get("align").item<bool>()), scale_(p.number("scale")), k_(p.number("k")),
      shrink_(p.number("shrink")), prior_(p.get("G").to(torch::kCPU, torch::kFloat64).clone()) {
  require(scale_ > 0 && k_ >= 0 && shrink_ >= 0 && shrink_ <= 1, "invalid neural input parameters");
  require(prior_.sizes() == torch::IntArrayRef({41, 41}), "neural input prior is not41×41");
  reset();
}
InputStream::InputStream(const InputStream &x)
    : align_(x.align_), scale_(x.scale_), k_(x.k_), shrink_(x.shrink_), prior_(x.prior_),
      sum_(x.sum_.clone()), count_(x.count_) {}
InputStream &InputStream::operator=(const InputStream &x) {
  if (this != &x) {
    align_ = x.align_;
    scale_ = x.scale_;
    k_ = x.k_;
    shrink_ = x.shrink_;
    prior_ = x.prior_;
    sum_ = x.sum_.clone();
    count_ = x.count_;
  }
  return *this;
}
void InputStream::reset() {
  count_ = 0;
  sum_ = torch::zeros_like(prior_);
}
Tensor InputStream::step(const Tensor &input) {
  require(input.device().is_cpu() && input.scalar_type() == torch::kFloat32 &&
              input.sizes() == torch::IntArrayRef({41, 480}),
          "neural input must be CPU float32(41,480)");
  finite(input, "network window");
  auto x = input.to(torch::kFloat64);
  if (!align_)
    return (x / scale_).to(torch::kFloat32);
  // The reference centers its float32 window before float64 covariance products.
  auto centered = (input - temporal_row_mean_float32(input)).to(torch::kFloat64);
  auto c = torch::matmul(centered, centered.t()) / 480.;
  c = (1 - shrink_) * c + shrink_ * c.trace() / 41 * torch::eye(41, c.options());
  sum_ = sum_ + c;
  ++count_;
  auto p = spectral((k_ * prior_ + sum_) / (k_ + count_), SpectralFunction::InverseRoot);
  return torch::matmul(p, x).to(torch::kFloat32);
}
void zero_stream(torch::jit::Module &module) {
  for (auto item : module.named_buffers(true)) {
    auto split = item.name.find_last_of('.');
    auto name = item.name.substr(split == std::string::npos ? 0 : split + 1);
    if (name == "s1" || name == "s2" || name == "n") {
      item.value.detach_();
      item.value.zero_();
    }
  }
}
std::vector<Tensor> trainable(torch::jit::Module &module) {
  std::vector<Tensor> result;
  for (auto p : module.parameters())
    if (p.requires_grad())
      result.push_back(p);
  for (auto item : module.named_buffers(true)) {
    auto split = item.name.find_last_of('.');
    auto name = item.name.substr(split == std::string::npos ? 0 : split + 1);
    if (name == "w" || name == "b")
      result.push_back(item.value);
  }
  require(!result.empty(), "fine-tuning module exposes no trainable tensors");
  return result;
}
} // namespace ni26::neural
