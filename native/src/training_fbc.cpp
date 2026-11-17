#include "ni26/math.h"
#include "training_internal.h"
#include <ATen/ops/fft_irfft.h>
#include <ATen/ops/fft_rfft.h>
#include <torch/torch.h>

namespace ni26::neural {
RawFBC::RawFBC(const Params &state, torch::Device device) : bn_(state, device) {
  auto get = [&](const char *key) {
    return state.get(key).to(device, torch::kFloat32).detach().clone().set_requires_grad(true);
  };
  spatial_weight_ = get("spatial_conv.0.parametrizations.weight.original");
  spatial_bias_ = get("spatial_conv.0.bias");
  head_weight_ = get("final_layer.parametrizations.weight.original");
  head_bias_ = get("final_layer.bias");
  require(spatial_weight_.sizes() == torch::IntArrayRef({288, 1, 41, 1}) &&
              head_weight_.sizes() == torch::IntArrayRef({3, 1152}),
          "native source FBC architecture differs");
  const std::vector<int64_t> sizes{199, 199, 133, 99, 81, 67, 57, 51, 45};
  for (int64_t i = 0; i < 9; ++i) {
    auto f = state.get("spectral_filtering.fir_list." + std::to_string(i))
                 .to(device, torch::kFloat32)
                 .detach()
                 .clone();
    require(f.sizes() == torch::IntArrayRef({sizes[i]}), "native source FIR shape differs");
    fir_.push_back(f);
    spectra_.push_back(at::fft_rfft(f, 480 + f.numel() - 1).view({1, 1, -1}).repeat({1, 41, 1}));
  }
}
Tensor RawFBC::forward(const Tensor &x, bool training) {
  require(x.dim() == 3 && x.size(1) == 41 && x.size(2) == 480 && x.scalar_type() == torch::kFloat32,
          "native source FBC input differs");
  std::map<int64_t, Tensor> cache;
  std::vector<Tensor> bands;
  for (size_t i = 0; i < fir_.size(); ++i) {
    auto n = 480 + fir_[i].numel() - 1;
    if (!cache.count(n))
      cache[n] = at::fft_rfft(x, n);
    auto start = (n - 480) / 2;
    bands.push_back(at::fft_irfft(cache.at(n) * spectra_[i], n).slice(-1, start, start + 480));
  }
  namespace F = torch::nn::functional;
  auto h = F::conv2d(torch::stack(bands, 1), spatial_weight_.renorm(2, 0, 2.),
                     F::Conv2dFuncOptions().bias(spatial_bias_).groups(9));
  h = torch::silu(bn_.forward(h, training)).view({x.size(0), 288, 4, 120});
  auto features = h.var(std::vector<int64_t>{3}, true, true).clamp(1e-6, 1e6).log().flatten(1);
  return F::linear(features, head_weight_.renorm(2, 0, .5), head_bias_);
}
std::vector<Tensor> RawFBC::parameters() const {
  return {spatial_weight_, spatial_bias_, bn_.weight, bn_.bias, head_weight_, head_bias_};
}
Params RawFBC::state() const {
  Params result;
  auto add = [&](const std::string &key, const Tensor &value) {
    result.tensors[key] = value.detach().cpu().clone();
  };
  add("spatial_conv.0.parametrizations.weight.original", spatial_weight_);
  add("spatial_conv.0.bias", spatial_bias_);
  add("spatial_conv.1.weight", bn_.weight);
  add("spatial_conv.1.bias", bn_.bias);
  add("spatial_conv.1.running_mean", bn_.mean);
  add("spatial_conv.1.running_var", bn_.variance);
  add("final_layer.parametrizations.weight.original", head_weight_);
  add("final_layer.bias", head_bias_);
  for (size_t i = 0; i < fir_.size(); ++i)
    add("spectral_filtering.fir_list." + std::to_string(i), fir_[i]);
  return result;
}
} // namespace ni26::neural
