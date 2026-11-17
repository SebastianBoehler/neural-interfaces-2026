#include "ni26/initialization.h"
#include "ni26/math.h"
#include <ATen/CPUGeneratorImpl.h>
#include <cmath>
#include <torch/torch.h>

namespace ni26 {
Params initialize_raw_fbc(const Params &recipe, uint64_t seed) {
  torch::NoGradGuard guard;
  auto generator = at::detail::createCPUGenerator(seed);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  Params state;
  const std::vector<int64_t> lengths{199, 199, 133, 99, 81, 67, 57, 51, 45};
  for (size_t i = 0; i < lengths.size(); ++i) {
    auto name = "spectral_filtering.fir_list." + std::to_string(i);
    auto coefficient = recipe.get(name);
    require(coefficient.device().is_cpu() && coefficient.scalar_type() == torch::kFloat32 &&
                coefficient.sizes() == torch::IntArrayRef({lengths[i]}),
            "raw FBC initialization FIR contract differs");
    finite(coefficient, name);
    state.tensors[name] = coefficient.detach().clone().contiguous();
    auto denominator = torch::zeros({lengths[i]}, options);
    denominator[0] = 1;
    state.tensors["spectral_filtering.fir_a_list." + std::to_string(i)] = denominator;
  }
  auto uniform = [&](torch::IntArrayRef shape, double bound) {
    return torch::empty(shape, options).uniform_(-bound, bound, generator);
  };
  auto kaiming_bound = [](double fan_in) {
    auto a = std::sqrt(5.);
    auto gain = std::sqrt(2. / (1. + a * a));
    return std::sqrt(3.) * (gain / std::sqrt(fan_in));
  };
  // Original Conv2d draws default weights/bias, then redraws weights with
  // Xavier. The discarded first weight draw is required for RNG equivalence.
  auto spatial = uniform({288, 1, 41, 1}, kaiming_bound(41));
  auto spatial_bias = uniform({288}, 1. / std::sqrt(41.));
  auto xavier = std::sqrt(3.) * std::sqrt(2. / (41. + 288. * 41.));
  spatial.uniform_(-xavier, xavier, generator);
  state.tensors["spatial_conv.0.parametrizations.weight.original"] = spatial;
  state.tensors["spatial_conv.0.bias"] = spatial_bias;
  state.tensors["spatial_conv.1.weight"] = torch::ones({288}, options);
  state.tensors["spatial_conv.1.bias"] = torch::zeros({288}, options);
  state.tensors["spatial_conv.1.running_mean"] = torch::zeros({288}, options);
  state.tensors["spatial_conv.1.running_var"] = torch::ones({288}, options);
  state.tensors["final_layer.parametrizations.weight.original"] =
      uniform({3, 1152}, kaiming_bound(1152));
  state.tensors["final_layer.bias"] = uniform({3}, 1. / std::sqrt(1152.));
  state.tensors["native_format"] = torch::tensor(int64_t(1));
  return state;
}
} // namespace ni26
