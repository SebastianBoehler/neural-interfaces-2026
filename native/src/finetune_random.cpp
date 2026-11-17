#include "networks_internal.h"
#include <ATen/CPUGeneratorImpl.h>
#include <torch/torch.h>
#ifdef NI26_WITH_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

namespace ni26::neural {
at::Generator generator(torch::Device device, int64_t seed) {
  if (device.is_cpu())
    return at::detail::createCPUGenerator(seed);
#ifdef NI26_WITH_CUDA
  if (device.is_cuda()) {
    auto result = at::cuda::detail::createCUDAGenerator(device.index());
    result.set_current_seed(seed);
    return result;
  }
#endif
  throw std::runtime_error("fine-tuning generator requires a compiled native implementation for " +
                           device.str());
}
Tensor strong(const Tensor &x, at::Generator &gen) {
  auto shift = at::randint(-15, 16, {1}, gen, x.options().dtype(torch::kInt64)).item<int64_t>();
  auto result = torch::roll(x, {shift}, {-1});
  result = result + .2 * result.std(std::vector<int64_t>{-1}, true, true) *
                        at::randn(result.sizes(), gen, result.options());
  return result * (at::rand({result.size(0), result.size(1), 1}, gen, result.options()) > .1)
                      .to(torch::kFloat32);
}
} // namespace ni26::neural
