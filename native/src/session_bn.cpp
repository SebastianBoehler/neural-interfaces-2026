#include "ni26/math.h"
#include "training_internal.h"
#include <torch/torch.h>

namespace ni26::neural {
SessionBN::SessionBN(const Params &state, torch::Device device) {
  auto get = [&](const char *name) {
    auto result = state.get(std::string("spatial_conv.1.") + name)
                      .to(device, torch::kFloat32)
                      .detach()
                      .clone();
    require(result.sizes() == torch::IntArrayRef({288}), "source SessionBN channel count differs");
    return result;
  };
  weight = get("weight").set_requires_grad(true);
  bias = get("bias").set_requires_grad(true);
  mean = get("running_mean");
  variance = get("running_var");
}
Tensor SessionBN::forward(const Tensor &x, bool training) {
  require(x.dim() == 4 && x.size(1) == 288, "source SessionBN shape differs");
  Tensor m, v;
  if (training) {
    require(x.size(0) % 32 == 0,
            "source SessionBN batch must contain complete32-row session chunks");
    auto groups = x.size(0) / 32;
    auto z = x.reshape({groups, 32, 288, x.size(2), x.size(3)});
    auto r1 = z.mean(std::vector<int64_t>{3, 4}), r2 = z.square().mean(std::vector<int64_t>{3, 4});
    auto gm = r1.mean(1), gv = (r2.mean(1) - gm.square()).clamp_min(0);
    m = gm.unsqueeze(1).expand({groups, 32, 288}).reshape({x.size(0), 288});
    v = gv.unsqueeze(1).expand({groups, 32, 288}).reshape({x.size(0), 288});
    {
      torch::NoGradGuard guard;
      mean.mul_(.9).add_(.1 * gm.mean(0));
      variance.mul_(.9).add_(.1 * gv.mean(0));
    }
  } else {
    auto r1 = x.mean(std::vector<int64_t>{2, 3}), r2 = x.square().mean(std::vector<int64_t>{2, 3});
    auto n = torch::arange(1, x.size(0) + 1, x.options()).unsqueeze(1);
    m = (20 * mean + r1.cumsum(0)) / (20 + n);
    v = ((20 * (variance + mean.square()) + r2.cumsum(0)) / (20 + n) - m.square()).clamp_min(0);
  }
  return (x - m.view({x.size(0), 288, 1, 1})) / torch::sqrt(v.view({x.size(0), 288, 1, 1}) + 1e-5) *
             weight.view({1, 288, 1, 1}) +
         bias.view({1, 288, 1, 1});
}
} // namespace ni26::neural
