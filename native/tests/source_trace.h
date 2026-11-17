#pragma once
#include "ni26/artifact.h"
#include "ni26/math.h"
#include <torch/torch.h>

// Standalone one-batch diagnostic only; never linked into the production package.
extern ni26::Params source_observations;
inline void source_observe(const std::string &name, const ni26::Tensor &value) {
  ni26::finite(value, "source trace " + name);
  source_observations.tensors[name] = value.detach().cpu().clone();
  source_observations.tensors[name + "_stride"] =
      torch::tensor(std::vector<int64_t>(value.strides().begin(), value.strides().end()), torch::kInt64);
  if (value.requires_grad())
    value.register_hook([name](const ni26::Tensor &gradient) {
      ni26::finite(gradient, "source trace gradient " + name);
      source_observations.tensors[name + "_grad"] = gradient.detach().cpu().clone();
      source_observations.tensors[name + "_grad_stride"] = torch::tensor(
          std::vector<int64_t>(gradient.strides().begin(), gradient.strides().end()), torch::kInt64);
      return gradient;
    });
}
