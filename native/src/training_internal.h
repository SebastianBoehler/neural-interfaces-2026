#pragma once
#include "ni26/training.h"
#include <vector>

namespace ni26::neural {
// Exact SessionNet group32, non-causal training; k20 causal evaluation.
class SessionBN {
public:
  explicit SessionBN(const Params &initial, torch::Device device);
  Tensor forward(const Tensor &x, bool training);
  Tensor weight, bias, mean, variance;
};
class RawFBC {
public:
  RawFBC(const Params &initial, torch::Device device);
  Tensor forward(const Tensor &x, bool training);
  std::vector<Tensor> parameters() const;
  Params state() const;

private:
  Tensor spatial_weight_, spatial_bias_, head_weight_, head_bias_;
  std::vector<Tensor> fir_, spectra_;
  SessionBN bn_;
};
std::pair<double, double> one_cycle(int64_t step, int64_t total);
} // namespace ni26::neural
