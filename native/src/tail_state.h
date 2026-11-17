#pragma once
#include "ni26/tail.h"
#include <vector>

namespace ni26 {
struct GaussianState {
  Tensor mean, sd, centers, inverse;
  double exponent = 0.;
  Tensor apply(const Tensor& features, const Tensor& probabilities) const;
};
struct EMStage {
  Params z;
  std::vector<Tensor> features, probabilities, events;
  std::optional<GaussianState> state;
  double margin_sum = 0., gate = 1.;
  explicit EMStage(const Params& params) : z(params) {}
  Tensor view(const Tensor& x, const Tensor& extra);
  Tensor step(const Tensor& x, const Tensor& p, const Tensor& extra);
  Tensor reapply(int64_t count) const;
};
struct TSEMStage {
  Params z;
  std::vector<Tensor> features, confidence, labels;
  std::optional<GaussianState> state;
  double prior_weight = 0.;
  Tensor prior;
  explicit TSEMStage(const Params& params) : z(params) {}
  Tensor step(const Tensor& u, const Tensor& p, double gate);
  Tensor reapply(int64_t count, const Tensor& p) const;
};
double early_boost(const Params& z, const std::string& field, int64_t count, double gate);
}
