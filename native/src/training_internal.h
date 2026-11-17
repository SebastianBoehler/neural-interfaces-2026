#pragma once
#include "ni26/preparation.h"
#include "ni26/sampling.h"
#include "ni26/training.h"
#include <torch/optim/adamw.h>
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
struct RawTrainingSource {
  RawNeuralSource source;
  Tensor labels;
  Params initial;
  std::vector<Batches> epochs;
};
RawTrainingSource prepare_raw_training_source(const Params &input);
struct SourceUpdate {
  Tensor logits, loss;
  double forward_seconds, backward_seconds, optimizer_seconds;
};
SourceUpdate raw_source_update(RawFBC &net, torch::optim::AdamW &optimizer,
                               const Tensor &x, const Tensor &y, int64_t step);
} // namespace ni26::neural
