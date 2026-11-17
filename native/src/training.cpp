#include "ni26/math.h"
#include "training_internal.h"
#include <cmath>
#include <torch/torch.h>

namespace ni26::neural {
std::pair<double, double> one_cycle(int64_t step, int64_t total) {
  require(total > 1 && step >= 0 && step < total,
          "invalid native OneCycle step");
  double first = .3 * total - 1, pct, start_lr, end_lr, start_beta, end_beta;
  if (step <= first) {
    pct = step / first;
    start_lr = .001 / 25;
    end_lr = .001;
    start_beta = .95;
    end_beta = .85;
  } else {
    pct = (step - first) / (total - 1 - first);
    start_lr = .001;
    end_lr = .001 / 25 / 10000;
    start_beta = .85;
    end_beta = .95;
  }
  auto cosine = [&](double a, double b) {
    return b + (a - b) / 2 * (std::cos(M_PI * pct) + 1);
  };
  return {cosine(start_lr, end_lr), cosine(start_beta, end_beta)};
}
} // namespace ni26::neural
namespace ni26 {
Params train_raw_fbc(const Params &input, torch::Device device) {
  torch::AutoGradMode enable_grad(true);
  auto source = neural::prepare_raw_training_source(input);
  neural::RawFBC net(source.initial, device);
  auto parameters = net.parameters();
  torch::optim::AdamW optimizer(
      parameters, torch::optim::AdamWOptions(.001).weight_decay(.001));
  std::vector<Tensor> losses;
  double gradient_max = 0;
  auto body_initial = parameters.front().detach().clone();
  auto head_initial = parameters[4].detach().clone();
  int64_t step = 0;
  for (const auto &epoch : source.epochs)
    for (const auto &batch : epoch) {
      auto ids = torch::tensor(batch, torch::kInt64);
      auto update = neural::raw_source_update(
          net, optimizer, source.source.x.index_select(0, ids).to(device),
          source.labels.index_select(0, ids).to(device), step++);
      gradient_max = std::max(
          gradient_max, parameters.front().grad().abs().max().item<double>());
      losses.push_back(update.loss.cpu());
    }
  require(step == 1240, "native source40epoch step count differs");
  double body_change =
      (parameters.front().detach() - body_initial).abs().max().item<double>();
  double head_change =
      (parameters[4].detach() - head_initial).abs().max().item<double>();
  require(gradient_max > 0 && body_change > 0 && head_change > 0,
          "native source optimizer did not update body/head");
  auto result = net.state();
  result.tensors["native_format"] = torch::tensor(1, torch::kInt64);
  result.tensors["G"] = source.source.covariance_prior;
  result.tensors["scale"] = torch::tensor(source.source.scale, torch::kFloat64);
  result.tensors["losses"] = torch::stack(losses);
  result.tensors["steps"] = torch::tensor(step, torch::kInt64);
  result.tensors["body_gradient_max"] =
      torch::tensor(gradient_max, torch::kFloat64);
  result.tensors["body_change_max"] =
      torch::tensor(body_change, torch::kFloat64);
  result.tensors["head_change_max"] =
      torch::tensor(head_change, torch::kFloat64);
  return result;
}
} // namespace ni26
