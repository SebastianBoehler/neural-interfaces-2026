#pragma once
#include "ni26/math.h"
#include <cmath>
#include <torch/torch.h>

namespace ni26::neural {
// Original torch.optim.Adam defaults, non-capturable single-tensor operation
// order. LibTorch Adam uses mul/add for m; Python uses lerp, which rounds
// differently.
class FineTuneAdam {
public:
  FineTuneAdam(std::vector<Tensor> parameters, double lr)
      : parameters_(std::move(parameters)), lr_(lr),
        steps_(parameters_.size(), 0) {
    require(lr > 0, "invalid fine-tuning Adam learning rate");
    for (const auto &p : parameters_) {
      require(!p.is_complex(), "fine-tuning Adam requires real parameters");
      first_.push_back(torch::zeros_like(p, torch::MemoryFormat::Preserve));
      second_.push_back(torch::zeros_like(p, torch::MemoryFormat::Preserve));
    }
  }
  void zero_grad() {
    for (auto &p : parameters_)
      p.mutable_grad() = Tensor();
  }
  void step() {
    torch::NoGradGuard guard;
    for (size_t i = 0; i < parameters_.size(); ++i) {
      auto &p = parameters_[i];
      const auto &g = p.grad();
      if (!g.defined())
        continue;
      require(!g.is_sparse(), "fine-tuning Adam requires dense gradients");
      double step = ++steps_[i];
      first_[i].lerp_(g, 1 - beta1_);
      second_[i].mul_(beta2_).addcmul_(g, g, 1 - beta2_);
      double correction1 = 1 - std::pow(beta1_, step);
      double correction2 = 1 - std::pow(beta2_, step);
      double step_size = lr_ / correction1;
      double correction2_sqrt = std::pow(correction2, 0.5);
      auto denominator = (second_[i].sqrt() / correction2_sqrt).add_(epsilon_);
      p.addcdiv_(first_[i], denominator, -step_size);
    }
  }
  const Tensor &first(size_t i) const { return first_.at(i); }
  const Tensor &second(size_t i) const { return second_.at(i); }
  int64_t steps(size_t i) const { return steps_.at(i); }

private:
  static constexpr double beta1_ = 0.9, beta2_ = 0.999, epsilon_ = 1e-8;
  std::vector<Tensor> parameters_, first_, second_;
  double lr_;
  std::vector<int64_t> steps_;
};
} // namespace ni26::neural
