#pragma once
#include "ni26/math.h"
#include <cmath>
#include <torch/torch.h>

namespace ni26::neural {
// Match Python Adam and AdamW dispatch for CPU and CUDA tensors.
class Adam {
public:
  Adam(std::vector<Tensor> parameters, double lr, double weight_decay = 0.)
      : parameters_(std::move(parameters)), lr_(lr),
        weight_decay_(weight_decay),
        steps_(parameters_.size(), 0) {
    require(lr > 0 && weight_decay >= 0, "invalid Adam learning rate or decay");
    for (const auto &p : parameters_) {
      require(!p.is_complex(), "Adam requires real parameters");
      first_.push_back(torch::zeros_like(p, torch::MemoryFormat::Preserve));
      second_.push_back(torch::zeros_like(p, torch::MemoryFormat::Preserve));
    }
  }
  void zero_grad() {
    for (auto &p : parameters_)
      p.mutable_grad() = Tensor();
  }
  void step() { step(lr_, .9); }
  void step(double lr, double beta1) {
    torch::NoGradGuard guard;
    if (!parameters_.empty() && parameters_.front().is_cuda()) {
      step_cuda(lr, beta1);
      return;
    }
    for (size_t i = 0; i < parameters_.size(); ++i) {
      auto &p = parameters_[i];
      const auto &g = p.grad();
      if (!g.defined())
        continue;
      require(!g.is_sparse(), "Adam requires dense gradients");
      double step = ++steps_[i];
      if (weight_decay_ != 0.)
        p.mul_(1 - lr * weight_decay_);
      first_[i].lerp_(g, 1 - beta1);
      second_[i].mul_(beta2_).addcmul_(g, g, 1 - beta2_);
      double correction1 = 1 - std::pow(beta1, step);
      double correction2 = 1 - std::pow(beta2_, step);
      double step_size = lr / correction1;
      double correction2_sqrt = std::pow(correction2, 0.5);
      auto denominator = (second_[i].sqrt() / correction2_sqrt).add_(epsilon_);
      p.addcdiv_(first_[i], denominator, -step_size);
    }
  }
  const Tensor &first(size_t i) const { return first_.at(i); }
  const Tensor &second(size_t i) const { return second_.at(i); }
  int64_t steps(size_t i) const { return steps_.at(i); }

private:
  void step_cuda(double lr, double beta1) {
    std::vector<Tensor> parameters, gradients, first, second;
    std::vector<at::Scalar> sizes, corrections;
    for (size_t i = 0; i < parameters_.size(); ++i) {
      const auto &p = parameters_[i], &g = p.grad();
      if (!g.defined())
        continue;
      require(!g.is_sparse(), "Adam requires dense gradients");
      parameters.push_back(p);
      gradients.push_back(g);
      first.push_back(first_[i]);
      second.push_back(second_[i]);
      double step = ++steps_[i];
      sizes.emplace_back(-lr / (1 - std::pow(beta1, step)));
      corrections.emplace_back(std::pow(1 - std::pow(beta2_, step), 0.5));
    }
    if (parameters.empty())
      return;
    if (weight_decay_ != 0.)
      at::_foreach_mul_(parameters, 1 - lr * weight_decay_);
    at::_foreach_lerp_(first, gradients, 1 - beta1);
    at::_foreach_mul_(second, beta2_);
    at::_foreach_addcmul_(second, gradients, gradients, 1 - beta2_);
    auto denominator = at::_foreach_sqrt(second);
    at::_foreach_div_(denominator, corrections);
    at::_foreach_add_(denominator, epsilon_);
    at::_foreach_addcdiv_(parameters, first, denominator, sizes);
  }

  static constexpr double beta2_ = 0.999, epsilon_ = 1e-8;
  std::vector<Tensor> parameters_, first_, second_;
  double lr_, weight_decay_;
  std::vector<int64_t> steps_;
};
} // namespace ni26::neural
