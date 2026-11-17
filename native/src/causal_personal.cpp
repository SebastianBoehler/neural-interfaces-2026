#include "ni26/output.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <torch/torch.h>
namespace ni26 {
using namespace torch::indexing;
namespace {
Tensor oas(const Tensor &raw) {
  auto x = raw - raw.mean(1, true);
  auto C = torch::matmul(x, x.t()) / x.size(1);
  double mu = C.trace().item<double>() / C.size(0),
         alpha = C.square().mean().item<double>();
  double denominator = (x.size(1) + 1.) * (alpha - mu * mu / C.size(0));
  double shrink =
      denominator == 0 ? 1. : std::min((alpha + mu * mu) / denominator, 1.);
  return (1 - shrink) * C + shrink * mu * torch::eye(C.size(0), C.options());
}
Tensor shrink_cov(const Tensor &raw) {
  auto centered = raw - raw.mean(0);
  auto variance = centered.square().mean(0), scale = variance.sqrt();
  auto upper = raw.size(0) * std::numeric_limits<double>::epsilon() * variance +
               torch::square(raw.size(0) * raw.mean(0) *
                             std::numeric_limits<double>::epsilon());
  scale = torch::where(variance <= upper, torch::ones_like(scale), scale);
  auto X = centered / scale;
  X = X - X.mean(0);
  int64_t n = X.size(0), d = X.size(1);
  auto C = torch::matmul(X.t(), X) / n;
  double mu = C.trace().item<double>() / d;
  double delta2 = C.square().sum().item<double>();
  double beta2 = torch::matmul(X.square().t(), X.square()).sum().item<double>();
  double beta = (beta2 / n - delta2) / (d * n);
  double delta = (delta2 - 2 * mu * C.trace().item<double>() + d * mu * mu) / d;
  beta = std::min(beta, delta);
  double amount = beta == 0 ? 0 : beta / delta;
  auto out = (1 - amount) * C + amount * mu * torch::eye(d, C.options());
  return scale.unsqueeze(1) * out * scale.unsqueeze(0);
}
Tensor logvariance(const Tensor &W, const Tensor &C) {
  return (W.unsqueeze(0) * torch::matmul(C, W)).sum(1).log();
}
} // namespace
struct CausalPers::Impl {
  Params z;
  std::vector<SosFilter> sos;
  std::vector<Tensor> C, P, Ws;
  Tensor coefficient, intercept, classes;
  OutputStats info;
  explicit Impl(const Params &p) : z(p), sos(filters(p, "v6_cp_sos")) {}
  Impl(const Impl &o)
      : z(o.z), sos(o.sos),
        coefficient(o.coefficient.defined() ? o.coefficient.clone() : Tensor()),
        intercept(o.intercept.defined() ? o.intercept.clone() : Tensor()),
        classes(o.classes.defined() ? o.classes.clone() : Tensor()),
        info(o.info) {
    for (auto &t : o.C)
      C.push_back(t.clone());
    for (auto &t : o.P)
      P.push_back(t.clone());
    for (auto &t : o.Ws)
      Ws.push_back(t.clone());
  }
  Tensor features(const Tensor &x) {
    std::vector<Tensor> result;
    for (auto &f : sos)
      result.push_back(oas(f(x)));
    return torch::stack(result);
  }
  void refit(int64_t t) {
    auto probabilities =
        torch::stack(std::vector<Tensor>(P.begin(), P.begin() + t));
    auto labels = probabilities.argmax(1),
         confidence = std::get<0>(probabilities.max(1));
    std::vector<int64_t> chosen;
    for (int64_t k = 0; k < 3; k++) {
      auto rows = torch::nonzero(labels == k).flatten();
      if (!rows.numel())
        continue;
      auto order = torch::argsort(-confidence.index({rows}));
      auto keep = rows.index({order.index(
          {Slice(0, std::max<int64_t>(
                        3, std::ceil(z.number("v6_cp_q") * rows.numel())))})});
      for (int64_t j = 0; j < keep.numel(); j++)
        chosen.push_back(keep[j].item<int64_t>());
    }
    std::sort(chosen.begin(), chosen.end());
    auto select = torch::tensor(chosen, torch::kLong);
    auto y = labels.index({select});
    classes = std::get<0>(torch::_unique(y, true, false));
    Ws.clear();
    coefficient = intercept = Tensor();
    if (classes.numel() < 2) {
      info.single_class++;
      return;
    }
    auto covs = torch::stack(std::vector<Tensor>(C.begin(), C.begin() + t))
                    .index({select});
    std::vector<Tensor> A;
    for (int64_t b = 0; b < covs.size(1); b++) {
      auto cb = covs.index({Slice(), b});
      std::vector<Tensor> filters;
      for (int64_t k = 0; k < classes.numel(); k++) {
        auto mask = y == classes[k];
        auto Ma = cb.index({mask}).mean(0), Mo = cb.index({~mask}).mean(0);
        Mo = .9 * Mo + .1 * Mo.trace() / Mo.size(0) *
                           torch::eye(Mo.size(0), Mo.options());
        auto L = torch::linalg_cholesky(Ma + Mo);
        auto Li = torch::linalg_inv(L);
        auto whiten = torch::matmul(torch::matmul(Li, Ma), Li.t());
        auto eig = torch::linalg_eigh(whiten);
        auto V = torch::linalg_solve(L.t(), std::get<1>(eig));
        filters.push_back(V.index({Slice(), Slice(0, 2)}));
        filters.push_back(V.index({Slice(), Slice(-2)}));
      }
      auto W = torch::cat(filters, 1);
      Ws.push_back(W);
      A.push_back(logvariance(W, cb));
    }
    auto X = torch::cat(A, 1);
    std::vector<Tensor> means;
    auto cov = torch::zeros({X.size(1), X.size(1)}, X.options());
    std::vector<double> prior;
    for (int64_t k = 0; k < classes.numel(); k++) {
      auto rows = X.index({y == classes[k]});
      double p = double(rows.size(0)) / X.size(0);
      prior.push_back(p);
      means.push_back(rows.mean(0));
      cov = cov + p * shrink_cov(rows);
    }
    auto mean = torch::stack(means);
    coefficient =
        std::get<0>(torch::linalg_lstsq(cov, mean.t(),
                                        std::numeric_limits<double>::epsilon(),
                                        "gelsd"))
            .t();
    intercept = -.5 * (mean * coefficient).sum(1) +
                torch::tensor(prior, torch::kFloat64).log();
  }
  Tensor predict(const Tensor &cw) {
    if (!coefficient.defined())
      return torch::full({3}, 1. / 3, torch::kFloat64);
    std::vector<Tensor> f;
    for (int64_t b = 0; b < cw.size(0); b++)
      f.push_back(logvariance(Ws[b], cw[b].unsqueeze(0)).flatten());
    auto scores = torch::matmul(torch::cat(f), coefficient.t()) + intercept;
    auto probs = torch::softmax(scores, 0);
    auto out = torch::zeros({3}, torch::kFloat64);
    out.index_put_({classes.to(torch::kLong)}, probs);
    return out;
  }
};
CausalPers::CausalPers(const Params &z) : impl_(std::make_unique<Impl>(z)) {}
CausalPers::~CausalPers() = default;
CausalPers::CausalPers(const CausalPers &o)
    : impl_(std::make_unique<Impl>(*o.impl_)) {}
CausalPers &CausalPers::operator=(const CausalPers &o) {
  if (this != &o)
    impl_ = std::make_unique<Impl>(*o.impl_);
  return *this;
}
void CausalPers::reset() {
  auto &s = *impl_;
  s.info = {};
  s.C.clear();
  s.P.clear();
  s.Ws.clear();
  s.coefficient = s.intercept = s.classes = Tensor();
}
OutputStats CausalPers::stats() const { return impl_->info; }
Tensor CausalPers::step(const Tensor &x, const Tensor &p, const Tensor &count_p,
                        bool recognized) {
  auto &s = *impl_;
  int64_t i = s.info.windows++;
  if (s.info.dead || s.info.stopped)
    return count_p;
  try {
    if (recognized) {
      s.info.stopped = i;
      s.C.clear();
      s.P.clear();
      s.Ws.clear();
      s.coefficient = Tensor();
      return count_p;
    }
    s.C.push_back(s.features(x));
    s.P.push_back(p.clone());
    int64_t start = s.z.integer("v6_cp_t0");
    if (i >= start && (i - start) % s.z.integer("v6_cp_k") == 0) {
      s.refit(i);
      s.info.refits++;
    }
    double w = s.z.number("v6_cp_cap") *
               std::clamp(double(i - s.z.integer("v6_cp_w0") + 1) /
                              s.z.integer("v6_cp_R"),
                          0., 1.);
    if (i < start || w <= 0)
      return count_p;
    auto result = count_p.clamp(1e-6, 1).log() +
                  w * s.predict(s.C[i]).clamp(1e-6, 1).log();
    finite(result, "causal personal scores");
    s.info.fused++;
    return result;
  } catch (const std::exception &error) {
    std::cerr << "[cpers] error at window " << s.info.windows - 1 << ": "
              << error.what() << std::endl;
    s.info.dead = true;
    s.info.errors++;
    return count_p;
  }
}
} // namespace ni26
