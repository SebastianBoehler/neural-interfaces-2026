#include "ni26/routing.h"
#include <iostream>
#include <torch/torch.h>
namespace ni26 {
struct Route::Impl {
  Params z;
  ContrastGate contrast;
  IdentityGate identity;
  OwnStage own;
  Tail tail;
  double a, g, kappa = 0;
  int64_t dim;
  Tensor prior, stage;
  std::vector<Tensor> U, Q;
  Decision last_c, last_i;
  RouteStats counters;
  Impl(const Params &p, double share, bool outer)
      : z(p), contrast(p), identity(p), own(p), tail(p, outer), a(share),
        g(p.number("route_slot_g")), dim(p.get("tsem_V").size(1)) {
    for (const auto *key : {"route_ps1_W", "route_ps1_c"})
      z.tensors[key] = p.get(key).to(torch::kFloat64);
  }
  Impl(const Impl &o)
      : z(o.z), contrast(o.contrast), identity(o.identity), own(o.own),
        tail(o.tail), a(o.a), g(o.g), kappa(o.kappa), dim(o.dim),
        prior(o.prior), stage(o.stage.defined() ? o.stage.clone() : Tensor()),
        last_c(o.last_c), last_i(o.last_i), counters(o.counters) {
    for (auto &t : o.U)
      U.push_back(t.clone());
    for (auto &t : o.Q)
      Q.push_back(t.defined() ? t.clone() : Tensor());
  }
  Tensor apply(const Tensor &x, const Tensor &u, const Tensor &mix,
               const Tensor &core1, const Tensor &shadow, const Tensor &extra,
               Decision c, Decision i) {
    last_c = c;
    last_i = i;
    int64_t j = -1;
    double r = 0;
    if (c) {
      j = c->person;
      r = c->weight;
    }
    if (i && i->weight > r) {
      j = i->person;
      r = i->weight;
    }
    auto p = mix;
    if (r > 0) {
      auto history = U;
      history.push_back(u);
      auto A = torch::stack(history), mean = A.mean(0);
      auto v = U.size() >= 4 ? (u - mean) / A.std(0, false).clamp_min(1e-3)
                             : u - mean;
      auto h = posterior(torch::matmul(v, z.get("route_ps1_W")[j].t()) +
                         z.get("route_ps1_c")[j]);
      auto candidate =
          (mix - r * a * core1 + g * r * a * h) / (1 - r * a + g * r * a);
      if (torch::isfinite(candidate).all().item<bool>())
        p = candidate;
    }
    if (prior.defined())
      tail.set_prior(kappa * r, r > 0 ? prior[j] : Tensor());
    auto out = tail.step(x, u.slice(0, 0, dim), p, extra);
    stage = Tensor();
    counters.windows++;
    counters.routed += r > 0;
    try {
      bool valid = own.observe(x);
      counters.degenerate += !valid;
      if (r > 0 && valid) {
        auto result = own.apply(out, j, r);
        if (result) {
          out = result->first;
          stage = torch::cat(
              {torch::tensor({double(j), r}, out.options()), result->second});
          counters.staged++;
        }
      }
    } catch (const std::exception &error) {
      std::cerr << "[own-stage] error at window " << counters.windows - 1
                << ": " << error.what() << std::endl;
      counters.errors++;
    }
    U.push_back(u.clone());
    Q.push_back(shadow.defined() ? shadow.clone() : Tensor());
    return out;
  }
};
Route::Route(const Params &z, double share, bool outer)
    : impl_(std::make_unique<Impl>(z, share, outer)) {}
Route::~Route() = default;
Route::Route(const Route &o) : impl_(std::make_unique<Impl>(*o.impl_)) {}
Route &Route::operator=(const Route &o) {
  if (this != &o)
    impl_ = std::make_unique<Impl>(*o.impl_);
  return *this;
}
void Route::reset() {
  auto &s = *impl_;
  s.U.clear();
  s.Q.clear();
  s.last_c.reset();
  s.last_i.reset();
  s.stage = Tensor();
  s.counters = {};
  s.tail.reset();
  s.identity.reset();
  s.own.reset();
}
Tensor Route::step(const Tensor &x, const Tensor &u, const Tensor &mix,
                   const Tensor &core1, const Tensor &shadow) {
  auto &s = *impl_;
  auto c = s.contrast.step(s.U, s.Q), i = s.identity.step(x);
  return s.apply(x, u, mix, core1, shadow, {}, c, i);
}
Tensor Route::step_given(const Tensor &x, const Tensor &u, const Tensor &mix,
                         const Tensor &core1, const Tensor &extra, Decision c,
                         Decision i) {
  return impl_->apply(x, u, mix, core1, {}, extra, c, i);
}
Tensor Route::reapply(int64_t count) const {
  return impl_->tail.reapply(count);
}
Decision Route::last_contrast() const { return impl_->last_c; }
Decision Route::last_identity() const { return impl_->last_i; }
Tensor Route::last_stage() const { return impl_->stage; }
RouteStats Route::stats() const { return impl_->counters; }
void Route::set_core1_share(double a) { impl_->a = a; }
void Route::set_prior(double k, const Tensor &m) {
  impl_->kappa = k;
  impl_->prior = m.to(torch::kFloat64).clone();
}
} // namespace ni26
