#include "tail_state.h"
#include <torch/torch.h>

namespace ni26 {
struct Tail::Impl {
  Params z;
  bool eye;
  EMStage em;
  TSEMStage tsem;
  std::vector<Tensor> history;
  Impl(const Params& params, bool columns) : z(params),eye(columns),em(params),tsem(params) {}
};
Tail::Tail(const Params& z, bool eye) : impl_(std::make_unique<Impl>(z,eye)) {}
Tail::~Tail() = default;
// Historical entries are immutable snapshots; all updates replace tensors rather
// than changing their storage. Copying containers isolates the evolving state.
Tail::Tail(const Tail& other) : impl_(std::make_unique<Impl>(*other.impl_)) {}
Tail& Tail::operator=(const Tail& other) {
  if (this != &other) impl_=std::make_unique<Impl>(*other.impl_);
  return *this;
}
void Tail::reset() { impl_=std::make_unique<Impl>(impl_->z,impl_->eye); }
Tensor Tail::step(const Tensor& x,const Tensor& u,const Tensor& mix,const Tensor& extra) {
  auto& s=*impl_;
  require(extra.defined()==s.eye, "tail eye-column configuration mismatch");
  double k=s.z.number("rb_k");
  auto sum=s.history.empty() ? torch::zeros_like(mix) : torch::stack(s.history).sum(0);
  auto mean=(k/mix.numel()+sum)/(k+s.history.size());
  auto q=normalize(mix/mean.pow(s.z.number("rb_alpha")));
  s.history.push_back(mix.clone());
  if (int64_t(s.history.size()) > s.z.integer("rb_w")) s.history.erase(s.history.begin());
  q=s.em.step(x,q,extra);
  return s.tsem.step(u,q,s.em.gate);
}
Tensor Tail::reapply(int64_t count) const {
  return impl_->tsem.reapply(count,impl_->em.reapply(count));
}
void Tail::set_prior(double kappa,const Tensor& means) {
  impl_->tsem.prior_weight=kappa;
  impl_->tsem.prior=means.defined() ? means.clone() : Tensor{};
}
}
