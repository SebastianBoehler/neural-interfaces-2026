#include "tail_state.h"
#include <torch/torch.h>
#include <algorithm>

namespace ni26 {
using torch::indexing::Slice;
double early_boost(const Params& z, const std::string& field, int64_t t, double gate) {
  double horizon = z.number("eb_T");
  return 1 + z.number(field) * std::clamp((2*horizon - t)/horizon, 0., 1.) * gate;
}
Tensor GaussianState::apply(const Tensor& x, const Tensor& p) const {
  return normalize(p * lda_posterior((x-mean)/sd, centers, inverse).pow(exponent));
}
Tensor EMStage::view(const Tensor& raw, const Tensor& extra) {
  auto x = raw.to(torch::kFloat64);
  int64_t bins = z.integer("em_bins"), length = x.size(-1)/bins*bins;
  auto average = [&](const std::string& key) {
    return x.index_select(0, z.get(key).to(torch::kInt64)).mean(0);
  };
  auto reduce = [&](const Tensor& value) { return value.slice(0,0,length).reshape({bins,-1}).mean(1); };
  auto h = average("em_hl") - average("em_hr");
  auto v = average("em_ev_vt") - average("em_ev_vb");
  auto filter = SosFilter(z.get("em_event_sos"));
  auto dh = torch::diff(filter(h)), dv = torch::diff(filter(v));
  int64_t ebins = z.integer("em_ev_bins"), elength = (x.size(-1)-1)/ebins*ebins;
  auto a = torch::stack({dh.slice(0,0,elength),dv.slice(0,0,elength)}).abs();
  events.push_back(ni26::median(a,1));
  double k = z.number("em_ev_k");
  auto baseline = (k*z.get("em_ev_g").to(torch::kFloat64)+torch::stack(events).sum(0))/(k+events.size());
  std::vector<Tensor> rates;
  for (int i = 0; i < 2; ++i)
    for (const auto& q : z.get("em_ev_q").flatten().unbind(0))
      rates.push_back((a[i] > q*baseline[i]).to(torch::kFloat64).reshape({ebins,-1}).mean(1));
  std::vector<Tensor> out{
    reduce(h), reduce(average("em_vt")-average("em_vb")), (torch::cat(rates)+1e-3).log()};
  if (extra.defined()) out.push_back(extra.to(torch::kCPU,torch::kFloat64));
  return torch::cat(out);
}
Tensor EMStage::step(const Tensor& x, const Tensor& p, const Tensor& extra) {
  auto f = view(x,extra), out = p;
  int64_t count = features.size();
  gate = count == 0 ? 1. : std::clamp((z.number("eb_gate_hi")-margin_sum/count)/
      (z.number("eb_gate_hi")-z.number("eb_gate_lo")),0.,1.);
  state.reset();
  if (count >= z.integer("em_nmin")) {
    try {
      auto a = torch::stack(features), b = torch::stack(probabilities);
      auto mean = a.mean(0), sd = a.std(0,false).clamp_min(1e-3), standardized = (a-mean)/sd;
      auto q = b;
      Tensor centers, inverse;
      for (int64_t i = 0; i < z.integer("em_iters"); ++i) {
        centers = torch::matmul((q/(q.sum(0,true)+z.number("em_m0"))).t(),standardized);
        auto c = covariance(standardized-torch::matmul(q,centers));
        c = (1-z.number("em_shrink"))*c+z.number("em_shrink")*torch::eye(c.size(0),c.options())*c.trace()/c.size(0);
        inverse = torch::linalg_inv(c);
        q = normalize(b*lda_posterior(standardized,centers,inverse).pow(z.number("em_lam")));
      }
      double exponent = z.number("em_lam")*count/(count+z.number("em_n0"))*early_boost(z,"em_boost",count,gate);
      GaussianState proposed{mean,sd,centers,inverse,exponent};
      auto candidate = proposed.apply(f.unsqueeze(0),p.unsqueeze(0))[0];
      if (torch::isfinite(candidate).all().item<bool>()) { out=candidate; state=proposed; }
    } catch (const c10::LinAlgError&) { /* Existing reference keeps p for a singular session view. */ }
  }
  features.push_back(f.clone()); probabilities.push_back(p.clone());
  auto sorted = std::get<0>(p.sort());
  margin_sum += (sorted[-1]-sorted[-2]).item<double>();
  return out;
}
Tensor EMStage::reapply(int64_t count) const {
  require(count > 0 && count <= int64_t(features.size()), "invalid EM reapply prefix");
  auto b = torch::stack(std::vector<Tensor>(probabilities.begin(),probabilities.begin()+count));
  if (!state) return b;
  auto a = torch::stack(std::vector<Tensor>(features.begin(),features.begin()+count));
  return state->apply(a,b);
}
}
