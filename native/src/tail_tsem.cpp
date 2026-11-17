#include "tail_state.h"
#include <torch/torch.h>
#include <algorithm>

namespace ni26 {
Tensor TSEMStage::step(const Tensor& u, const Tensor& p, double gate) {
  int64_t count = features.size();
  auto out = p;
  state.reset();
  if (count >= z.integer("tsem_nmin")) {
    try {
      auto a = torch::stack(features), conf = torch::stack(confidence), lab = torch::stack(labels);
      auto mean = a.mean(0), sd = a.std(0,false).clamp_min(1e-3), standardized = (a-mean)/sd;
      int64_t k = std::max<int64_t>(2,z.number("tsem_rho")*count/3);
      std::vector<Tensor> selected, centers, residuals;
      bool enough = true;
      for (int64_t c = 0; c < 3; ++c) {
        auto ids = torch::nonzero(lab==c).flatten();
        auto order = torch::argsort(-conf.index_select(0,ids));
        ids = ids.index_select(0,order.slice(0,0,std::min(k,ids.numel())));
        selected.push_back(ids); enough &= ids.numel() >= 2;
      }
      if (enough) {
        for (int64_t c = 0; c < 3; ++c) {
          auto rows = standardized.index_select(0,selected[c]);
          auto center = rows.sum(0);
          double denominator = selected[c].numel()+z.number("tsem_m0");
          if (prior.defined()) { center=center+prior_weight*prior[c]; denominator+=prior_weight; }
          center=center/denominator; centers.push_back(center); residuals.push_back(rows-center);
        }
        auto c = covariance(torch::cat(residuals));
        c = (1-z.number("tsem_shrink"))*c+z.number("tsem_shrink")*torch::eye(c.size(0),c.options())*c.trace()/c.size(0);
        double exponent=z.number("tsem_lam")*count/(count+z.number("tsem_n0"))*early_boost(z,"tsem_boost",count,gate);
        GaussianState proposed{mean,sd,torch::stack(centers),torch::linalg_inv(c),exponent};
        auto candidate=proposed.apply(u.unsqueeze(0),p.unsqueeze(0))[0];
        if (torch::isfinite(candidate).all().item<bool>()) { out=candidate; state=proposed; }
      }
    } catch (const c10::LinAlgError&) { /* Same singular-view behavior as the fixed reference. */ }
  }
  features.push_back(u.clone()); confidence.push_back(p.max().clone()); labels.push_back(p.argmax().clone());
  return out;
}
Tensor TSEMStage::reapply(int64_t count, const Tensor& p) const {
  if (!state) return p;
  require(count > 0 && count <= int64_t(features.size()), "invalid TSEM reapply prefix");
  auto a=torch::stack(std::vector<Tensor>(features.begin(),features.begin()+count));
  return state->apply(a,p);
}
}
