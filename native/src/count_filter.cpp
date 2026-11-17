#include "ni26/output.h"
#include <torch/torch.h>
namespace ni26 {
using namespace torch::indexing;
struct CountFilter::Impl {
  Params z;
  std::vector<SosFilter> sos;
  Tensor al, cum, bounds, previous, grid;
  std::vector<Tensor> views;
  OutputStats info;
  int64_t H, K;
  explicit Impl(const Params &p) : z(p), sos(filters(p, "v6_cf_sos")) {
    auto L = p.get("v6_cf_Ls");
    H = L.size(0);
    K = p.integer("v6_cf_K");
    bounds = L.cumsum(1);
    auto k = torch::arange(K, torch::kFloat64);
    auto mesh = torch::meshgrid({k, k, k}, "ij");
    grid = torch::stack(mesh, -1);
    reset();
  }
  Impl(const Impl &o)
      : z(o.z), sos(o.sos), al(o.al.clone()), cum(o.cum.clone()),
        bounds(o.bounds),
        previous(o.previous.defined() ? o.previous.clone() : Tensor()),
        grid(o.grid), info(o.info), H(o.H), K(o.K) {
    for (auto &t : o.views)
      views.push_back(t.clone());
  }
  void reset() {
    info = {};
    views.clear();
    al = torch::zeros({H, K, K, K}, torch::kFloat64);
    cum = torch::zeros({H}, torch::kFloat64);
    previous = Tensor();
  }
  Tensor view(const Tensor &x) {
    std::vector<Tensor> v;
    for (auto &f : sos)
      v.push_back((f(x).var(-1, false) + 1e-12).log());
    return torch::matmul(torch::cat(v).to(torch::kFloat32).to(torch::kFloat64),
                         z.get("v6_cf_W")) /
           z.get("v6_cf_sw");
  }
  double lag(int64_t n) {
    double near = 0, far = 0;
    int64_t nn = 0, nf = 0;
    for (int64_t i = 0; i < n; i++)
      for (int64_t j = i + 1; j < n; j++) {
        double d = (views[i] - views[j]).norm().item<double>();
        if (j - i <= 3) {
          near += d;
          nn++;
        }
        if (j - i >= 15) {
          far += d;
          nf++;
        }
      }
    return (near / nn) / (far / nf);
  }
  Tensor urn(const Tensor &p) {
    auto block = (bounds <= info.windows).sum(1);
    auto reset =
        previous.defined() ? block != previous : torch::ones({H}, torch::kBool);
    previous = block.clone();
    auto cap = z.get("v6_cf_cap")
                   .index({torch::arange(H, torch::kLong), block})
                   .to(torch::kFloat64);
    al.index_put_({reset}, 0);
    auto selected = torch::nonzero(reset).flatten();
    al.index_put_({selected, 0, 0, 0}, 1);
    auto L = p.clamp(1e-9, 1).pow(z.number("v6_cf_tau"));
    L = L / L.sum();
    auto tr = (cap.reshape({H, 1, 1, 1, 1}) - grid.unsqueeze(0)).clamp_min(0) +
              z.number("v6_cf_floor");
    tr = tr / tr.sum(-1, true);
    auto v = al.unsqueeze(-1) * tr * L;
    auto out = v.reshape({H, -1, 3}).sum(1), next = torch::zeros_like(al);
    next.index({Slice(), Slice(1), Slice(), Slice()})
        .add_(v.index({Slice(), Slice(0, -1), Slice(), Slice(), 0}));
    next.index({Slice(), -1, Slice(), Slice()})
        .add_(v.index({Slice(), -1, Slice(), Slice(), 0}));
    next.index({Slice(), Slice(), Slice(1), Slice()})
        .add_(v.index({Slice(), Slice(), Slice(0, -1), Slice(), 1}));
    next.index({Slice(), Slice(), -1, Slice()})
        .add_(v.index({Slice(), Slice(), -1, Slice(), 1}));
    next.index({Slice(), Slice(), Slice(), Slice(1)})
        .add_(v.index({Slice(), Slice(), Slice(), Slice(0, -1), 2}));
    next.index({Slice(), Slice(), Slice(), -1})
        .add_(v.index({Slice(), Slice(), Slice(), -1, 2}));
    al = next / out.sum(1).reshape({H, 1, 1, 1});
    return out;
  }
};
CountFilter::CountFilter(const Params &z) : impl_(std::make_unique<Impl>(z)) {}
CountFilter::~CountFilter() = default;
CountFilter::CountFilter(const CountFilter &o)
    : impl_(std::make_unique<Impl>(*o.impl_)) {}
CountFilter &CountFilter::operator=(const CountFilter &o) {
  if (this != &o)
    impl_ = std::make_unique<Impl>(*o.impl_);
  return *this;
}
void CountFilter::reset() { impl_->reset(); }
OutputStats CountFilter::stats() const { return impl_->info; }
Tensor CountFilter::step(const Tensor &x, const Tensor &p) {
  auto &s = *impl_;
  if (s.info.dead)
    return p;
  try {
    auto checks = s.z.get("v6_cf_checks").flatten();
    if (!s.info.stopped)
      for (int64_t i = 0; i < checks.numel(); i++)
        if (checks[i].item<int64_t>() == s.info.windows &&
            s.lag(s.info.windows) < s.z.get("v6_cf_thr")[i].item<double>())
          s.info.stopped = s.info.windows;
    s.views.push_back(s.view(x));
    auto pr = s.urn(p);
    auto weights = torch::softmax(s.z.get("v6_cf_lp") + s.cum, 0);
    auto out = (weights.unsqueeze(1) * pr).sum(0);
    out = out / out.sum();
    s.cum = s.cum + pr.sum(1).clamp_min(1e-300).log();
    s.info.windows++;
    finite(out, "count filter");
    return s.info.stopped ? out : p;
  } catch (const std::exception &) {
    s.info.dead = true;
    s.info.errors++;
    s.info.windows++;
    return p;
  }
}
} // namespace ni26
