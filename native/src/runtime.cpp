#include "ni26/runtime.h"
#include "ni26/core.h"
#include "ni26/routing.h"
#include <torch/torch.h>
#include <algorithm>
#include <cctype>
#include <future>

namespace ni26 {
using torch::indexing::Slice;
struct Runtime::Impl {
  Params z;
  Tensor weights, permutation;
  TangentCore core1,core2;
  NetBank bank;
  EyeNet eye;
  RestMember rest;
  FineTune fine_tune;
  Tail shadow;
  Route inner,outer;
  CountFilter count_filter;
  CausalPers personalization;
  bool recognized=false;
  int64_t windows=0, recognized_at=-1;
  std::vector<Tensor> history;
  Impl(const Params& params,torch::Device device,const std::filesystem::path& root)
    : z(params),weights(normalize(z.get("member_weights").to(torch::kFloat64))),
      core1(z.subset("ts_")),core2(z.subset("ts2_")),bank(z,device,root),
      eye(z,device,root),rest(z,device,root),fine_tune(z,weights,device,root),
      shadow(z),inner(z,weights[0].item<double>()),
      outer(z,weights[0].item<double>()*(1-z.number("v6_rest_w")),true),
      count_filter(z),personalization(z) {
    require(z.integer("v6_routed")==1 && z.integer("v6_ft_labels")==0,
            "native port currently requires the actual frozen v6.6-cap250 routed/label recipe");
    require(z.integer("v6_cf_on")==1 && z.integer("v6_cp_on")==1 && z.integer("v6_ft_cap")==250,
            "native artifact is not the pinned cap250 count-filter/personalization recipe");
    if (z.integer("v6_tpri_on")) outer.set_prior(z.number("v6_tpri_kappa"),z.get("v6_tpri"));
  }
  Tensor mix_outer(const Tensor& mix,const NetworkStep& networks,const Tensor& rest_p,
                   const std::map<std::string,Tensor>& tuned) const {
    auto result=mix.clone();
    std::map<int64_t,Tensor> members;
    for (size_t i=0;i<networks.ids.size();++i) members.emplace(networks.ids[i],networks.probabilities[i]);
    for (const auto& slot : z.texts("v6_slots")) {
      if (!tuned.count(slot)) continue;
      const auto& p=tuned.at(slot);
      const auto& ids=fine_tune.slots().at(slot);
      auto family=torch::zeros_like(mix);
      double total=0.;
      for (auto i:ids) { auto w=weights[1+i].item<double>();total+=w;family=family+w*members.at(i); }
      auto updated=(family/total+p-members.at(fine_tune.members().at(slot))).clamp_min(1e-6);
      result=result-family+total*normalize(updated);
    }
    return (1-z.number("v6_rest_w"))*result+z.number("v6_rest_w")*rest_p;
  }
};
Runtime::Runtime(const Params& z,torch::Device device,const std::filesystem::path& root)
    : impl_(std::make_unique<Impl>(z,device,root)) {}
Runtime::~Runtime()=default;
Runtime::Runtime(const Runtime& other):impl_(std::make_unique<Impl>(*other.impl_)) {}
Runtime& Runtime::operator=(const Runtime& other) {
  if(this!=&other) impl_=std::make_unique<Impl>(*other.impl_);
  return *this;
}
void Runtime::reset() {
  auto& s=*impl_;
  s.core1.reset();s.core2.reset();s.bank.reset();s.eye.reset();s.rest.reset();s.fine_tune.reset();
  s.shadow.reset();s.inner.reset();s.outer.reset();s.count_filter.reset();s.personalization.reset();
  s.recognized=false;s.windows=0;s.recognized_at=-1;s.history.clear();
}
Tensor Runtime::step(const Tensor& window) {
  auto& s=*impl_;
  require(window.scalar_type()==torch::kFloat32 && window.sizes()==torch::IntArrayRef({41,480}),
          "native model input must be float32(41,480)");
  // Input sanitization and montage mapping are native, not adapter computation.
  auto x=torch::nan_to_num(window.to(torch::kCPU),0.,0.,0.);
  if(s.permutation.defined()) {
    auto mapped=torch::zeros_like(x),valid=s.permutation>=0;
    mapped.index_put_({valid},x.index_select(0,s.permutation.index({valid})));x=mapped;
  }
  torch::NoGradGuard no_grad;
  // Independent members process this same window concurrently; no future input.
  auto cores=std::async(std::launch::async,[&s,x] {
    torch::NoGradGuard worker_no_grad;
    auto [first,features]=s.core1.step(x);
    auto second=s.core2.step(x).first;
    auto centered=features-s.z.get("tsem_m");
    auto view=torch::cat({torch::matmul(centered,s.z.get("tsem_V").to(torch::kFloat64)),
                         torch::matmul(centered,s.z.get("route_V_hi").to(torch::kFloat64))});
    return std::make_tuple(first,second,view);
  });
  auto networks=s.bank.step(x);
  auto [first,second,view]=cores.get();
  auto mix=s.weights[0]*first;
  for(size_t i=0;i<networks.ids.size();++i) mix=mix+s.weights[1+networks.ids[i]]*networks.probabilities[i];
  mix=mix+s.weights[-1]*second;finite(mix,"native inner mixture");
  auto shadow=s.shadow.step(x,view.slice(0,0,s.z.get("tsem_V").size(1)),mix);
  auto old=s.inner.step(x,view,mix,first,shadow);
  auto contrast=s.inner.last_contrast(),identity=s.inner.last_identity();
  bool newly_recognized=!s.recognized && ((contrast && contrast->weight>0) || (identity && identity->weight>0));
  if(newly_recognized) { s.recognized=true;s.recognized_at=s.windows;s.fine_tune.relabel(s.history); }
  s.history.push_back(old.clone());
  auto extra=s.eye.step(x),rest=s.rest.step(x);
  std::map<std::string,Tensor> tuned;
  if(!s.recognized) {
    RelabelFn relabel;
    if(s.z.integer("v6_ft_relabel")) relabel=[&s](int64_t count)->std::optional<Tensor>{
      if(s.recognized)return std::nullopt;
      return s.inner.reapply(count);
    };
    tuned=s.fine_tune.step(networks,old,relabel);
  }
  auto outer_mix=s.mix_outer(mix,networks,rest,tuned);finite(outer_mix,"native outer mixture");
  auto out=s.outer.step_given(x,view,outer_mix,first,extra,contrast,identity);
  finite(out,"native routed posterior");
  auto counted=s.count_filter.step(x,out);
  auto result=s.personalization.step(x,out,counted,s.recognized);
  finite(result,"native final scores");++s.windows;
  return result;
}
Tensor Runtime::predict(const Tensor& windows) {
  require(windows.dim()==3,"native predict input must have batch,channel,time dimensions");
  std::vector<Tensor> labels;
  for(const auto& x:windows.unbind(0)) labels.push_back(step(x).argmax());
  require(!labels.empty(),"empty native prediction batch");
  return torch::stack(labels).to(torch::kInt64);
}
void Runtime::set_channels(const std::vector<std::string>& names) {
  const std::vector<std::string> montage={"C2","CP2","P2","PZ","CPZ","CZ","C1","CP1","P1","P3","PO7","P5","TP7","CP5","CP3","C3","PO8","P6","P4","TP8","CP6","CP4","C6","C4","FC6","FC4","FC2","F6","F4","F2","AF4","AF3","F5","F3","F1","FZ","FC5","FC3","FC1","FCZ","C5"};
  std::vector<std::string> normalized=names;
  for(auto& name:normalized) {
    auto first=name.find_first_not_of(" \t\n\r"),last=name.find_last_not_of(" \t\n\r");
    name=first==std::string::npos?"":name.substr(first,last-first+1);
    for(auto& c:name)c=std::toupper(static_cast<unsigned char>(c));
  }
  int matches=0;std::vector<int64_t> indices;
  for(const auto& name:montage) {
    auto it=std::find(normalized.begin(),normalized.end(),name);
    indices.push_back(it==normalized.end()?-1:it-normalized.begin());matches+=it!=normalized.end();
  }
  impl_->permutation=normalized==montage || matches<20?Tensor{}:torch::tensor(indices,torch::kInt64);
}
RuntimeStats Runtime::stats() const {
  auto& s=*impl_;
  return {s.windows,s.recognized_at,s.fine_tune.stats(),s.count_filter.stats(),s.personalization.stats()};
}
}
