#include "networks_internal.h"
#include <torch/torch.h>

namespace ni26::neural {
namespace {
std::string bytes(const Tensor& value) {
  auto x=value.detach().cpu().contiguous();
  return std::string(static_cast<const char*>(x.const_data_ptr()),x.nbytes());
}
}
std::string input_key(const Params& p) {
  std::string result;
  for(const auto* key:{"align","k","G","scale","shrink"}) result+=bytes(p.get(key));
  return result;
}
InputStream::InputStream(const Params& p):align_(p.get("align").item<bool>()),
    scale_(p.number("scale")),k_(p.number("k")),shrink_(p.number("shrink")),
    prior_(p.get("G").to(torch::kCPU,torch::kFloat64).clone()) {
  require(scale_>0 && k_>=0 && shrink_>=0 && shrink_<=1,"invalid neural input parameters");
  require(prior_.sizes()==torch::IntArrayRef({41,41}),"neural input prior is not41×41");
  reset();
}
InputStream::InputStream(const InputStream& x):align_(x.align_),scale_(x.scale_),k_(x.k_),
    shrink_(x.shrink_),prior_(x.prior_),sum_(x.sum_.clone()),count_(x.count_) {}
void InputStream::reset(){ count_=0;sum_=torch::zeros_like(prior_); }
Tensor InputStream::step(const Tensor& input) {
  require(input.device().is_cpu() && input.scalar_type()==torch::kFloat32 &&
      input.sizes()==torch::IntArrayRef({41,480}),"neural input must be CPU float32(41,480)");
  finite(input,"network window");auto x=input.to(torch::kFloat64);
  if(!align_) return (x/scale_).to(torch::kFloat32);
  // The reference centers its float32 window before float64 covariance products.
  auto centered=(input-input.mean(-1,true)).to(torch::kFloat64);
  auto c=torch::matmul(centered,centered.t())/480.;
  c=(1-shrink_)*c+shrink_*c.trace()/41*torch::eye(41,c.options());
  sum_=sum_+c;++count_;
  auto p=spectral((k_*prior_+sum_)/(k_+count_),SpectralFunction::InverseRoot);
  return torch::matmul(p,x).to(torch::kFloat32);
}
void zero_stream(torch::jit::Module& module) {
  for(auto item:module.named_buffers(true)) {
    auto split=item.name.find_last_of('.');auto name=item.name.substr(split==std::string::npos?0:split+1);
    if(name=="s1" || name=="s2" || name=="n") { item.value.detach_();item.value.zero_(); }
  }
}
std::vector<Tensor> trainable(torch::jit::Module& module) {
  std::vector<Tensor> result;
  for(auto p:module.parameters()) if(p.requires_grad())result.push_back(p);
  for(auto item:module.named_buffers(true)) {
    auto split=item.name.find_last_of('.');auto name=item.name.substr(split==std::string::npos?0:split+1);
    if(name=="w" || name=="b")result.push_back(item.value);
  }
  require(!result.empty(),"fine-tuning module exposes no trainable tensors");return result;
}
}
