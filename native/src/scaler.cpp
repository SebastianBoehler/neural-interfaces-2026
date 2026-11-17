#include "ni26/linear.h"
#include "ni26/math.h"
#include "linear_internal.h"
#include <torch/torch.h>
#include <cmath>
#include <limits>
#include <vector>

namespace ni26 {
StandardScaler StandardScaler::fit(const Tensor& x) {
  torch::NoGradGuard guard;
  detail::linear_input(x,2);
  auto n=x.size(0),d=x.size(1);
  auto mean=torch::zeros({d},x.options()),scale=torch::empty({d},x.options());
  auto* means=mean.data_ptr<double>();
  const auto* values=x.data_ptr<double>();
  for (int64_t i=0;i<n;++i) for (int64_t j=0;j<d;++j) means[j]+=values[i*d+j];
  for (int64_t j=0;j<d;++j) {
    means[j]/=n;require(std::isfinite(means[j]),"native scaler mean overflow");
  }
  std::vector<double> correction(d),squared(d);
  for (int64_t i=0;i<n;++i) for (int64_t j=0;j<d;++j) {
    auto delta=values[i*d+j]-means[j];correction[j]+=delta;squared[j]+=delta*delta;
  }
  const auto eps=std::numeric_limits<double>::epsilon();
  auto* scales=scale.data_ptr<double>();
  for (int64_t j=0;j<d;++j) {
    auto variance=(squared[j]-correction[j]*correction[j]/n)/n;
    require(std::isfinite(variance),"native scaler variance overflow");
    // sklearn's float64 near-constant criterion, including exact zero variance.
    auto offset=n*means[j]*eps;
    auto upper=n*eps*variance+offset*offset;
    scales[j]=variance<=upper?1.:std::sqrt(variance);
    require(std::isfinite(scales[j]) && scales[j]>0,"invalid native scaler scale");
  }
  return {mean,scale};
}
Tensor StandardScaler::transform(const Tensor& x) const {
  torch::NoGradGuard guard;
  require(x.dim()==1 || x.dim()==2,"native scaler input must be a row or matrix");
  detail::linear_input(x,x.dim());
  detail::linear_input(mean,1);detail::linear_input(scale,1);
  require(mean.sizes()==scale.sizes() && x.size(-1)==mean.numel(),"native scaler width differs");
  for (int64_t j=0;j<scale.numel();++j)
    require(scale.data_ptr<double>()[j]>0,"native scaler scale must be positive");
  auto result=((x-mean)/scale).contiguous();
  detail::linear_input(result,result.dim());
  return result;
}
}
