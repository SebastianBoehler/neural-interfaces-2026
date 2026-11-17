#include "ni26/linear.h"
#include "ni26/math.h"
#include "linear_internal.h"
#include <torch/torch.h>
#include <algorithm>
#include <cmath>

namespace ni26 {
namespace detail {
void linear_input(const Tensor& x, int64_t dimensions) {
  require(x.defined() && x.device().is_cpu() && x.scalar_type()==torch::kFloat64 &&
          x.is_contiguous() && x.dim()==dimensions && x.numel()>0 && !x.requires_grad(),
          "linear input must be detached contiguous nonempty CPU float64 with correct dimensions");
  const auto* values=x.data_ptr<double>();
  for (int64_t i=0;i<x.numel();++i)
    require(std::isfinite(values[i]), "nonfinite native linear input");
}
}
namespace {
class Objective {
 public:
  Objective(const Tensor& x, const Tensor& y, double c)
      : x_(x), y_(y), c_(c), n_(x.size(0)), d_(x.size(1)),
        logits_(torch::empty({n_,3},x.options())),
        weights_(torch::empty({3,d_},x.options())),
        gradient_(torch::empty({3,d_},x.options())) {}
  double operator()(const double* theta, double* gradient) {
    auto* weights=weights_.data_ptr<double>();
    for (int64_t j=0;j<d_;++j) for (int k=0;k<3;++k) {
      auto value=theta[3*j+k];
      require(std::isfinite(value),"nonfinite native linear parameters");
      weights[k*d_+j]=value;
    }
    for (int k=0;k<3;++k)
      require(std::isfinite(theta[3*d_+k]),"nonfinite native linear bias");
    torch::mm_out(logits_,x_,weights_.t());
    auto* residual=logits_.data_ptr<double>();
    const auto* target=y_.data_ptr<int64_t>();
    double loss=0, bias_gradient[3]{};
    for (int64_t i=0;i<n_;++i) {
      double z[3], largest=-INFINITY, denominator=0;
      for (int k=0;k<3;++k) {
        z[k]=residual[3*i+k]+theta[3*d_+k]; largest=std::max(largest,z[k]);
      }
      for (double value:z) denominator+=std::exp(value-largest);
      loss+=largest+std::log(denominator)-z[target[i]];
      for (int k=0;k<3;++k) {
        auto value=(std::exp(z[k]-largest)/denominator-(target[i]==k))/n_;
        residual[3*i+k]=value; bias_gradient[k]+=value;
      }
    }
    torch::mm_out(gradient_,logits_.t(),x_);
    const auto* class_gradient=gradient_.data_ptr<double>();
    double penalty=0;
    for (int64_t j=0;j<d_;++j) for (int k=0;k<3;++k) {
      auto index=3*j+k;
      penalty+=theta[index]*theta[index];
      gradient[index]=class_gradient[k*d_+j]+theta[index]/(c_*n_);
    }
    for (int k=0;k<3;++k) gradient[3*d_+k]=bias_gradient[k];
    return loss/n_+penalty/(2*c_*n_);
  }
 private:
  Tensor x_, y_;
  double c_;
  int64_t n_,d_;
  Tensor logits_,weights_,gradient_;
};
}
static LinearFit fit_reference(const Tensor& x, const Tensor& y, double c,
                               detail::LinearReference reference) {
  torch::NoGradGuard guard;
  detail::linear_input(x,2);
  require(std::isfinite(c) && c>0,"linear C must be finite and positive");
  require(y.defined() && y.device().is_cpu() && y.scalar_type()==torch::kInt64 &&
          y.is_contiguous() && y.dim()==1 && y.size(0)==x.size(0),
          "native linear labels must be contiguous CPU int64 matching source rows");
  bool seen[3]{};
  for (int64_t i=0;i<y.numel();++i) {
    auto value=y.data_ptr<int64_t>()[i];
    require(value>=0 && value<3,"native linear labels must be 0,1,2");seen[value]=true;
  }
  require(seen[0] && seen[1] && seen[2],"native linear training requires all three classes");
  Objective objective(x,y,c);
  auto fit=detail::minimize_lbfgsb(3*(x.size(1)+1),
      [&](const double* theta,double* gradient) { return objective(theta,gradient); }, reference);
  auto theta=torch::from_blob(fit.parameters.data(),{x.size(1)+1,3},x.options()).t().contiguous();
  LinearFit result;
  result.weights=theta.slice(1,0,x.size(1)).contiguous();
  result.bias=theta.select(1,x.size(1)).contiguous();
  result.nit=fit.nit;result.nfev=fit.nfev;
  result.converged=fit.status==4;result.status=fit.status;result.message_code=fit.message_code;
  result.objective=fit.objective;
  for (auto value:fit.gradient) result.max_gradient=std::max(result.max_gradient,std::abs(value));
  return result;
}
LinearFit fit_multinomial(const Tensor& x, const Tensor& y, double c) {
  return fit_reference(x,y,c,detail::LinearReference::Qualified343);
}
LinearFit fit_multinomial_core(const Tensor& x, const Tensor& y, double c) {
  return fit_reference(x,y,c,detail::LinearReference::ActualCore);
}

}
