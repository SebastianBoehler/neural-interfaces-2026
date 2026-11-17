#include "ni26/linear.h"
#include "ni26/math.h"
#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
extern "C" void scipy_openblas_set_num_threads(int);
extern "C" const char* scipy_openblas_get_config();

namespace {
void must_fail(const std::function<void()>& operation,const std::string& expected) {
  try { operation(); }
  catch (const std::exception& error) {
    ni26::require(std::string(error.what()).find(expected)!=std::string::npos,
                  "wrong native linear validation error");
    return;
  }
  throw std::runtime_error("native linear validation accepted bad input");
}
double compare(const ni26::Tensor& actual,const ni26::Tensor& expected,double tolerance,
               const std::string& label) {
  ni26::require(actual.sizes()==expected.sizes(),label+" shape differs");
  auto maximum=(actual-expected).abs().max().item<double>();
  ni26::require(std::isfinite(maximum) && maximum<=tolerance,label+" differs "+std::to_string(maximum));
  return maximum;
}
}
int main(int argc,char** argv) {
  try {
    ni26::require(argc==3,"usage: ni26_linear_source FIXTURE OUTPUT");
    auto fixture=ni26::Params::load(argv[1]);
    auto threads=fixture.integer("threads");
    ni26::require(threads>0,"invalid source fixture thread count");
    torch::set_num_threads(threads);scipy_openblas_set_num_threads(static_cast<int>(threads));
    std::string config=scipy_openblas_get_config();
    ni26::require(config.find("USE64BITINT")==std::string::npos,"LP64 optimizer BLAS required");
    auto x=fixture.get("x"),y=fixture.get("y");
    auto bad=x.slice(0,0,16).slice(1,0,32).clone().contiguous();
    for (double value:{std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
      bad.index_put_({0,0},value);
      must_fail([&]{ni26::fit_multinomial(bad,y.slice(0,0,16).contiguous(),.3);},"nonfinite");
    }
    auto bad_y=y.clone();bad_y.index_put_({0},-1);
    must_fail([&]{ni26::fit_multinomial(x,bad_y,.3);},"0,1,2");
    must_fail([&]{ni26::fit_multinomial(x,y,0);},"C must");
    auto scaler=ni26::StandardScaler::fit(fixture.get("scaler_x"));
    auto mean_error=compare(scaler.mean,fixture.get("scaler_mean"),1e-12,"native scaler mean");
    auto scale_error=compare(scaler.scale,fixture.get("scaler_scale"),1e-12,"native scaler scale");
    auto transform_error=compare(scaler.transform(fixture.get("scaler_x")),fixture.get("scaler_expected"),1e-12,"native scaler transform");
    auto tick=std::chrono::steady_clock::now();
    auto fit=ni26::fit_multinomial(x,y,fixture.number("C"));
    auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();
    ni26::require(fit.converged,"native L-BFGS-B did not converge status="+std::to_string(fit.status)+"/"+std::to_string(fit.message_code));
    ni26::require(fit.nit==fixture.integer("reference_nit") && fit.nfev==fixture.integer("reference_nfev"),"native optimizer trajectory counters differ");
    auto w=fit.weights.clone().set_requires_grad(true),b=fit.bias.clone().set_requires_grad(true);
    auto logits=torch::matmul(x,w.t())+b;
    auto loss=(torch::logsumexp(logits,1)-logits.gather(1,y.unsqueeze(1)).squeeze(1)).mean()+w.square().sum()/(2*fixture.number("C")*x.size(0));
    loss.backward();
    auto independent_gradient=std::max(w.grad().abs().max().item<double>(),b.grad().abs().max().item<double>());
    ni26::require(std::abs(loss.item<double>()-fit.objective)<=1e-9 && std::abs(independent_gradient-fit.max_gradient)<=1e-9,"native independent objective/gradient mismatch");
    ni26::require(fit.objective<=fixture.number("reference_objective")+1e-5 && fit.max_gradient<=1e-5,"native fit missed common quality");
    double maximum=0;
    for (const std::string name:{"source","within0","full0"}) {
      auto input=name=="source"?x:fixture.get(name+"_x");
      auto actual=torch::softmax(torch::matmul(input,fit.weights.t())+fit.bias,1);
      auto expected=fixture.get(name+"_probabilities");
      auto error=compare(actual,expected,1e-3,"native "+name+" probabilities");
      ni26::require(torch::equal(actual.argmax(1),expected.argmax(1)),"native "+name+" changed labels");
      maximum=std::max(maximum,error);
    }
    ni26::Params output;
    output.tensors={{"native_format",torch::tensor(int64_t(1))},{"weights",fit.weights},{"bias",fit.bias},
      {"nit",torch::tensor(fit.nit)},{"nfev",torch::tensor(fit.nfev)},
      {"objective",torch::tensor(fit.objective,torch::kFloat64)},
      {"max_gradient",torch::tensor(fit.max_gradient,torch::kFloat64)},
      {"max_probability_error",torch::tensor(maximum,torch::kFloat64)},
      {"seconds",torch::tensor(seconds,torch::kFloat64)},
      {"scaler_mean_error",torch::tensor(mean_error,torch::kFloat64)},
      {"scaler_scale_error",torch::tensor(scale_error,torch::kFloat64)},
      {"scaler_transform_error",torch::tensor(transform_error,torch::kFloat64)}};
    output.save(argv[2]);
    std::cout<<"NATIVE_LINEAR_SOURCE_PASS n="<<x.size(0)<<" d="<<x.size(1)
      <<" nit="<<fit.nit<<" nfev="<<fit.nfev<<" seconds="<<seconds
      <<" maxprob="<<maximum<<" BLAS="<<config<<'\n';
    return 0;
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
