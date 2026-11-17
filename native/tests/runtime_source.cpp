#include "ni26/runtime.h"
#include "ni26/math.h"
#include <torch/torch.h>
#include <iostream>

int main(int argc,char** argv) {
  try {
    ni26::require(argc==5,"usage: ni26_runtime_source PACKAGE FIXTURE DEVICE OUTPUT");
    torch::set_num_threads(1);
    auto root=std::filesystem::path(argv[1]);
    auto z=ni26::Params::load(root/"native.pt"),fixture=ni26::Params::load(argv[2]);
    ni26::Runtime model(z,torch::Device(argv[3]),root);
    auto x=fixture.get("x");
    std::vector<ni26::Tensor> scores;
    for(int64_t i=0;i<x.size(0);++i) scores.push_back(model.step(x[i]));
    auto result=torch::stack(scores);
    auto expected=fixture.get("runtime_scores");
    ni26::finite(result,"runtime source outputs");
    double maximum=(result-expected).abs().max().item<double>();
    ni26::require(maximum<=1e-6 && torch::equal(result.argmax(1),expected.argmax(1)),
                  "full native runtime source parity failed maxabs="+std::to_string(maximum));
    auto info=model.stats();
    ni26::require(info.count_filter.errors==0 && info.personalization.errors==0,
                  "native optional stage bypassed on source fixture");
    model.reset();
    for(int64_t i=0;i<std::min<int64_t>(17,x.size(0));++i)
      ni26::require(torch::equal(model.step(x[i]),scores[i]),"native runtime reset prefix changed");
    ni26::Params out;
    out.tensors={{"native_format",torch::tensor(int64_t(1))},{"scores",result},
      {"max_error",torch::tensor(maximum,torch::kFloat64)},
      {"windows",torch::tensor(info.windows)},{"recognized_at",torch::tensor(info.recognized_at)},
      {"fine_tune_refits",torch::tensor(info.fine_tune.refits)},
      {"personalization_refits",torch::tensor(info.personalization.refits)}};
    out.save(argv[4]);
    std::cout<<"NATIVE_RUNTIME_SOURCE_PASS windows="<<info.windows<<" maxabs="<<maximum<<'\n';
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
