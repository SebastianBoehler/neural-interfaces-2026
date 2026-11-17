#include "ni26/runtime.h"
#include "ni26/math.h"
#include <torch/torch.h>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv) {
  try {
    ni26::require(argc==5,"usage: ni26_runtime_source PACKAGE FIXTURE DEVICE OUTPUT");
    torch::set_num_threads(1);
    auto root=std::filesystem::path(argv[1]);
    auto z=ni26::Params::load(root/"native.pt"),fixture=ni26::Params::load(argv[2]);
    ni26::Runtime model(z,torch::Device(argv[3]),root);
    auto x=fixture.get("x"),expected=fixture.get("scores");
    ni26::require(x.dim()==3 && x.size(0)>0 && x.size(1)==41 && x.size(2)==480 &&
                  expected.sizes()==torch::IntArrayRef({x.size(0),3}),
                  "invalid natural-gate source fixture");
    std::vector<ni26::Tensor> scores;
    ni26::Runtime second(model);
    for(int64_t i=0;i<x.size(0);++i) {
      scores.push_back(model.step(x[i]));
      ni26::require(torch::equal(second.step(x[i]),scores.back()),
                    "interleaved copies changed source output");
    }
    auto result=torch::stack(scores);
    ni26::finite(result,"runtime source outputs");
    double maximum=(result-expected).abs().max().item<double>();
    auto changed_labels=result.argmax(1).ne(expected.argmax(1)).sum().item<int64_t>();
    auto info=model.stats();
    const auto counters=torch::tensor({info.windows,info.recognized_at,info.fine_tune.refits,
        info.fine_tune.steps,info.fine_tune.relabels,info.personalization.refits},torch::kInt64);
    const auto expected_counters=fixture.get("counters");
    model.reset();
    for(int64_t i=0;i<std::min<int64_t>(17,x.size(0));++i)
      ni26::require(torch::equal(model.step(x[i]),scores[i]),"native runtime reset prefix changed");
    ni26::Params out;
    out.tensors={{"native_format",torch::tensor(int64_t(1))},{"scores",result},
      {"max_error",torch::tensor(maximum,torch::kFloat64)},
      {"windows",torch::tensor(info.windows)},{"recognized_at",torch::tensor(info.recognized_at)},
      {"counters",counters}};
    out.save(argv[4]);
    std::cout<<std::setprecision(17)<<"source windows="<<info.windows
             <<" maxabs="<<maximum<<" changed_labels="<<changed_labels<<std::endl;
    std::cout<<"native counters "<<counters<<"reference counters "<<expected_counters<<std::endl;
    ni26::require(info.count_filter.errors==0 && info.personalization.errors==0,
                  "native optional stage bypassed on source fixture");
    ni26::require(torch::equal(counters,expected_counters),"native runtime activity differs");
    ni26::require(maximum<=1e-6 && changed_labels==0,"full native runtime source parity failed");
    std::cout<<"NATIVE_RUNTIME_SOURCE_PASS windows="<<info.windows<<" maxabs="<<maximum<<'\n';
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
