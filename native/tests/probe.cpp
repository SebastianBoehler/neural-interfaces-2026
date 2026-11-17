#include "ni26/core.h"
#include "ni26/tail.h"
#include <torch/torch.h>
#include <iostream>

int main(int argc,char** argv) {
  try {
    ni26::require(argc==4,"usage: ni26_probe MODEL.pt SOURCE_FIXTURE.pt OUTPUT.pt");
    torch::set_num_threads(1);
    torch::NoGradGuard no_grad;
    auto z=ni26::Params::load(argv[1]), fixture=ni26::Params::load(argv[2]);
    auto x=fixture.get("x");
    ni26::require(x.dim()==3 && x.size(0)>=80,"real source fixture must activate session stages");
    ni26::TangentCore first(z.subset("ts_")),second(z.subset("ts2_"));
    ni26::Tail tail(z);
    std::vector<ni26::Tensor> p1,p2,features,emitted;
    for (int64_t i=0;i<x.size(0);++i) {
      auto [p,f]=first.step(x[i]);
      auto [q,unused]=second.step(x[i]);
      auto u=torch::matmul(f-z.get("tsem_m"),z.get("tsem_V").to(torch::kFloat64));
      p1.push_back(p);p2.push_back(q);features.push_back(f);
      emitted.push_back(tail.step(x[i],u,fixture.get("mix")[i]));
    }
    ni26::Params out;
    out.tensors={{"native_format",torch::tensor(int64_t(1))},{"p1",torch::stack(p1)},
      {"p2",torch::stack(p2)},{"features",torch::stack(features)},
      {"tail",torch::stack(emitted)},{"reapply",tail.reapply(x.size(0)-1)}};
    out.save(argv[3]);
    std::cout<<"native source component run complete: "<<x.size(0)<<" windows\n";
    return 0;
  } catch (const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
