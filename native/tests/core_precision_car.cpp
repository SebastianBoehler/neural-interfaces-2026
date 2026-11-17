#include "ni26/math.h"
#include <torch/torch.h>
#include <iostream>

int main(int argc,char** argv) {
  try {
    ni26::require(argc==3,"usage: core_precision_car SOURCE NUMPY_CAR_ORACLE");
    torch::set_num_threads(1);torch::NoGradGuard guard;
    auto source=ni26::Params::load(argv[1]),oracle=ni26::Params::load(argv[2]);
    auto raw=source.get("x"),expected=oracle.get("car");
    ni26::require(raw.size(0)==96 && raw.sizes()==expected.sizes(),"expected all96 original source windows");
    for (int64_t i=0;i<raw.size(0);++i)
      ni26::require(torch::equal(ni26::common_average_reference_float32(raw[i]),expected[i]),
                    "native CAR is not NumPy-bitwise at source row "+std::to_string(i));
    std::cout<<"ALL96_NUMPY_CAR_BITWISE_PASS values="<<raw.numel()<<'\n';return 0;
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
