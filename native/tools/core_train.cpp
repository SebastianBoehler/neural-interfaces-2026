#include "ni26/core_training.h"
#include <torch/torch.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc,char** argv) {
  try {
    ni26::require(argc==4,"usage: ni26_core_train RAW_SOURCE.pt IMMUTABLE_RECIPES.pt OUTPUT.pt");
    ni26::require(!std::filesystem::exists(argv[3]),"native core output already exists");
    auto source=ni26::Params::load(argv[1]),recipes=ni26::Params::load(argv[2]);
    for (const auto& [key,value]:recipes.tensors)
      ni26::require(key=="native_format" || key.rfind("ts_",0)==0 || key.rfind("ts2_",0)==0,
                    "unsupported core recipe archive field: "+key);
    ni26::Params output;output.tensors.emplace("native_format",torch::tensor(int64_t(1)));
    for (const std::string prefix:{"ts_","ts2_"}) {
      auto recipe=recipes.subset(prefix);
      ni26::require(recipe.integer("hilbert")==int(prefix=="ts2_"),"core recipe prefix/Hilbert mismatch");
      auto tick=std::chrono::steady_clock::now();
      auto result=ni26::train_classical_core(source,recipe);
      for (const auto& [key,value]:result.model.tensors) output.tensors.emplace(prefix+key,value);
      std::cout<<prefix<<" NATIVE_CORE_TRAINED source_rows="<<result.source_rows.numel()
        <<" torch_threads="<<torch::get_num_threads()<<" width="<<result.source_features.size(1)<<" C="<<result.model.number("training_C")
        <<" gtol="<<result.model.number("training_gtol")<<" nit="<<result.head.nit
        <<" nfev="<<result.head.nfev<<" seconds="
        <<std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count()<<'\n';
    }
    output.tensors.emplace("training_held_person",source.get("held_person").clone());
    ni26::require(!std::filesystem::exists(argv[3]),"native core output appeared during training");
    output.save(argv[3]);
    return 0;
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
