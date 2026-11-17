// Prepared real-source training comparison. No accuracy or held-label access.
#include "ni26/core_training.h"
#include <torch/torch.h>
#include <chrono>
#include <filesystem>
#include <iostream>

namespace {
double compare(const ni26::Tensor& actual,const ni26::Tensor& expected,const char* name,double bound) {
  ni26::require(actual.sizes()==expected.sizes(),std::string(name)+" shape differs");
  ni26::finite(actual,name);
  auto error=(actual-expected).abs().max().item<double>();
  std::cout<<name<<" max_abs="<<error<<" limit="<<bound<<'\n';
  ni26::require(error<=bound,std::string(name)+" parity failed");
  return error;
}
void exclusion_guards(const ni26::Params& source,const ni26::Params& recipe) {
  auto rows=torch::nonzero(source.get("fit_mask")).flatten();
  auto row=rows[0].item<int64_t>();
  auto bad=source;bad.tensors["fit_mask"]=source.get("fit_mask").clone();
  bad.tensors["fit_mask"][row]=false;
  bool rejected=false;
  try {ni26::train_classical_core(bad,recipe);} catch(const std::exception& error) {
    rejected=std::string(error.what()).find("partial-session")!=std::string::npos;
  }
  ni26::require(rejected,"partial-session fit was not rejected before preprocessing");
  bad=source;bad.tensors["held_person"]=source.get("person")[row].reshape({1}).clone();
  rejected=false;
  try {ni26::train_classical_core(bad,recipe);} catch(const std::exception& error) {
    rejected=std::string(error.what()).find("explicitly held person")!=std::string::npos;
  }
  ni26::require(rejected,"selected held person was not rejected before preprocessing");
}
}
int main(int argc,char** argv) {
  try {
    ni26::require(argc==5,"usage: core_training_source SOURCE.pt RECIPE.pt REFERENCE.pt OUTPUT.pt");
    ni26::require(!std::filesystem::exists(argv[4]),"new source-training output required");
    torch::set_num_threads(2);torch::NoGradGuard guard;
    auto source=ni26::Params::load(argv[1]),recipe=ni26::Params::load(argv[2]);
    recipe.tensors.erase("native_format");
    auto reference=ni26::Params::load(argv[3]);
    ni26::require(recipe.integer("hilbert")==0,"registered first source test is REAL core only");
    exclusion_guards(source,recipe);
    auto started=std::chrono::steady_clock::now();
    auto fitted=ni26::train_classical_core(source,recipe);
    ni26::require(torch::equal(fitted.source_rows,reference.get("rows")),"source row order differs");
    auto delta=fitted.source_features-reference.get("features");
    auto relative=delta.norm().item<double>()/reference.get("features").norm().item<double>();
    std::cout.precision(12);
    std::cout<<"features relative_L2="<<relative<<" limit=1e-6\n";
    ni26::require(relative<=1e-6,"source feature relative error failed");
    compare(fitted.source_features,reference.get("features"),"features",1e-4);
    compare(fitted.model.get("log_glob"),reference.get("log_glob"),"log_glob",1e-5);
    compare(fitted.model.get("mu"),reference.get("mu"),"mu",1e-4);
    compare(fitted.model.get("sd"),reference.get("sd"),"sd",1e-4);
    auto z=(fitted.source_features-fitted.model.get("mu"))/fitted.model.get("sd");
    auto logits=torch::matmul(z,fitted.head.weights.t())+fitted.head.bias;
    auto probabilities=torch::softmax(logits,1);
    compare(probabilities,reference.get("probabilities"),"source_probability",1e-3);
    auto changed=(probabilities.argmax(1)!=reference.get("probabilities").argmax(1)).sum().item<int64_t>();
    ni26::require(changed==0,"source predicted labels changed");
    auto y=source.get("y").index_select(0,fitted.source_rows);
    auto loss=-torch::log_softmax(logits,1).gather(1,y.unsqueeze(1)).mean()+
      fitted.head.weights.square().sum()/(2*10.*y.numel());
    ni26::require(std::abs(loss.item<double>()-fitted.head.objective)<=1e-10,"independent final objective differs");
    fitted.model.tensors["native_format"]=torch::tensor(int64_t(1));
    fitted.model.save(argv[4]);
    std::cout<<"OK native REAL source fit rows="<<y.numel()<<" changed_labels="<<changed
      <<" nit="<<fitted.head.nit<<" nfev="<<fitted.head.nfev
      <<" objective="<<fitted.head.objective<<" max_gradient="<<fitted.head.max_gradient
      <<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<'\n';
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
