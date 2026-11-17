#include "ni26/networks.h"
#include "ni26/routing.h"
#include "ni26/math.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

using namespace ni26;
namespace {
std::map<std::string,double> errors;
void same(const std::string& key,const Tensor& actual,const Tensor& expected,double tolerance=1e-6){
  require(actual.sizes()==expected.sizes(),"shape differs: "+key);finite(actual,key);
  auto error=(actual.to(torch::kCPU,torch::kFloat64)-expected.to(torch::kCPU,torch::kFloat64)).abs().max().item<double>();
  errors[key]=std::max(errors[key],error);require(error<=tolerance,key+" maxabs="+std::to_string(error));
}
Decision decision(const Tensor& x){if(x[0].item<int64_t>()<0)return {};return GateDecision{x[0].item<int64_t>(),x[1].item<double>(),x[2].item<double>()};}
}
int main(int argc,char** argv){
  try {
    require(argc==4,"usage: neural_source PARAMS FIXTURE MODULE_ROOT");torch::set_num_threads(1);
    auto z=Params::load(argv[1]),f=Params::load(argv[2]);auto weights=z.get("member_weights").to(torch::kFloat64);weights=weights/weights.sum();
    const std::filesystem::path root=argv[3];torch::Device device(torch::kCPU);
    for(const std::string mode:{"known","unknown"}){
      NetBank bank(z,device,root);EyeNet eye(z,device,root);RestMember rest(z,device,root);FineTune ft(z,weights,device,root);
      Route outer(z,weights[0].item<double>()*(1-z.number("v6_rest_w")),true);
      if(z.integer("v6_tpri_on"))outer.set_prior(z.number("v6_tpri_kappa"),z.get("v6_tpri"));
      auto x=f.get(mode+"_x"),input_ids=f.get("input_ids");auto names=z.texts("v6_slots");
      for(int64_t i=0;i<x.size(0);++i){torch::NoGradGuard runtime_guard;
        auto read=[&](const std::string& key){return f.get(mode+"_"+key)[i];};
        auto step=bank.step(x[i]);same(mode+"/ids",torch::tensor(step.ids,torch::kInt64),read("ids"),0);same(mode+"/members",step.probabilities,read("probs"));
        for(int64_t j=0;j<input_ids.numel();++j)same(mode+"/input",step.inputs.at(input_ids[j].item<int64_t>()),read("input")[j]);
        same(mode+"/eye",eye.step(x[i]),read("eye"));same(mode+"/rest",rest.step(x[i]),read("rest"));
        RelabelFn relabel=[&](int64_t count)->std::optional<Tensor>{auto key=mode+"_relabel_"+std::to_string(count);return f.has(key)?std::optional<Tensor>(f.get(key)):std::nullopt;};
        auto tuned=ft.step(step,read("label"),relabel);auto active=read("ft_active");
        for(size_t j=0;j<names.size();++j){require(bool(tuned.count(names[j]))==bool(active[j].item<int64_t>()),"fine-tuning activity differs");if(tuned.count(names[j]))same(mode+"/fine_tune/"+names[j],tuned.at(names[j]),read("ft")[j]);}
        auto stats=ft.stats();auto expected=read("ft_stats");std::vector<int64_t> counts{stats.refits,stats.steps,stats.relabels,stats.capped,stats.over};
        same(mode+"/ft_counters",torch::tensor(counts,torch::kInt64),expected.slice(0,0,5),0);
        same(mode+"/ft_change",torch::tensor(stats.maximum_parameter_change,torch::kFloat64),expected[5]);
        auto out=outer.step_given(x[i],read("u"),read("mix6"),read("core1"),read("eye"),decision(read("contrast")),decision(read("identity")));
        same(mode+"/outer",out,read("outer"),1e-8);
        if(i==4){NetBank a=bank,b=bank;auto pa=a.step(x[i+1]),pb=b.step(x[i+1]);same(mode+"/copy_bank",pa.probabilities,pb.probabilities,0);
          EyeNet e=eye;same(mode+"/copy_eye_reset",e.step(x[0]),f.get(mode+"_eye")[0]);
          RestMember ra=rest,rb=rest;same(mode+"/copy_rest",ra.step(x[i+1]),rb.step(x[i+1]),0);
          FineTune copy=ft;require(copy.stats().windows==0 &&copy.stats().refits==0,"fine-tuning session copy retained state");}
      }
      if(mode=="unknown")require(ft.stats().refits==2 &&ft.stats().steps==40 &&ft.stats().maximum_parameter_change>0,"native two-refit path inactive");
      bank.reset();eye.reset();rest.reset();ft.reset();outer.reset();
      require(ft.stats().windows==0 &&ft.stats().steps==0,"native fine-tuning reset retained state");
      for(int64_t i=0;i<8;++i){same(mode+"/reset_bank",bank.step(x[i]).probabilities,f.get(mode+"_probs")[i]);same(mode+"/reset_eye",eye.step(x[i]),f.get(mode+"_eye")[i]);same(mode+"/reset_rest",rest.step(x[i]),f.get(mode+"_rest")[i]);}
    }
    for(const auto& [key,error]:errors)std::cout<<key<<" maxabs="<<std::setprecision(17)<<error<<'\n';
    std::cout<<"NEURAL_SOURCE_PASS\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
