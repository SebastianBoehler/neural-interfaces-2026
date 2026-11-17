#include "training_internal.h"
#include "ni26/math.h"
#include <torch/torch.h>
#include <cmath>
#include <set>

namespace ni26::neural {
std::pair<double,double> one_cycle(int64_t step,int64_t total) {
  require(total>1 &&step>=0 &&step<total,"invalid native OneCycle step");
  double first=.3*total-1,pct,start_lr,end_lr,start_beta,end_beta;
  if(step<=first){pct=step/first;start_lr=.001/25;end_lr=.001;start_beta=.95;end_beta=.85;}
  else{pct=(step-first)/(total-1-first);start_lr=.001;end_lr=.001/25/10000;start_beta=.85;end_beta=.95;}
  auto cosine=[&](double a,double b){return b+(a-b)/2*(std::cos(M_PI*pct)+1);};
  return {cosine(start_lr,end_lr),cosine(start_beta,end_beta)};
}
}
namespace ni26 {
Params train_raw_fbc(const Params& fixture,torch::Device device) {
  torch::AutoGradMode enable_grad(true);
  require(fixture.integer("source_only")==1 &&fixture.integer("epochs")==40,"native source fit requires registered source-only40-epoch fixture");
  auto people=fixture.texts("source_people"),excluded=fixture.texts("excluded_people");
  require(std::set<std::string>(people.begin(),people.end())==std::set<std::string>({"P04","P05","P06","P07","P08","P10"}),"native source6 identities differ");
  require(std::set<std::string>(excluded.begin(),excluded.end())==std::set<std::string>({"P02","P03","P09"}),"native exposed query exclusion differs");
  auto x=fixture.get("source_X"),y=fixture.get("source_y"),session=fixture.get("source_session");
  require(x.device().is_cpu() &&x.scalar_type()==torch::kFloat32 &&x.sizes()==torch::IntArrayRef({1898,41,480}),"native source official tensor differs");finite(x,"native source input");
  require(y.scalar_type()==torch::kInt64 &&y.numel()==1898 &&y.min().item<int64_t>()>=0 &&y.max().item<int64_t>()<3,"native source labels differ");
  require(session.numel()==1898,"native source session map differs");
  auto batches=fixture.get("batch_indices").to(torch::kCPU,torch::kInt64),sizes=fixture.get("batch_sizes").to(torch::kCPU,torch::kInt64);
  require(batches.sizes()==torch::IntArrayRef({1240,64}) &&sizes.numel()==1240,"native exact40×31 source batch plan differs");
  neural::RawFBC net(fixture.subset("initial_"),device);auto parameters=net.parameters();
  torch::optim::AdamW optimizer(parameters,torch::optim::AdamWOptions(.001).weight_decay(.001));
  std::vector<Tensor> losses;double gradient_max=0,parameter_change=0;
  auto initial=parameters.back().detach().clone();
  for(int64_t step=0;step<1240;++step){auto count=sizes[step].item<int64_t>();require(count==32 ||count==64,"native source padded batch size differs");
    auto ids=batches[step].slice(0,0,count);require(ids.min().item<int64_t>()>=0 &&ids.max().item<int64_t>()<1898,"native source batch row outside fit");
    auto sessions=session.index_select(0,ids).reshape({-1,32});require((sessions==sessions.select(1,0).unsqueeze(1)).all().item<bool>(),"native source32-row chunk crosses sessions");
    auto [lr,beta]=neural::one_cycle(step,1240);auto& options=static_cast<torch::optim::AdamWOptions&>(optimizer.param_groups()[0].options());options.lr(lr);options.betas(std::make_tuple(beta,.999));
    optimizer.zero_grad();auto logits=net.forward(x.index_select(0,ids).to(device),true);auto loss=torch::nn::functional::cross_entropy(logits,y.index_select(0,ids).to(device));finite(loss,"native source loss");loss.backward();
    require(parameters.front().grad().defined() &&parameters.back().grad().defined(),"native source body/head gradient inactive");
    gradient_max=std::max(gradient_max,parameters.front().grad().abs().max().item<double>());optimizer.step();losses.push_back(loss.detach().cpu());
  }
  parameter_change=(parameters.back().detach()-initial).abs().max().item<double>();require(gradient_max>0 &&parameter_change>0,"native source optimizer did not update body/head");
  auto result=net.state();result.tensors["native_format"]=torch::tensor(1,torch::kInt64);result.tensors["losses"]=torch::stack(losses);result.tensors["steps"]=torch::tensor(1240,torch::kInt64);
  result.tensors["body_gradient_max"]=torch::tensor(gradient_max,torch::kFloat64);result.tensors["head_bias_change_max"]=torch::tensor(parameter_change,torch::kFloat64);return result;
}
}
