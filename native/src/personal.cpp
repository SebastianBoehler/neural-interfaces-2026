#include "ni26/routing.h"
#include <torch/torch.h>
namespace ni26 {
using namespace torch::indexing;
OwnStage::OwnStage(const Params& z):z_(z),sos_(filters(z,"own_sos")),slow_(z.get("own_slow_sos")){reset();}
OwnStage::OwnStage(const OwnStage& o):z_(o.z_),sos_(o.sos_),slow_(o.slow_),n_(o.n_){for(auto pair:{std::pair<Tensor*,const Tensor*>{&csum_,&o.csum_},{&s1_,&o.s1_},{&s2_,&o.s2_},{&current_c_,&o.current_c_},{&current_s_,&o.current_s_}})*pair.first=pair.second->defined()?pair.second->clone():Tensor();}
OwnStage& OwnStage::operator=(const OwnStage& o){if(this!=&o){OwnStage tmp(o);std::swap(z_,tmp.z_);std::swap(sos_,tmp.sos_);std::swap(slow_,tmp.slow_);std::swap(csum_,tmp.csum_);std::swap(s1_,tmp.s1_);std::swap(s2_,tmp.s2_);std::swap(current_c_,tmp.current_c_);std::swap(current_s_,tmp.current_s_);std::swap(n_,tmp.n_);}return *this;}
void OwnStage::reset(){n_=0;csum_=s1_=s2_=current_c_=current_s_=Tensor();}
bool OwnStage::observe(const Tensor& raw){
 auto x=raw.to(torch::kFloat64);current_c_=current_s_=Tensor();auto sd=x.std(1,false);
 if(!torch::isfinite(x).all().item<bool>() || sd.min().item<double>()<=1e-9*std::max(sd.max().item<double>(),1e-30))return false;
 auto centered=x-x.mean(0,true);std::vector<Tensor> covs;
 for(const auto& sos:sos_){auto b=sos(centered);auto C=torch::matmul(b,b.t())/b.size(-1);C=C+1e-3*C.trace()*torch::eye(C.size(0),C.options())/C.size(0);covs.push_back(C);}
 current_c_=torch::stack(covs);current_s_=slow_(x).index({Slice(),Slice(7,torch::indexing::None,15)}).flatten();
 csum_=csum_.defined()?csum_+current_c_:current_c_.clone();s1_=s1_.defined()?s1_+current_s_:current_s_.clone();s2_=s2_.defined()?s2_+current_s_.square():current_s_.square();n_++;return true;
}
std::optional<std::pair<Tensor,Tensor>> OwnStage::apply(const Tensor& out,int64_t j,double r)const{
 if(!current_c_.defined())return {};
 auto reference=(z_.number("own_kref")*z_.get("own_prior")[j]+csum_)/(z_.number("own_kref")+n_);auto W=spectral(reference,SpectralFunction::InverseRoot,0);auto C=torch::matmul(torch::matmul(W,current_c_),W);
 auto tangent=upper_features(spectral(C,SpectralFunction::Log,0)).flatten();auto pt=posterior(torch::matmul((tangent-z_.get("own_t_mu")[j])/z_.get("own_t_sd")[j],z_.get("own_t_W")[j].t())+z_.get("own_t_c")[j]);
 auto csp=z_.get("own_csp")[j];auto features=(csp*torch::matmul(C,csp)).sum(1).log().flatten();auto pc=posterior(torch::matmul(features,z_.get("own_c_W")[j].t())+z_.get("own_c_c")[j]);
 auto mean=s1_/n_;auto slow=n_>=4?(current_s_-mean)/(s2_/n_-mean.square()).clamp_min(1e-9).sqrt():torch::zeros_like(current_s_);auto ps=posterior(torch::matmul(slow,z_.get("own_s_W")[j].t())+z_.get("own_s_c")[j]);
 auto lm=(pt.clamp(1e-3,1).log()+pc.clamp(1e-3,1).log()+ps.clamp(1e-3,1).log())/3;auto result=posterior(out.clamp(1e-9,1).log()+z_.number("own_beta")*r*lm);
 if(!torch::isfinite(lm).all().item<bool>()||!torch::isfinite(result).all().item<bool>())return {};return std::make_pair(result,lm);
}
}
