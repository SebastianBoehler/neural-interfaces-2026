#include "ni26/routing.h"
#include <torch/torch.h>
namespace ni26 {
using namespace torch::indexing;
ContrastGate::ContrastGate(const Params& z):basis_(z.get("route_Q")),tau_(z.number("route_tau")),width_(z.number("route_ramp")),rho_(z.number("route_rho")),clip_(z.number("route_clip")),start_(z.integer("route_t_start")) {
 auto g=z.get("route_gal").flatten(1); auto a=g-torch::matmul(torch::matmul(g,basis_),basis_.t()); gallery_=a/a.norm(2,1,true);
}
Decision ContrastGate::step(const std::vector<Tensor>& U,const std::vector<Tensor>& Q) const {
 int64_t n=U.size(); if(n<start_)return {};
 auto X=torch::stack(U),P=torch::stack(Q); auto lab=P.argmax(1),confidence=std::get<0>(P.max(1)); auto weights=torch::zeros({n,3},X.options());
 int64_t q=std::max<int64_t>(2,static_cast<int64_t>(rho_*n/3));
 for(int64_t k=0;k<3;k++){auto ids=torch::nonzero(lab==k).flatten();auto order=torch::argsort(-confidence.index({ids}));ids=ids.index({order.index({Slice(0,q)})});weights.index_put_({ids,k},1);}
 auto counts=weights.sum(0);if((counts<1e-9).any().item<bool>())return {};
 auto med=ni26::median(X,0),mad=1.4826*ni26::median((X-med).abs(),0);auto Z=((X-med)/mad.clamp_min(1e-6)).clamp(-clip_,clip_);
 auto mu=torch::matmul(weights.t(),Z)/counts.unsqueeze(1);auto a=(mu-mu.mean(0)).flatten();a=a-torch::matmul(basis_,torch::matmul(basis_.t(),a));
 double norm=a.norm().item<double>();auto scores=norm>1e-9?torch::matmul(gallery_,a)/norm:torch::zeros({gallery_.size(0)},X.options());
 if(!torch::isfinite(scores).all().item<bool>())return {};int64_t j=scores.argmax().item<int64_t>();double score=scores[j].item<double>();return GateDecision{j,score,ramp(score,tau_,width_)};
}
IdentityGate::IdentityGate(const Params& z):z_(z){reset();}
IdentityGate::IdentityGate(const IdentityGate& o):z_(o.z_),sum_(o.sum_.defined()?o.sum_.clone():Tensor()),n_(o.n_){}
IdentityGate& IdentityGate::operator=(const IdentityGate& o){if(this!=&o){z_=o.z_;sum_=o.sum_.defined()?o.sum_.clone():Tensor();n_=o.n_;}return *this;}
void IdentityGate::reset(){sum_=Tensor();n_=0;}
Decision IdentityGate::step(const Tensor& raw){
 auto x=raw.to(torch::kFloat64);x=x-x.mean(0,true);x=x-x.mean(1,true);std::vector<Tensor> segments;
 auto win=torch::hann_window(120,false,x.options());for(int s=0;s<=360;s+=60)segments.push_back(x.index({Slice(),Slice(s,s+120)})*win);
 auto fft=torch::fft::rfft(torch::stack(segments,1),{},-1);auto L=(fft.abs().square().mean(1).index({Slice(),Slice(2,41)})+1e-12).log();auto f=(L-L.mean(1,true)).flatten();sum_=sum_.defined()?sum_+f:f.clone();n_++;
 if(n_-1<z_.integer("bio_t0"))return {};
 auto V=z_.get("bio_V"),B=z_.get("bio_B"),W=z_.get("bio_W"),S=z_.get("bio_S"),G=z_.get("bio_gal"),m=z_.get("bio_m");
 auto u=torch::matmul((sum_/n_-z_.get("bio_mu"))/z_.get("bio_sd"),V);auto C=B+W+S/n_;auto xc=torch::dot(u,torch::linalg_solve(C,u));auto ldc=std::get<1>(torch::linalg_slogdet(C));std::vector<Tensor> scores;
 for(int64_t j=0;j<G.size(0);j++){auto A=B+W/m[j].item<double>();auto same=torch::cat({torch::cat({A,B},1),torch::cat({B,C},1)},0);auto v=torch::cat({G[j],u});auto quad=torch::dot(v,torch::linalg_solve(same,v));auto gag=torch::dot(G[j],torch::linalg_solve(A,G[j]));scores.push_back((-.5*(quad+std::get<1>(torch::linalg_slogdet(same)))+.5*(gag+xc+std::get<1>(torch::linalg_slogdet(A))+ldc))/V.size(1));}
 auto scores_t=torch::stack(scores);if(!torch::isfinite(scores_t).all().item<bool>())return {};int64_t j=scores_t.argmax().item<int64_t>();double score=scores_t[j].item<double>();return GateDecision{j,score,ramp(score,z_.number("bio_tau"),z_.number("bio_ramp"))};
}
}
