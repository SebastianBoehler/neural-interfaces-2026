#pragma once
#include "ni26/tail.h"
#include <optional>
#include <memory>
namespace ni26 {
struct GateDecision { int64_t person; double score,weight; };
using Decision=std::optional<GateDecision>;
class ContrastGate {
 public: explicit ContrastGate(const Params& z); Decision step(const std::vector<Tensor>& U,const std::vector<Tensor>& Q) const;
 private: Tensor basis_,gallery_; double tau_,width_,rho_,clip_; int64_t start_;
};
class IdentityGate {
 public: explicit IdentityGate(const Params& z); IdentityGate(const IdentityGate&); IdentityGate& operator=(const IdentityGate&); void reset(); Decision step(const Tensor& x);
 private: Params z_; Tensor sum_; int64_t n_=0;
};
class OwnStage {
 public: explicit OwnStage(const Params& z); OwnStage(const OwnStage&); OwnStage& operator=(const OwnStage&); void reset(); bool observe(const Tensor& x); std::optional<std::pair<Tensor,Tensor>> apply(const Tensor& out,int64_t person,double weight) const;
 private: Params z_; std::vector<SosFilter> sos_; SosFilter slow_; Tensor csum_,s1_,s2_,current_c_,current_s_; int64_t n_=0;
};
struct RouteStats { int64_t windows=0,routed=0,degenerate=0,staged=0,errors=0; };
class Route {
 public:
 explicit Route(const Params& z,double core1_share,bool outer=false); ~Route(); Route(const Route&); Route& operator=(const Route&);
 void reset(); Tensor step(const Tensor& x,const Tensor& u,const Tensor& mix,const Tensor& core1,const Tensor& shadow);
 Tensor step_given(const Tensor& x,const Tensor& u,const Tensor& mix,const Tensor& core1,const Tensor& extra,Decision contrast,Decision identity);
 Decision last_contrast() const; Decision last_identity() const; Tensor last_stage() const; RouteStats stats() const;
 void set_core1_share(double value); void set_prior(double kappa,const Tensor& means);
 private: struct Impl; std::unique_ptr<Impl> impl_;
};
}
