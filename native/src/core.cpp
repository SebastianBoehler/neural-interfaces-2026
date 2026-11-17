#include "ni26/core.h"
#include <torch/torch.h>

namespace ni26 {
using torch::indexing::Slice;
TangentCore::TangentCore(const Params& z)
    : z_(z), bank_(filters(z,"sos")), slow_(z.get("slow_sos")) { reset(); }
TangentCore::TangentCore(const TangentCore& other)
    : z_(other.z_),bank_(other.bank_),slow_(other.slow_),
      log_sum_(other.log_sum_.clone()),slow_sum_(other.slow_sum_.clone()),count_(other.count_) {}
TangentCore& TangentCore::operator=(const TangentCore& other) {
  if (this != &other) {
    z_=other.z_;bank_=other.bank_;slow_=other.slow_;
    log_sum_=other.log_sum_.clone();slow_sum_=other.slow_sum_.clone();count_=other.count_;
  }
  return *this;
}
void TangentCore::reset() {
  log_sum_=torch::zeros_like(z_.get("log_glob"));slow_sum_=Tensor{};count_=0;
}
std::pair<Tensor,Tensor> TangentCore::step(const Tensor& raw) {
  require(raw.sizes()==torch::IntArrayRef({41,480}) && raw.scalar_type()==torch::kFloat32,
          "tangent input must be float32(41,480)");
  finite(raw,"tangent input");
  auto x=raw-raw.mean(0,true);
  std::vector<Tensor> bands;
  for (const auto& filter : bank_) bands.push_back(filter(x).to(torch::kFloat32));
  auto b=torch::stack(bands);
  const bool analytic=z_.has("hilbert") && z_.number("hilbert")!=0;
  if (analytic) {
    auto multiplier=torch::zeros({480},b.options());
    multiplier.index_put_({0},1.);multiplier.index_put_({240},1.);
    multiplier.slice(0,1,240).fill_(2.);
    b=torch::fft_ifft(torch::fft_fft(b,480,-1)*multiplier,480,-1);
  }
  std::vector<Tensor> segments{b};
  auto pieces=torch::tensor_split(b,z_.integer("nsub"),-1);
  segments.insert(segments.end(),pieces.begin(),pieces.end());
  std::vector<Tensor> covariances;
  double shrink=z_.number("shrink");
  for (const auto& segment : segments) {
    auto centered=(segment-segment.mean(-1,true)).to(analytic?torch::kComplexDouble:torch::kFloat64);
    auto covariance=torch::matmul(centered,centered.transpose(-2,-1).conj())/segment.size(-1);
    auto trace=torch::real(covariance.diagonal(0,-2,-1).sum(-1)).reshape({-1,1,1})/41.;
    covariances.push_back((1-shrink)*covariance+shrink*trace*torch::eye(41,covariance.options()));
  }
  auto cov=torch::stack(covariances,1).reshape({-1,41,41});
  log_sum_=log_sum_+spectral(cov,SpectralFunction::Log);++count_;
  double prior=z_.number("k");
  auto reference=(prior*z_.get("log_glob")+log_sum_)/(prior+count_);
  auto whitening=spectral(reference,SpectralFunction::NegativeHalfExp);
  auto tangent=spectral(torch::matmul(torch::matmul(whitening,cov),whitening),SpectralFunction::Log);
  auto features=upper_features(tangent).flatten();
  auto step=z_.integer("slow_step");
  auto slow=slow_(x).slice(-1,step/2,480,step).flatten().to(torch::kFloat32).to(torch::kFloat64);
  slow_sum_=slow_sum_.defined()?slow_sum_+slow:slow.clone();
  features=torch::cat({features,slow-slow_sum_/(prior+count_)});
  auto standardized=(features-z_.get("mu"))/z_.get("sd");
  finite(standardized,"tangent features");
  return {posterior(torch::matmul(z_.get("W"),standardized)+z_.get("b")),standardized};
}
}
