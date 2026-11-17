#include "ni26/core_training.h"
#include <torch/torch.h>
#include <cmath>

namespace ni26 {
ClassicalPreprocessor::ClassicalPreprocessor(const Params& recipe)
    : bank_(filters(recipe,"sos")),slow_(recipe.get("slow_sos")),
      nsub_(recipe.integer("nsub")),slow_step_(recipe.integer("slow_step")),
      analytic_(recipe.has("hilbert") && recipe.integer("hilbert")!=0),shrink_(recipe.number("shrink")) {
  require(recipe.integer("car")==1 && nsub_==2 && slow_step_==60 &&
          recipe.number("slow_cutoff")==1. && recipe.number("k")==2. && shrink_==.1,
          "unsupported classical core preset; expected actual CAR/fs2/k2/t0.1/slow1Hz-step60");
  // The original real-core artifact omits this flag; absence encodes real, as in TangentCore.
  require(!recipe.has("hilbert") || recipe.integer("hilbert")==0 || recipe.integer("hilbert")==1,"invalid Hilbert flag");
  std::vector<std::vector<double>> expected=analytic_
    ?std::vector<std::vector<double>>{{1,4},{4,8},{8,13},{13,20},{20,30},{30,45},{45,58}}
    :std::vector<std::vector<double>>{{1,3},{2,4},{3,6},{4,8},{6,10},{8,12},{10,14},{12,16},
       {14,18},{16,20},{18,24},{20,28},{24,32},{28,36},{32,40},{36,45},{40,58}};
  auto sos=recipe.get("sos"),slow_sos=recipe.get("slow_sos");
  require(sos.device().is_cpu() && sos.scalar_type()==torch::kFloat64 &&
          sos.sizes()==torch::IntArrayRef({int64_t(expected.size()),4,6}) &&
          slow_sos.device().is_cpu() && slow_sos.scalar_type()==torch::kFloat64 &&
          slow_sos.sizes()==torch::IntArrayRef({2,6}),"actual core SOS coefficient shape/dtype differs");
  auto bands=recipe.get("bands");
  require(bands.device().is_cpu() && bands.scalar_type()==torch::kFloat64 &&
          bands.sizes()==torch::IntArrayRef({int64_t(expected.size()),2}) &&
          int64_t(bank_.size())==int64_t(expected.size()),"classical bands/SOS count differs");
  for (int64_t i=0;i<bands.size(0);++i) for (int j=0;j<2;++j)
    require(bands[i][j].item<double>()==expected[i][j],"classical frequency-band preset differs");
}
ClassicalWindow ClassicalPreprocessor::operator()(const Tensor& raw) const {
  torch::NoGradGuard guard;
  require(raw.device().is_cpu() && raw.scalar_type()==torch::kFloat32 &&
          raw.sizes()==torch::IntArrayRef({41,480}),"classical input must be CPU float32(41,480)");
  finite(raw,"classical source window");
  auto x=common_average_reference_float32(raw);
  std::vector<Tensor> bands;
  for (const auto& filter:bank_) bands.push_back(filter(x).to(torch::kFloat32));
  auto b=torch::stack(bands);
  if (analytic_) {
    auto multiplier=torch::zeros({480},b.options());
    multiplier.index_put_({0},1.);multiplier.index_put_({240},1.);
    multiplier.slice(0,1,240).fill_(2.);
    b=torch::fft_ifft(torch::fft_fft(b,480,-1)*multiplier,480,-1).to(torch::kComplexFloat);
  }
  std::vector<Tensor> segments{b};
  auto pieces=torch::tensor_split(b,nsub_,-1);
  segments.insert(segments.end(),pieces.begin(),pieces.end());
  std::vector<Tensor> covariances;
  for (const auto& segment:segments) {
    auto centered=(segment-segment.mean(-1,true)).to(analytic_?torch::kComplexDouble:torch::kFloat64);
    auto covariance=torch::matmul(centered,centered.transpose(-2,-1).conj())/segment.size(-1);
    auto trace=torch::real(covariance.diagonal(0,-2,-1).sum(-1)).reshape({-1,1,1})/41.;
    covariances.push_back((1-shrink_)*covariance+shrink_*trace*torch::eye(41,covariance.options()));
  }
  auto covariance=torch::stack(covariances,1).reshape({-1,41,41});
  auto log=spectral(covariance,SpectralFunction::Log);
  auto slow=slow_(x).slice(-1,slow_step_/2,480,slow_step_).flatten().to(torch::kFloat32).to(torch::kFloat64);
  finite(covariance,"classical covariance");finite(log,"classical matrix log");finite(slow,"classical slow waveform");
  return {covariance,log,slow};
}
int64_t ClassicalPreprocessor::feature_width() const {
  auto block=analytic_?41*41:41*42/2;
  return int64_t(bank_.size())*(1+nsub_)*block+41*(480/slow_step_);
}
}
