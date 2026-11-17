#include "ni26/core.h"
#include <torch/torch.h>
#include <iostream>
#include <string>

namespace {
void dump(ni26::Params& output,const ni26::Params& z,const ni26::Tensor& raw,
          const std::string& prefix,const ni26::Tensor& override_car={},
          const ni26::Tensor& override_means={}) {
  auto mean=raw.mean(0,true);
  auto x=override_car.defined()?override_car:raw-mean;
  auto bank=ni26::filters(z,"sos");ni26::SosFilter slow_filter(z.get("slow_sos"));
  std::vector<ni26::Tensor> filtered;
  for (const auto& filter:bank) filtered.push_back(filter(x));
  auto b64=torch::stack(filtered),b=b64.to(torch::kFloat32);
  bool analytic=z.has("hilbert") && z.number("hilbert")!=0;
  if (analytic) {
    auto multiplier=torch::zeros({480},b.options());
    multiplier.index_put_({0},1.);multiplier.index_put_({240},1.);multiplier.slice(0,1,240).fill_(2.);
    b=torch::fft_ifft(torch::fft_fft(b,480,-1)*multiplier,480,-1);
  }
  std::vector<ni26::Tensor> segments{b},covariances,means;
  auto pieces=torch::tensor_split(b,z.integer("nsub"),-1);
  segments.insert(segments.end(),pieces.begin(),pieces.end());
  int64_t index=0;
  for (const auto& segment:segments) {
    auto center=override_means.defined()?override_means.select(1,index):segment.mean(-1,true);
    means.push_back(center);++index;
    auto centered=(segment-center).to(analytic?torch::kComplexDouble:torch::kFloat64);
    auto covariance=torch::matmul(centered,centered.transpose(-2,-1).conj())/segment.size(-1);
    auto trace=torch::real(covariance.diagonal(0,-2,-1).sum(-1)).reshape({-1,1,1})/41.;
    auto shrink=z.number("shrink");
    covariances.push_back((1-shrink)*covariance+shrink*trace*torch::eye(41,covariance.options()));
  }
  auto cov=torch::stack(covariances,1).reshape({-1,41,41});
  auto log=ni26::spectral(cov,ni26::SpectralFunction::Log);
  auto reference=(z.number("k")*z.get("log_glob")+log)/(z.number("k")+1);
  auto whitening=ni26::spectral(reference,ni26::SpectralFunction::NegativeHalfExp);
  auto tangent=ni26::spectral(torch::matmul(torch::matmul(whitening,cov),whitening),ni26::SpectralFunction::Log);
  auto step=z.integer("slow_step");
  auto slow64=slow_filter(x).slice(-1,step/2,480,step).flatten();
  auto slow=slow64.to(torch::kFloat32).to(torch::kFloat64);
  auto feature=torch::cat({ni26::upper_features(tangent).flatten(),slow-slow/(z.number("k")+1)});
  auto standardized=(feature-z.get("mu"))/z.get("sd");
  auto logits=torch::matmul(z.get("W"),standardized)+z.get("b");
  auto add=[&](const std::string& key,const ni26::Tensor& value){output.tensors.emplace(prefix+key,value);};
  add("car_mean",mean);add("car",x);add("filtered64",b64);add("filtered32",b64.to(torch::kFloat32));
  add("analytic",b);add("time_mean",torch::stack(means,1));add("covariance",cov);
  add("log_covariance",log);add("reference_log",reference);add("whitening",whitening);
  add("tangent",tangent);add("slow64",slow64);add("slow",slow);add("feature",feature);
  add("standardized",standardized);add("logits",logits);add("probabilities",ni26::posterior(logits));
}
}
int main(int argc,char** argv) {
  try {
    ni26::require(argc==6,"usage: core_precision_dump MODEL SOURCE ORACLE OUTPUT MODE");
    torch::set_num_threads(1);torch::NoGradGuard guard;
    auto model=ni26::Params::load(argv[1]),source=ni26::Params::load(argv[2]);
    auto oracle=ni26::Params::load(argv[3]);
    ni26::Params output;output.tensors.emplace("native_format",torch::tensor(int64_t(1)));
    auto raw=source.get("x")[0];std::string mode=argv[5];
    ni26::require(mode=="initial" || mode=="means","unsupported diagnostic mode");
    for (const std::string stem:{"ts_","ts2_"}) {
      auto z=model.subset(stem);
      if (mode=="initial") {
        dump(output,z,raw,stem+"native_");
        dump(output,z,raw,stem+"refcar_",oracle.get(stem+"car"));
      } else {
        dump(output,z,raw,stem+"npmean_",{},oracle.get(stem+"native_np_time_mean"));
        dump(output,z,raw,stem+"both_",oracle.get(stem+"car"),oracle.get(stem+"refcar_np_time_mean"));
      }
    }
    output.save(argv[4]);
    std::cout<<"SOURCE_FIRST_WINDOW_DUMP "<<mode<<'\n';return 0;
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
