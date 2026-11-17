// Prepared source-only PCA test: exact original input, no labels or head fitting.
#include "ni26/pca.h"
#include "ni26/math.h"
#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <iostream>
#include <sys/resource.h>
extern "C" int scipy_openblas_get_num_threads();
namespace {
int64_t current_rss_KiB() {
  std::ifstream input("/proc/self/statm");int64_t pages=0,resident=0;
  ni26::require(bool(input>>pages>>resident),"cannot read Linux process RSS");
  return resident*sysconf(_SC_PAGESIZE)/1024;
}
double error(const ni26::Tensor& a,const ni26::Tensor& b) {
  ni26::require(a.sizes()==b.sizes(),"PCA reference shape mismatch");
  return (a-b).abs().max().item<double>();
}
}
int main(int argc,char** argv) {
  try {
    ni26::require(argc==3,"usage: ni26_pca_source COMMON_SOURCE.pt NEW_NATIVE_OUTPUT.pt");
    ni26::require(!std::filesystem::exists(argv[2]),"new PCA output required");
    torch::NoGradGuard guard;torch::set_num_threads(6);
    ni26::require(scipy_openblas_get_num_threads()==6,"native exact-SVD provider must use six threads");
    auto input=ni26::Params::load(argv[1]);auto z=input.get("source_Z");
    ni26::require(z.sizes()==torch::IntArrayRef({2278,44239}),"registered PCA source dimensions differ");
    rusage baseline{};getrusage(RUSAGE_SELF,&baseline);
    auto loaded_rss=current_rss_KiB();
    ni26::TsemBasis result;std::vector<double> seconds;
    for (int iteration=0;iteration<2;++iteration) {
      auto start=std::chrono::steady_clock::now();result=ni26::fit_tsem_basis(z);
      seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
      std::cout<<"pca_fit="<<iteration<<" seconds="<<seconds.back()
        <<" workspace_elements="<<result.workspace_elements<<std::endl;
    }
    rusage peak{};getrusage(RUSAGE_SELF,&peak);
    auto mean_error=error(result.mean,input.get("reference_mean"));
    auto basis_error=error(result.basis,input.get("reference_basis"));
    auto singular_relative=error(result.singular_values,input.get("reference_singular"))/
      input.get("reference_singular")[0].item<double>();
    ni26::require(mean_error<=1e-12 && basis_error<=1e-6 && singular_relative<=1e-10,
                  "native PCA mean/basis/all-singular-value parity failed");
    auto basis=result.basis.to(torch::kFloat64);
    auto orthogonal=error(torch::matmul(basis.t(),basis),torch::eye(256,basis.options()));
    ni26::require(orthogonal<=1e-6,"native float32 PCA basis orthogonality failed");
    double projection_max=0,projection_error2=0,projection_norm2=0,energy=0,captured=0;
    for (int64_t start=0;start<z.size(0);start+=64) {
      auto stop=std::min(z.size(0),start+64);
      auto centered=z.slice(0,start,stop)-result.mean;
      auto view=torch::matmul(centered,basis);
      auto expected=input.get("reference_pc").slice(0,start,stop).to(torch::kFloat64);
      auto difference=view.to(torch::kFloat32).to(torch::kFloat64)-expected;
      projection_max=std::max(projection_max,difference.abs().max().item<double>());
      projection_error2+=difference.square().sum().item<double>();projection_norm2+=expected.square().sum().item<double>();
      energy+=centered.square().sum().item<double>();captured+=view.square().sum().item<double>();
    }
    auto projection_relative=std::sqrt(projection_error2/projection_norm2);
    auto total_singular=result.singular_values.square().sum().item<double>();
    auto kept_singular=result.singular_values.slice(0,0,256).square().sum().item<double>();
    auto energy_error=std::abs(total_singular-energy)/energy;
    auto captured_error=std::abs(kept_singular-captured)/kept_singular;
    ni26::require(projection_max<=1e-3 && projection_relative<=1e-6 &&
                  energy_error<=1e-10 && captured_error<=1e-6,"native PCA projection/energy residual failed");
    ni26::Params output;output.tensors={{"native_format",torch::tensor(int64_t(1))},
      {"mean",result.mean},{"basis",result.basis},{"singular_values",result.singular_values},
      {"fit_seconds",torch::tensor(seconds,torch::kFloat64)},
      {"workspace_elements",torch::tensor(result.workspace_elements)},
      {"loaded_rss_KiB",torch::tensor(loaded_rss)},
      {"loaded_peak_KiB",torch::tensor(int64_t(baseline.ru_maxrss))},
      {"fit_peak_KiB",torch::tensor(int64_t(peak.ru_maxrss))}};
    output.save(argv[2]);
    std::cout.precision(12);
    std::cout<<"PCA_SOURCE_PASS mean="<<mean_error<<" basis="<<basis_error
      <<" singular_relative="<<singular_relative<<" orthogonal="<<orthogonal
      <<" projection_max="<<projection_max<<" projection_relative="<<projection_relative
      <<" full_energy_residual="<<energy_error<<" captured_energy_residual="<<captured_error
      <<" loaded_rss_KiB="<<loaded_rss<<" loaded_peak_KiB="<<baseline.ru_maxrss<<" fit_peak_KiB="<<peak.ru_maxrss<<'\n';
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
