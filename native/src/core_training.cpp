#include "ni26/core_training.h"
#include <torch/torch.h>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ni26 {
namespace {
const std::set<std::string> fixed_fields{"sos","slow_sos","bands","car","nsub","hilbert",
                                        "shrink","k","slow_cutoff","slow_step"};
void vector_field(const Tensor& value,int64_t n,torch::ScalarType type,const std::string& name) {
  require(value.device().is_cpu() && value.is_contiguous() && value.scalar_type()==type &&
          value.dim()==1 && value.size(0)==n,"invalid source vector: "+name);
}
struct Session {int64_t person=-1,total=0,selected=0;Tensor log,slow;std::vector<int64_t> rows;};
}
CoreTrainingResult train_classical_core(const Params& source,const Params& recipe) {
  torch::NoGradGuard guard;
  require(source.integer("native_format")==1,"unsupported raw source archive");
  const std::set<std::string> source_fields{"native_format","x","y","stream","person","fit_mask","held_person"};
  for (const auto& [key,value]:source.tensors)
    require(source_fields.count(key)!=0,"unsupported raw source field: "+key);
  for (const auto& [key,value]:recipe.tensors)
    require(fixed_fields.count(key)!=0,"learned/unsupported core recipe field: "+key);
  require(recipe.tensors.size()==fixed_fields.size(),"incomplete immutable core recipe");
  ClassicalPreprocessor preprocess(recipe);
  auto raw=source.get("x"),y=source.get("y"),stream=source.get("stream"),person=source.get("person");
  auto mask=source.get("fit_mask"),held=source.get("held_person");
  require(raw.device().is_cpu() && raw.scalar_type()==torch::kFloat32 && raw.dim()==3 &&
          raw.size(1)==41 && raw.size(2)==480 && raw.size(0)>0 && raw.is_contiguous() && !raw.requires_grad(),"raw source must be CPU float32[N,41,480]");
  auto n=raw.size(0);
  vector_field(y,n,torch::kInt64,"y");vector_field(stream,n,torch::kInt64,"stream");
  vector_field(person,n,torch::kInt64,"person");vector_field(mask,n,torch::kBool,"fit_mask");
  require(held.dim()==1,"held_person must be a vector");
  vector_field(held,held.numel(),torch::kInt64,"held_person");
  std::set<int64_t> excluded;
  for (int64_t i=0;i<held.numel();++i)
    require(held[i].item<int64_t>()>=0 && excluded.insert(held[i].item<int64_t>()).second,
            "held_person IDs must be nonnegative and unique");
  std::map<int64_t,Session> sessions;
  auto rows=torch::nonzero(mask).flatten().contiguous();
  require(rows.numel()>=3,"insufficient selected source rows");
  for (int64_t i=0;i<n;++i) {
    auto id=stream[i].item<int64_t>(),who=person[i].item<int64_t>();
    require(id>=0 && who>=0,"source stream/person IDs must be nonnegative");
    auto& session=sessions[id];
    require(session.person<0 || session.person==who,"stream ID crosses people");session.person=who;
    ++session.total;
    if (mask[i].item<bool>()) {
      require(!excluded.count(who),"selected source row belongs to an explicitly held person");
      ++session.selected;
    }
  }
  for (const auto& [id,session]:sessions)
    require(session.selected==0 || session.selected==session.total,"partial-session source fit mask is unsupported");
  std::vector<ClassicalWindow> windows;
  windows.reserve(rows.numel());
  Tensor global_log;
  for (int64_t i=0;i<rows.numel();++i) {
    auto index=rows[i].item<int64_t>();
    auto window=preprocess(raw[index]);
    auto& session=sessions[stream[index].item<int64_t>()];
    if (!global_log.defined()) global_log=torch::zeros_like(window.log_covariance);
    if (!session.log.defined()) {
      session.log=torch::zeros_like(window.log_covariance);session.slow=torch::zeros_like(window.slow);
    }
    // Preserve original source-row accumulation order; all windows have equal weight.
    global_log.add_(window.log_covariance);session.log.add_(window.log_covariance);
    session.slow.add_(window.slow);session.rows.push_back(i);
    window.log_covariance=Tensor{};windows.push_back(std::move(window));
  }
  global_log.div_(rows.numel());
  auto features=torch::empty({rows.numel(),preprocess.feature_width()},torch::kFloat64);
  for (const auto& [id,session]:sessions) if (session.selected) {
    auto reference=spectral(session.log/session.selected,SpectralFunction::Exp);
    auto whitening=spectral(reference,SpectralFunction::InverseRoot);
    auto slow_mean=session.slow/session.selected;
    for (auto i:session.rows) {
      const auto& window=windows[i];
      auto tangent=spectral(torch::matmul(torch::matmul(whitening,window.covariance),whitening),SpectralFunction::Log);
      auto value=torch::cat({upper_features(tangent).flatten(),window.slow-slow_mean});
      require(value.numel()==features.size(1),"native source core feature width differs");
      features[i].copy_(value);
    }
  }
  windows.clear();
  finite(global_log,"source-only global log prior");
  // StandardScaler validates the complete feature matrix without another full-size temporary.
  auto scaler=StandardScaler::fit(features);
  auto standardized=scaler.transform(features);
  auto c=recipe.integer("hilbert")?3.:10.;
  auto head=fit_multinomial_core(standardized,y.index_select(0,rows).contiguous(),c);
  require(head.converged,"actual core head did not converge status="+std::to_string(head.status)+"/"+std::to_string(head.message_code));
  Params model;
  for (const auto& [key,value]:recipe.tensors) model.tensors.emplace(key,value.clone());
  model.tensors.insert({{"log_glob",global_log},{"mu",scaler.mean},{"sd",scaler.scale},
    {"W",head.weights},{"b",head.bias},{"training_rows",rows.clone()},
    {"training_person",person.index_select(0,rows)},{"training_stream",stream.index_select(0,rows)},
    {"training_C",torch::tensor(c,torch::kFloat64)},{"training_gtol",torch::tensor(1e-4,torch::kFloat64)},
    {"training_nit",torch::tensor(head.nit)},{"training_nfev",torch::tensor(head.nfev)},
    {"training_objective",torch::tensor(head.objective,torch::kFloat64)},
    {"training_max_gradient",torch::tensor(head.max_gradient,torch::kFloat64)}});
  return {model,features,rows,head};
}
}
