#include "ni26/preparation.h"
#include "ni26/math.h"
#include <cmath>
#include <iostream>
#include <torch/torch.h>

namespace {
void same(const ni26::Tensor &actual, const ni26::Tensor &expected,
          const std::string &name, double bound) {
  ni26::require(actual.sizes() == expected.sizes(), "source preparation shape differs: " + name);
  ni26::finite(actual, name);
  auto error = (actual.to(torch::kFloat64) - expected.to(torch::kFloat64)).abs().max().item<double>();
  std::cout << name << " maxabs=" << error << " bound=" << bound << '\n';
  ni26::require(error <= bound, "source preparation mismatch: " + name);
}
} // namespace
int main(int argc, char **argv) {
  try {
    ni26::require(argc == 2, "usage: ni26_preparation_source LABEL_FREE_SOURCE_FIXTURE");
    torch::set_num_threads(1);
    auto f = ni26::Params::load(argv[1]);
    ni26::require(!f.has("y"), "neural preparation qualification must not load labels");
    ni26::SourceData source{f.get("x"), f.get("stream"), f.get("person"), f.get("fit_mask"),
                             f.get("held_person")};
    auto result = ni26::prepare_raw_neural_source(source);
    ni26::require(torch::equal(result.rows, f.get("expected_rows")), "source ordering changed");
    ni26::require(torch::equal(result.session, f.get("expected_session")), "source sessions changed");
    std::cout.precision(17);
    same(result.covariance_prior, f.get("expected_G"), "source_G", 1e-8);
    ni26::require(std::abs(result.scale - f.number("expected_scale")) <= 1e-10,
                  "source scale mismatch");
    same(result.x, f.get("expected_x"), "source_X", 1e-6);
    auto relative = (result.x - f.get("expected_x")).to(torch::kFloat64).norm().item<double>() /
                    f.get("expected_x").to(torch::kFloat64).norm().item<double>();
    ni26::require(relative <= 1e-7, "source_X relative error exceeds1e-7");
    auto bad = source;
    bad.fit_mask = source.fit_mask.clone();
    auto row = result.rows[0].item<int64_t>();
    bad.fit_mask[row] = false;
    bool rejected = false;
    try { ni26::select_source_rows(bad); }
    catch (const std::exception &e) {
      rejected = std::string(e.what()).find("partial-session") != std::string::npos;
    }
    ni26::require(rejected, "partial-session source selection accepted");
    bad = source;
    bad.held_person = torch::cat({source.held_person, source.person[row].reshape({1})});
    rejected = false;
    try { ni26::select_source_rows(bad); }
    catch (const std::exception &e) {
      rejected = std::string(e.what()).find("explicitly held person") != std::string::npos;
    }
    ni26::require(rejected, "explicitly held source person accepted");
    std::cout << "NEURAL_SOURCE_PREPARATION_PASS rows=" << result.rows.numel()
              << " relative_L2=" << relative << " labels_loaded=0\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
