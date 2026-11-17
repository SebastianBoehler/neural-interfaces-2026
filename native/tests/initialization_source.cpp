#include "ni26/initialization.h"
#include "ni26/math.h"
#include <ATen/CPUGeneratorImpl.h>
#include <iostream>
#include <torch/torch.h>

int main(int argc, char **argv) {
  try {
    ni26::require(argc == 2, "usage: ni26_initialization_source ORIGINAL_INITIALIZERS");
    torch::set_num_threads(1);
    auto fixture = ni26::Params::load(argv[1]);
    auto recipe = fixture.subset("recipe_");
    auto global = at::detail::getDefaultCPUGenerator();
    auto before = global.get_state().clone();
    int64_t checked = 0;
    for (uint64_t seed : {uint64_t(0), uint64_t(1)}) {
      auto reference = fixture.subset("seed" + std::to_string(seed) + "_");
      auto state = ni26::initialize_raw_fbc(recipe, seed);
      ni26::require(state.tensors.size() == reference.tensors.size() + 1,
                    "native initializer omitted or added state fields");
      for (const auto &[key, value] : reference.tensors) {
        auto result = state.get(key);
        ni26::require(result.scalar_type() == value.scalar_type() &&
                          result.sizes() == value.sizes() && torch::equal(result, value),
                      "initializer differs from original: seed" + std::to_string(seed) + "/" + key);
        ++checked;
      }
    }
    ni26::require(torch::equal(before, global.get_state()), "initializer mutated global CPU RNG");
    auto broken = recipe;
    broken.tensors["spectral_filtering.fir_list.0"] = torch::zeros({198});
    bool rejected = false;
    try {
      ni26::initialize_raw_fbc(broken, 0);
    } catch (const std::exception &) {
      rejected = true;
    }
    ni26::require(rejected, "initializer accepted wrong FIR shape");
    std::cout << "NATIVE_INITIALIZATION_PASS exact_state_fields=" << checked
              << " global_rng_unchanged=1 wrong_shape_rejected=1\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
