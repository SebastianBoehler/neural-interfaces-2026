#include "../src/training_internal.h"
#include "ni26/initialization.h"
#include "source_trace.h"
#include <iostream>
#include <torch/torch.h>
ni26::Params source_observations;
using namespace ni26;
int main(int argc, char **argv) {
  try {
    require(argc == 4, "usage: neural_source_stages FIXTURE ACTUAL_NATIVE_UPDATE OUTPUT");
    torch::set_num_threads(1);
    auto fixture = Params::load(argv[1]), actual = Params::load(argv[2]);
    auto original = fixture.subset("initial_");
    auto initialized = initialize_raw_fbc(original, fixture.integer("initialization_seed"));
    for (const auto &[key, value] : original.tensors)
      require(torch::equal(initialized.get(key), value), "trace initializer differs: " + key);
    neural::RawFBC net(initialized, torch::kCPU);
    auto parameters = net.parameters();
    std::vector<Tensor> before;
    for (const auto &p : parameters)
      before.push_back(p.detach().clone());
    auto count = fixture.get("batch_sizes")[0].item<int64_t>();
    auto ids = fixture.get("batch_indices")[0].slice(0, 0, count);
    auto x = fixture.get("source_X").index_select(0, ids);
    auto y = fixture.get("source_y").index_select(0, ids);
    auto logits = net.forward(x, true);
    require(torch::equal(logits, actual.get("first_logits")), "native trace logits do not reproduce production");
    auto loss = torch::nn::functional::cross_entropy(logits, y);
    require(torch::equal(loss, actual.get("first_loss")), "native trace loss does not reproduce production");
    loss.backward();
    const std::vector<std::string> names{"spatial_conv.0.parametrizations.weight.original",
      "spatial_conv.0.bias", "spatial_conv.1.weight", "spatial_conv.1.bias",
      "final_layer.parametrizations.weight.original", "final_layer.bias"};
    for (size_t i = 0; i < parameters.size(); ++i) {
      require(torch::equal(parameters[i], before[i]), "source trace updated a parameter");
      require(parameters[i].grad().defined() && torch::equal(parameters[i].grad(), actual.get("gradient_" + names[i])),
              "native trace gradient does not reproduce production: " + names[i]);
      source_observe("parameter_gradient_" + names[i], parameters[i].grad());
    }
    source_observations.tensors["loss"] = loss.detach();
    for (const auto &[key, value] : net.state().tensors)
      if (key.find("running_") != std::string::npos) {
        require(torch::equal(value, actual.get("state_" + key)), "native trace population state does not reproduce production");
        source_observations.tensors["state_" + key] = value;
      }
    source_observations.tensors["native_format"] = torch::tensor(1, torch::kInt64);
    source_observations.save(argv[3]);
    std::cout << "NATIVE_SOURCE_STAGE_TRACE_REPRODUCES_PRODUCTION_BYTE_EXACT optimizer_updates=0\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
