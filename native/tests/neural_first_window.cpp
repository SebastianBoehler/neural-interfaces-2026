#include "ni26/math.h"
#include "ni26/networks.h"
#include <iostream>
#include <torch/torch.h>

int main(int argc, char **argv) {
  try {
    ni26::require(argc == 5, "usage: neural_first_window PARAMS FIXTURE MODULE_ROOT OUTPUT");
    torch::set_num_threads(1);
    torch::NoGradGuard guard;
    auto z = ni26::Params::load(argv[1]), f = ni26::Params::load(argv[2]);
    ni26::NetBank bank(z, torch::kCPU, argv[3]);
    auto step = bank.step(f.get("known_x")[0]);
    ni26::Params output;
    output.tensors["native_format"] = torch::tensor(1, torch::kInt64);
    output.tensors["ids"] = torch::tensor(step.ids, torch::kInt64);
    output.tensors["probs"] = step.probabilities;
    for (const auto &[id, input] : step.inputs)
      output.tensors["input_" + std::to_string(id)] = input;
    ni26::EyeNet eye(z, torch::kCPU, argv[3]);
    ni26::RestMember rest(z, torch::kCPU, argv[3]);
    output.tensors["eye"] = eye.step(f.get("known_x")[0]);
    output.tensors["rest"] = rest.step(f.get("known_x")[0]);
    output.save(argv[4]);
    std::cout << "REAL_FIRST_WINDOW_CAPTURED members=" << step.ids.size() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
