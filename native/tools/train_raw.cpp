#include "ni26/math.h"
#include "ni26/training.h"
#include <iostream>
#include <torch/torch.h>

int main(int argc, char **argv) {
  try {
    ni26::require(argc == 4,
                  "usage: ni26_train_raw RAW_SOURCE_RECIPE OUTPUT DEVICE");
    torch::set_num_threads(1);
    ni26::require(!std::filesystem::exists(argv[2]),
                  "refusing to overwrite a native checkpoint");
    auto source = ni26::Params::load(argv[1]);
    auto result = ni26::train_raw_fbc(source, torch::Device(argv[3]));
    result.tensors["native_format"] = torch::tensor(int64_t(1));
    result.save(argv[2]);
    std::cout << "native raw FBC completed with native preparation, "
                 "initialization and sampling\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
