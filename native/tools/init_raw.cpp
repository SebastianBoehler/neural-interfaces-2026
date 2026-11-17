#include "ni26/initialization.h"
#include "ni26/math.h"
#include <charconv>
#include <iostream>
#include <string_view>

int main(int argc, char **argv) {
  try {
    ni26::require(argc == 4, "usage: ni26_init_raw FIR_RECIPE SEED OUTPUT");
    std::string_view seed_text(argv[2]);
    uint64_t seed = 0;
    auto [end, error] = std::from_chars(seed_text.begin(), seed_text.end(), seed);
    ni26::require(error == std::errc{} && end == seed_text.end(), "invalid initializer seed");
    ni26::require(!std::filesystem::exists(argv[3]), "refusing to overwrite native initialization");
    auto recipe = ni26::Params::load(argv[1]);
    ni26::initialize_raw_fbc(recipe, seed).save(argv[3]);
    std::cout << "raw FBC initialized in C++; fixed FIR coefficient provenance remains recipe-owned\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
