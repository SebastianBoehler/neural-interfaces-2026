#pragma once
#include "ni26/artifact.h"
#include <cstdint>

namespace ni26 {
// Source-only raw FBC/SessionBN initializer. FIR coefficients are fixed data;
// weights, biases, BN state and a private CPU RNG are constructed in C++.
Params initialize_raw_fbc(const Params &fir_recipe, uint64_t seed);
} // namespace ni26
