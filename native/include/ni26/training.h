#pragma once
#include "ni26/artifact.h"

namespace ni26 {
// Existing source6 raw-FBC recipe: raw x/stream/person/fit_mask/held_person/y,
// fixed fir_ coefficients, source identities and initialization_seed. Source
// preprocessing, initialization and all40epoch batches are constructed
// natively.
Params train_raw_fbc(const Params &source, torch::Device device);
} // namespace ni26
