#pragma once
#include "ni26/source.h"

namespace ni26 {
struct RawNeuralSource {
  Tensor x, rows, session, covariance_prior;
  double scale;
};
// Original unaligned raw-FBC source preprocessing, independent of labels/RNG.
RawNeuralSource prepare_raw_neural_source(const SourceData &source);
} // namespace ni26
