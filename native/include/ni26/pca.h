#pragma once
#include "ni26/artifact.h"

namespace ni26 {
struct TsemBasis {
  Tensor mean;             // float64[D], centering of standardized source features
  Tensor basis;            // float32[D,256]; low128=TSEM, high128=route
  Tensor singular_values;  // float64[N], every thin-SVD singular value retained
  int64_t workspace_elements;
};
// Exact wide thin DGESDD + sklearn svd_flip(u_based_decision=false).
// Input is detached contiguous CPU float64 standardized source features, N<=D.
// No scaler, randomized approximation, rank threshold or alternative SVD driver.
TsemBasis fit_tsem_basis(const Tensor& standardized_source);
}
