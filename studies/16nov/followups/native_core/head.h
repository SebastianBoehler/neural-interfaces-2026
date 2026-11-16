#pragma once
#include <cstdint>

// CPU float64, row-major arrays; classes are exactly 0,1,2. No state retained.
// Predictor consumes raw features and the nine-person reference scaler/head.
extern "C" int nc_head_standardize(const double* x, std::int64_t n, std::int64_t d,
    const double* mean, const double* scale, double* output);
extern "C" int nc_head_predict(const double* x, std::int64_t n, std::int64_t d,
    const double* mean, const double* scale, const double* coefficients,
    const double* intercepts, double* logits, double* probabilities);
// Objective/fit consume standardized features. theta is 3 x (d+1), with each
// class intercept in its last column. Mean CE + ||W||²/(2*C*n); no intercept L2.
extern "C" int nc_head_objective(const double* x, const std::int64_t* y,
    std::int64_t n, std::int64_t d, const double* theta, double c,
    double* loss, double* gradient);
// Fit starts at zero; output count is objective evaluations, not optimizer steps.
extern "C" int nc_head_fit(const double* x, const std::int64_t* y,
    std::int64_t n, std::int64_t d, double c, double* theta,
    double* loss, double* max_gradient, std::int64_t* evaluations);
extern "C" const char* nc_head_error();
