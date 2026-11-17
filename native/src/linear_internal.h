#pragma once
#include "ni26/linear.h"
#include <functional>
#include <vector>

namespace ni26::detail {
void linear_input(const Tensor& x, int64_t dimensions);
struct LbfgsbResult {
  std::vector<double> parameters, gradient;
  int64_t nit = 0, nfev = 0;
  int status = 0, message_code = 0;
  double objective = 0;
};
using ValueGradient = std::function<double(const double*, double*)>;
LbfgsbResult minimize_lbfgsb(int64_t variables, const ValueGradient& objective);
}
