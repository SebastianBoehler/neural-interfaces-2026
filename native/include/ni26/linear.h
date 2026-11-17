#pragma once
#include "ni26/artifact.h"

namespace ni26 {
// CPU float64, three fixed classes, unweighted mean CE + ||W||²/(2*C*N).
// Intercepts are unpenalized. Source X must be detached and already standardized.
struct LinearFit {
  Tensor weights, bias;
  int64_t nit = 0, nfev = 0;
  bool converged = false;
  int status = 0, message_code = 0;
  double objective = 0, max_gradient = 0;
};
LinearFit fit_multinomial(const Tensor& x, const Tensor& y, double c);

struct StandardScaler {
  Tensor mean, scale;
  static StandardScaler fit(const Tensor& x);
  Tensor transform(const Tensor& x) const;
};
}
