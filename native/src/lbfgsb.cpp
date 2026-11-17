#include "linear_internal.h"
#include "ni26/math.h"
#include <array>
#include <climits>
#include <cmath>
extern "C" {
#include "../vendor/scipy_lbfgsb/lbfgsb.h"
}

namespace ni26::detail {
LbfgsbResult minimize_lbfgsb(int64_t variables, const ValueGradient& objective) {
  static_assert(sizeof(CBLAS_INT) == 4, "qualified L-BFGS-B uses LP64 BLAS integers");
  require(variables > 0 && variables <= INT_MAX, "invalid native L-BFGS-B dimension");
  constexpr int history = 10, max_iterations = 2000, max_evaluations = 15000;
  constexpr int max_linesearch = 50;
  auto n = static_cast<int>(variables);
  LbfgsbResult result;
  result.parameters.resize(n); result.gradient.resize(n);
  std::vector<double> lower(n), upper(n), work((2*history+5)*size_t(n)+11*history*history+8*history);
  std::vector<int> bounds(n), integer_work(3*size_t(n));
  std::array<int,2> task{}, line_task{};
  std::array<int,4> logical_save{};
  std::array<int,44> integer_save{};
  std::array<double,29> double_save{};
  while (true) {
    setulb(n, history, result.parameters.data(), lower.data(), upper.data(), bounds.data(),
           &result.objective, result.gradient.data(), 64., 1e-5, work.data(), integer_work.data(),
           task.data(), logical_save.data(), integer_save.data(), double_save.data(),
           max_linesearch, line_task.data());
    if (task[0] == 3) {
      result.objective = objective(result.parameters.data(), result.gradient.data());
      ++result.nfev;
      require(std::isfinite(result.objective), "nonfinite native L-BFGS-B objective");
      for (double value : result.gradient)
        require(std::isfinite(value), "nonfinite native L-BFGS-B gradient");
    } else if (task[0] == 1) {
      ++result.nit;
      // Match SciPy's checks AFTER accepting the iteration, including strict >maxfun.
      if (result.nit >= max_iterations) { task[0]=5; task[1]=504; }
      else if (result.nfev > max_evaluations) { task[0]=5; task[1]=502; }
    } else break;
  }
  result.status = task[0]; result.message_code = task[1];
  return result;
}
}
