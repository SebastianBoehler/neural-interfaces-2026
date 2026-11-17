#include "ni26/runtime.h"
#include <cstdint>
#include <algorithm>
#include <iterator>

extern "C" void ni26_observe(const void* model, int64_t* output) {
  const auto s = static_cast<const ni26::Runtime*>(model)->stats();
  const int64_t fields[] = {s.windows, s.recognized_at, s.fine_tune.refits, s.fine_tune.steps,
      s.fine_tune.relabels, s.fine_tune.capped, s.fine_tune.over, s.personalization.refits,
      s.personalization.fused, s.count_filter.errors + s.personalization.errors};
  std::copy(std::begin(fields), std::end(fields), output);
}
