#pragma once
#include "ni26/math.h"
#include <memory>
#include <optional>

namespace ni26 {
class Tail {
 public:
  explicit Tail(const Params& z, bool eye_columns = false);
  ~Tail();
  Tail(const Tail& other);
  Tail& operator=(const Tail& other);
  void reset();
  Tensor step(const Tensor& x, const Tensor& u, const Tensor& mix,
              const Tensor& extra = {});
  Tensor reapply(int64_t count) const;
  void set_prior(double kappa, const Tensor& means);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
