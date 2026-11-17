#pragma once
#include "ni26/math.h"
#include <utility>

namespace ni26 {
class TangentCore {
 public:
  explicit TangentCore(const Params& z);
  TangentCore(const TangentCore& other);
  TangentCore& operator=(const TangentCore& other);
  void reset();
  std::pair<Tensor,Tensor> step(const Tensor& raw);
 private:
  Params z_;
  std::vector<SosFilter> bank_;
  SosFilter slow_;
  Tensor log_sum_, slow_sum_;
  int64_t count_ = 0;
};
}
