#pragma once
#include "ni26/networks.h"
#include "ni26/output.h"
#include <memory>

namespace ni26 {
struct RuntimeStats {
  int64_t windows=0, recognized_at=-1;
  FineTuneStats fine_tune;
  OutputStats count_filter, personalization;
};
class Runtime {
 public:
  Runtime(const Params& z, torch::Device device, const std::filesystem::path& root);
  ~Runtime();
  Runtime(const Runtime&);
  Runtime& operator=(const Runtime&);
  void reset();
  Tensor step(const Tensor& window);
  Tensor predict(const Tensor& windows);
  void set_channels(const std::vector<std::string>& names);
  RuntimeStats stats() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
