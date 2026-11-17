#pragma once
#include "ni26/math.h"
#include "ni26/networks.h"
#include <torch/script.h>

namespace ni26::neural {
struct WindowCache {
  std::map<int64_t, Tensor> spectra;
  std::map<std::string, Tensor> banks;
  Tensor shared_sbn;
};
class InputStream {
public:
  explicit InputStream(const Params &p);
  InputStream(const InputStream &);
  InputStream &operator=(const InputStream &);
  void reset();
  Tensor step(const Tensor &x);

private:
  bool align_;
  double scale_, k_, shrink_;
  Tensor prior_, sum_;
  int64_t count_ = 0;
};
std::string input_key(const Params &p);
Tensor temporal_row_mean_float32(const Tensor &x);
class Network {
public:
  virtual ~Network() = default;
  virtual torch::ScalarType dtype() const { return torch::kFloat32; }
  virtual void reset() {}
  virtual Tensor proba(const Tensor &x, WindowCache &cache) = 0;
  virtual std::shared_ptr<Network> clone() const = 0;
};
std::shared_ptr<Network> build(const Params &p, torch::Device device,
                               const std::filesystem::path &root);
void zero_stream(torch::jit::Module &module);
std::vector<Tensor> trainable(torch::jit::Module &module);
Tensor strong(const Tensor &x, at::Generator &generator);
at::Generator generator(torch::Device device, int64_t seed);
} // namespace ni26::neural
