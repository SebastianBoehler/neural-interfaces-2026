#pragma once
#include "ni26/artifact.h"
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ni26 {
struct NetworkStep {
  std::vector<int64_t> ids;
  Tensor probabilities;             // CPU float64, [member, class], original accumulation order.
  std::map<int64_t, Tensor> inputs; // CPU float32, [channel, time], before each network.
};

class NetBank {
public:
  NetBank(const Params &z, torch::Device device, const std::filesystem::path &root);
  ~NetBank();
  NetBank(const NetBank &);
  NetBank &operator=(const NetBank &);
  void reset();
  NetworkStep step(const Tensor &x);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class EyeNet {
public:
  EyeNet(const Params &z, torch::Device device, const std::filesystem::path &root);
  ~EyeNet();
  EyeNet(const EyeNet &);
  EyeNet &operator=(const EyeNet &);
  void reset();
  Tensor step(const Tensor &x); // Three log probabilities for SessionEM.
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class RestMember {
public:
  RestMember(const Params &z, torch::Device device, const std::filesystem::path &root);
  ~RestMember();
  RestMember(const RestMember &);
  RestMember &operator=(const RestMember &);
  void reset();
  Tensor step(const Tensor &x); // Three-way softmax of the first three logits.
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct FineTuneStats {
  int64_t windows = 0, refits = 0, steps = 0, capped = 0, over = 0, relabels = 0;
  double maximum_parameter_change = 0, refit_seconds = 0;
};
using RelabelFn = std::function<std::optional<Tensor>(int64_t)>;

class FineTune {
public:
  FineTune(const Params &z, const Tensor &weights, torch::Device device,
           const std::filesystem::path &root);
  ~FineTune();
  FineTune(const FineTune &);
  FineTune &operator=(const FineTune &);
  void reset();
  std::map<std::string, Tensor> step(const NetworkStep &members, const Tensor &label,
                                     const RelabelFn &relabel = {});
  void relabel(const std::vector<Tensor> &history);
  const std::map<std::string, std::vector<int64_t>> &slots() const;
  const std::map<std::string, int64_t> &members() const;
  FineTuneStats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace ni26
