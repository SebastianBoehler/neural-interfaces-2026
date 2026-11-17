#pragma once
#include <torch/types.h>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace ni26 {
using Tensor = torch::Tensor;
// Native, versioned tensor archive. No Python objects or executable callbacks.
class Params {
 public:
  std::map<std::string, Tensor> tensors;
  static Params load(const std::filesystem::path& path);
  void save(const std::filesystem::path& path) const;
  bool has(const std::string& key) const;
  Tensor get(const std::string& key) const;
  double number(const std::string& key) const;
  int64_t integer(const std::string& key) const;
  std::string text(const std::string& key) const;
  std::vector<std::string> texts(const std::string& key) const;
  Params subset(const std::string& prefix) const;
};
}
