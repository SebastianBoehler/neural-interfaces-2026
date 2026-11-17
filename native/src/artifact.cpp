#include "ni26/artifact.h"
#include "ni26/math.h"
#include <torch/serialize.h>
#include <fstream>
#include <iterator>
#include <cmath>
#include <limits>

namespace ni26 {
Params Params::load(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  require(bool(stream), "cannot open native archive: " + path.string());
  std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), {});
  auto object = torch::pickle_load(bytes);
  require(object.isGenericDict(), "native archive must be a tensor dictionary");
  Params result;
  for (const auto& entry : object.toGenericDict()) {
    require(entry.key().isString() && entry.value().isTensor(), "non-tensor native archive entry");
    auto value = entry.value().toTensor().to(torch::kCPU).contiguous();
    require(!value.requires_grad(), "artifact tensors must be detached");
    result.tensors.emplace(entry.key().toStringRef(), value);
  }
  require(result.integer("native_format") == 1, "unsupported native archive version");
  return result;
}
void Params::save(const std::filesystem::path& path) const {
  c10::Dict<std::string, Tensor> values;
  for (const auto& [key, tensor] : tensors) values.insert(key, tensor.detach().cpu().contiguous());
  auto bytes = torch::pickle_save(values);
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  require(bool(stream), "cannot create native archive: " + path.string());
  stream.write(bytes.data(), bytes.size());
  require(bool(stream), "failed to write native archive: " + path.string());
}
bool Params::has(const std::string& key) const { return tensors.count(key) != 0; }
Tensor Params::get(const std::string& key) const {
  auto it = tensors.find(key);
  require(it != tensors.end(), "missing native artifact field: " + key);
  return it->second;
}
double Params::number(const std::string& key) const {
  auto value = get(key);
  require(value.numel() == 1, "expected scalar: " + key);
  return value.item<double>();
}
int64_t Params::integer(const std::string& key) const {
  auto value = get(key);
  require(value.numel() == 1 && !value.is_complex(), "expected integer scalar: " + key);
  if (value.is_floating_point()) {
    double number = value.item<double>();
    require(std::isfinite(number) && std::trunc(number) == number &&
            number >= static_cast<double>(std::numeric_limits<int64_t>::min()) &&
            number < static_cast<double>(std::numeric_limits<int64_t>::max()),
            "nonintegral or out-of-range integer setting: " + key);
    return static_cast<int64_t>(number);
  }
  return value.item<int64_t>();
}
std::vector<std::string> Params::texts(const std::string& key) const {
  auto value = get(key);
  require(value.scalar_type() == torch::kUInt8 && value.dim() == 2, "expected UTF8 rows: " + key);
  std::vector<std::string> result;
  for (int64_t row = 0; row < value.size(0); ++row) {
    const auto* start = value[row].data_ptr<uint8_t>();
    int64_t length = 0;
    while (length < value.size(1) && start[length]) ++length;
    result.emplace_back(reinterpret_cast<const char*>(start), length);
  }
  return result;
}
std::string Params::text(const std::string& key) const {
  auto values = texts(key);
  require(values.size() == 1, "expected one UTF8 row: " + key);
  return values[0];
}
Params Params::subset(const std::string& prefix) const {
  Params result;
  for (const auto& [key, value] : tensors)
    if (key.rfind(prefix, 0) == 0) result.tensors.emplace(key.substr(prefix.size()), value);
  require(!result.tensors.empty(), "missing artifact prefix: " + prefix);
  return result;
}
}
