#include "networks_internal.h"
#include <algorithm>
#include <torch/torch.h>

namespace ni26 {
struct NetBank::Impl {
  struct Group {
    neural::InputStream stream;
    std::vector<std::pair<int64_t, std::shared_ptr<neural::Network>>> nets;
  };
  std::vector<Group> groups;
  torch::Device device;
  Impl(const Params &z, torch::Device dev, const std::filesystem::path &root) : device(dev) {
    std::map<std::string, size_t> indices;
    auto n = z.integer("n_nets");
    auto weights = z.get("member_weights").flatten();
    require(weights.numel() == n + 2, "network member weight count differs");
    for (int64_t i = 0; i < n; ++i) {
      auto p = z.subset("net" + std::to_string(i) + "_");
      auto key = neural::input_key(p);
      if (!indices.count(key)) {
        indices[key] = groups.size();
        groups.push_back({neural::InputStream(p), {}});
      }
      if (weights[i + 1].item<double>() != 0)
        groups[indices.at(key)].nets.push_back({i, neural::build(p, dev, root)});
    }
    groups.erase(
        std::remove_if(groups.begin(), groups.end(), [](const Group &g) { return g.nets.empty(); }),
        groups.end());
  }
  Impl(const Impl &other) : device(other.device) {
    for (const auto &g : other.groups) {
      Group next{g.stream, {}};
      for (const auto &[id, net] : g.nets)
        next.nets.push_back({id, net->clone()});
      groups.push_back(std::move(next));
    }
  }
};
NetBank::NetBank(const Params &z, torch::Device dev, const std::filesystem::path &root)
    : impl_(std::make_unique<Impl>(z, dev, root)) {}
NetBank::~NetBank() = default;
NetBank::NetBank(const NetBank &x) : impl_(std::make_unique<Impl>(*x.impl_)) {}
NetBank &NetBank::operator=(const NetBank &x) {
  if (this != &x)
    impl_ = std::make_unique<Impl>(*x.impl_);
  return *this;
}
void NetBank::reset() {
  for (auto &g : impl_->groups) {
    g.stream.reset();
    for (auto &[id, n] : g.nets)
      n->reset();
  }
}
NetworkStep NetBank::step(const Tensor &x) {
  torch::NoGradGuard no_grad;
  NetworkStep result;
  std::vector<Tensor> probs;
  for (auto &g : impl_->groups) {
    auto h = g.stream.step(x);
    neural::WindowCache cache;
    std::map<torch::ScalarType, Tensor> inputs;
    for (auto &[id, net] : g.nets) {
      auto dtype = net->dtype();
      if (!inputs.count(dtype))
        inputs[dtype] = h.unsqueeze(0).to(impl_->device, dtype);
      auto p = net->proba(inputs.at(dtype), cache);
      require(p.sizes() == torch::IntArrayRef({1, 3}), "neural posterior shape differs");
      finite(p, "neural posterior");
      result.ids.push_back(id);
      result.inputs[id] = h;
      probs.push_back(p[0].cpu());
    }
  }
  require(!probs.empty(), "no active neural members");
  result.probabilities = torch::stack(probs);
  return result;
}
} // namespace ni26
