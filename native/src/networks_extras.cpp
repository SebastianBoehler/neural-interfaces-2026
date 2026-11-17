#include "networks_internal.h"
#include <torch/torch.h>

namespace ni26 {
namespace {
Tensor channels(std::initializer_list<int64_t> ids) {
  return torch::tensor(std::vector<int64_t>(ids), torch::kInt64);
}
Tensor select(const Tensor &x, const Tensor &ids) { return x.index_select(0, ids).mean(0); }
Tensor temporal_gradient(const Tensor &x) {
  auto y = torch::empty_like(x);
  y[0].copy_(x[1] - x[0]);
  y[-1].copy_(x[-1] - x[-2]);
  y.slice(0, 1, -1).copy_(.5 * (x.slice(0, 2) - x.slice(0, 0, -2)));
  return y;
}
struct SavGol {
  Tensor center, left, right;
  SavGol() {
    auto q = torch::linspace(-1., 1., 25, torch::kFloat64);
    auto v = torch::stack({torch::ones_like(q), q, q.square(), q.pow(3)}, 1);
    auto inverse = torch::linalg_pinv(v);
    center = inverse[0].view({1, 1, 25});
    left = torch::matmul(v.slice(0, 0, 12), inverse);
    right = torch::matmul(v.slice(0, 13, 25), inverse);
  }
  Tensor operator()(const Tensor &x) const {
    auto y = torch::empty_like(x);
    y.slice(0, 12, 468).copy_(torch::nn::functional::conv1d(x.view({1, 1, 480}), center).flatten());
    y.slice(0, 0, 12).copy_(torch::matmul(left, x.slice(0, 0, 25)));
    y.slice(0, 468, 480).copy_(torch::matmul(right, x.slice(0, 455, 480)));
    return y;
  }
};
std::vector<std::shared_ptr<torch::jit::Module>>
modules(const Params &z, const char *key, torch::Device dev, const std::filesystem::path &root) {
  std::vector<std::shared_ptr<torch::jit::Module>> result;
  for (const auto &name : z.texts(key)) {
    auto m = std::make_shared<torch::jit::Module>(torch::jit::load((root / name).string(), dev));
    m->eval();
    result.push_back(m);
  }
  require(!result.empty(), std::string("empty ") + key);
  return result;
}
} // namespace
struct EyeNet::Impl {
  torch::Device device;
  std::vector<std::shared_ptr<torch::jit::Module>> nets;
  SavGol smooth;
  Tensor s1, s2;
  int64_t count = 0;
  Impl(const Params &z, torch::Device dev, const std::filesystem::path &root)
      : device(dev), nets(modules(z, "v6_eye_files", dev, root)) {
    reset();
  }
  Impl(const Impl &x) : device(x.device), nets(x.nets), smooth(x.smooth) { reset(); }
  void reset() {
    count = 0;
    s1 = torch::zeros({7}, torch::kFloat64);
    s2 = torch::zeros_like(s1);
  }
  Tensor traces(const Tensor &input) {
    require(input.sizes() == torch::IntArrayRef({41, 480}), "eye input must be41×480");
    auto x = input.to(torch::kCPU, torch::kFloat64);
    auto h1 = smooth(select(x, channels({32, 31, 33})) - select(x, channels({27, 30, 28})));
    auto h2 = smooth(select(x, channels({36, 40})) - select(x, channels({24, 22})));
    auto top = select(x, channels({31, 30}));
    auto v1 = smooth(top - select(x, channels({8, 2, 3})));
    auto v2 = smooth(top - select(x, channels({6, 0, 5})));
    auto t = torch::stack({h1, h2, v1, v2, torch::sqrt(h1.square() + v1.square()),
                           temporal_gradient(h1), temporal_gradient(v1)});
    ++count;
    s1 = s1 + t.mean(-1);
    s2 = s2 + t.square().mean(-1);
    auto mean = s1 / count;
    return ((t - mean.unsqueeze(-1)) /
            torch::sqrt((s2 / count - mean.square()).clamp_min(1e-12)).unsqueeze(-1))
        .clamp(-8, 8)
        .to(torch::kFloat32);
  }
};
EyeNet::EyeNet(const Params &z, torch::Device d, const std::filesystem::path &r)
    : impl_(std::make_unique<Impl>(z, d, r)) {}
EyeNet::~EyeNet() = default;
EyeNet::EyeNet(const EyeNet &x) : impl_(std::make_unique<Impl>(*x.impl_)) {}
EyeNet &EyeNet::operator=(const EyeNet &x) {
  if (this != &x)
    impl_ = std::make_unique<Impl>(*x.impl_);
  return *this;
}
void EyeNet::reset() { impl_->reset(); }
Tensor EyeNet::step(const Tensor &x) {
  torch::NoGradGuard guard;
  auto t = impl_->traces(x).unsqueeze(0).to(impl_->device);
  std::vector<Tensor> values;
  for (auto &m : impl_->nets)
    values.push_back(torch::softmax(m->forward({t}).toTensor().to(torch::kFloat64), 1)[0]);
  return torch::stack(values).mean(0).cpu().clamp(1e-3, 1).log();
}

struct RestMember::Impl {
  neural::InputStream stream;
  torch::Device device;
  std::vector<std::shared_ptr<torch::jit::Module>> nets;
  Impl(const Params &z, torch::Device d, const std::filesystem::path &r)
      : stream(z.subset("v6_rest_")), device(d), nets(modules(z, "v6_rest_files", d, r)) {}
  Impl(const Impl &x) : stream(x.stream), device(x.device) {
    for (const auto &m : x.nets)
      nets.push_back(std::make_shared<torch::jit::Module>(m->clone()));
  }
};
RestMember::RestMember(const Params &z, torch::Device d, const std::filesystem::path &r)
    : impl_(std::make_unique<Impl>(z, d, r)) {}
RestMember::~RestMember() = default;
RestMember::RestMember(const RestMember &x) : impl_(std::make_unique<Impl>(*x.impl_)) {}
RestMember &RestMember::operator=(const RestMember &x) {
  if (this != &x)
    impl_ = std::make_unique<Impl>(*x.impl_);
  return *this;
}
void RestMember::reset() {
  impl_->stream.reset();
  for (auto &m : impl_->nets)
    neural::zero_stream(*m);
}
Tensor RestMember::step(const Tensor &x) {
  torch::NoGradGuard guard;
  auto h = impl_->stream.step(x).unsqueeze(0).to(impl_->device);
  std::vector<Tensor> values;
  for (auto &m : impl_->nets)
    values.push_back(
        torch::softmax(m->forward({h}).toTensor().slice(1, 0, 3).to(torch::kFloat64), 1)[0]);
  return torch::stack(values).mean(0).cpu();
}
} // namespace ni26
