#include "networks_internal.h"
#include "finetune_adam.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/torch.h>
#ifdef NI26_WITH_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace ni26 {
namespace {
void release_cache(torch::Device device) {
#ifdef NI26_WITH_CUDA
  // A CPU model must not initialize or query a CUDA driver during reset/refit.
  if (!device.is_cuda())
    return;
  if (c10::cuda::getDeviceIndexWithPrimaryContext())
    c10::cuda::CUDACachingAllocator::emptyCache();
#else
  (void)device;
#endif
}
std::vector<int64_t> integers(const Tensor &value) {
  auto x = value.to(torch::kCPU, torch::kInt64).contiguous().flatten();
  return {x.data_ptr<int64_t>(), x.data_ptr<int64_t>() + x.numel()};
}
void check_label(const Tensor &value) {
  require(value.sizes() == torch::IntArrayRef({3}), "fine-tuning label must have three classes");
  finite(value, "fine-tuning label");
}
} // namespace
struct FineTune::Impl {
  std::vector<std::string> names;
  std::map<std::string, std::vector<int64_t>> slots;
  std::map<std::string, int64_t> members;
  std::map<std::string, std::string> blobs;
  std::map<std::string, std::shared_ptr<torch::jit::Module>> nets;
  std::map<std::string, std::vector<Tensor>> initial, inputs;
  std::map<std::string, at::Generator> generators;
  std::vector<int64_t> labels;
  std::vector<double> confidence;
  torch::Device device;
  int64_t start, every, steps, seed, cap;
  double lr, lambda, seconds;
  bool on;
  FineTuneStats stats;
  Impl(const Params &z, const Tensor &weights, torch::Device dev, const std::filesystem::path &root)
      : names(z.texts("v6_slots")), device(dev), start(z.integer("v6_ft_t0")),
        every(z.integer("v6_ft_every")), steps(z.integer("v6_ft_steps")),
        seed(z.integer("v6_ft_seed")), cap(z.integer("v6_ft_cap")), lr(z.number("v6_ft_lr")),
        lambda(z.number("v6_ft_lam")), seconds(z.number("v6_ft_session_s")),
        on(z.integer("v6_ft_on")) {
    auto ids = integers(z.get("v6_ft_member"));
    require(ids.size() == names.size(), "fine-tuning family/member count differs");
    require(start > 0 && every > 0 && steps > 0 && cap > 0 && seconds > 0 && lr > 0,
            "invalid fine-tuning schedule");
    for (size_t i = 0; i < names.size(); ++i) {
      const auto &s = names[i];
      members[s] = ids[i];
      slots[s] = integers(z.get("v6_slot_" + s));
      bool present = false;
      for (auto id : slots[s]) {
        require(weights[id + 1].item<double>() > 0, "inactive fine-tuning slot member");
        present |= id == ids[i];
      }
      require(present, "fine-tuning member absent from slot");
      std::ifstream file(root / z.text("net" + std::to_string(ids[i]) + "_sjit_file"),
                         std::ios::binary);
      require(bool(file), "cannot read fine-tuning module");
      blobs[s] = std::string(std::istreambuf_iterator<char>(file), {});
    }
    reset();
  }
  Impl(const Impl &x)
      : names(x.names), slots(x.slots), members(x.members), blobs(x.blobs), device(x.device),
        start(x.start), every(x.every), steps(x.steps), seed(x.seed), cap(x.cap), lr(x.lr),
        lambda(x.lambda), seconds(x.seconds), on(x.on) {
    reset();
  }
  void reset() {
    nets.clear();
    initial.clear();
    generators.clear();
    inputs.clear();
    for (const auto &s : names)
      inputs[s] = {};
    labels.clear();
    confidence.clear();
    stats = {};
    release_cache(device);
  }
  void load() {
    if (!nets.empty())
      return;
    for (const auto &s : names) {
      std::istringstream bytes(blobs.at(s));
      auto m = std::make_shared<torch::jit::Module>(torch::jit::load(bytes, device));
      nets[s] = m;
      for (const auto &p : neural::trainable(*m))
        initial[s].push_back(p.detach().clone());
      generators.emplace(s, neural::generator(device, seed));
    }
  }
  void refit(int64_t count, const RelabelFn &relabel) {
    // Runtime inference deliberately disables autograd. This scope must restore it.
    torch::AutoGradMode enable_grad(true);
    load();
    auto lab =
        torch::tensor(std::vector<int64_t>(labels.begin(), labels.begin() + count), torch::kInt64)
            .to(device);
    auto wt = torch::tensor(std::vector<double>(confidence.begin(), confidence.begin() + count),
                            torch::kFloat32)
                  .to(device);
    if (relabel) {
      auto replacement = relabel(count);
      if (replacement) {
        require(replacement->sizes() == torch::IntArrayRef({count, 3}),
                "relabelled prefix shape differs");
        finite(*replacement, "relabelled prefix");
        lab = replacement->argmax(1).to(device);
        wt = std::get<0>(replacement->max(1)).to(device, torch::kFloat32);
        ++stats.relabels;
      }
    }
    torch::jit::GraphOptimizerEnabledGuard optimizer_guard(false);
    try {
      for (const auto &s : names) {
        auto &m = *nets.at(s);
        auto x =
            torch::stack(std::vector<Tensor>(inputs.at(s).begin(), inputs.at(s).begin() + count))
                .to(device);
        auto parameters = neural::trainable(m);
        for (auto &p : parameters)
          p.requires_grad_(true);
        neural::FineTuneAdam optimizer(parameters, lr);
        for (int64_t step = 0; step < steps; ++step) {
          optimizer.zero_grad();
          neural::zero_stream(m);
          Tensor clean;
          {
            torch::NoGradGuard guard;
            clean = m.forward({x}).toTensor().slice(1, 0, 3);
          }
          neural::zero_stream(m);
          auto augmented =
              m.forward({neural::strong(x, generators.at(s))}).toTensor().slice(1, 0, 3);
          namespace F = torch::nn::functional;
          auto ce = (wt * F::cross_entropy(augmented, lab,
                                           F::CrossEntropyFuncOptions().reduction(torch::kNone)))
                        .sum() /
                    wt.sum().clamp_min(1e-6);
          auto kl = F::kl_div(torch::log_softmax(augmented, 1), torch::softmax(clean, 1),
                              F::KLDivFuncOptions().reduction(torch::kBatchMean));
          auto loss = ce + lambda * kl;
          require(loss.requires_grad(), "fine-tuning gradient unexpectedly disabled");
          loss.backward();
          optimizer.step();
        }
        stats.steps += steps;
        {
          torch::NoGradGuard guard;
          neural::zero_stream(m);
          m.forward({x});
          auto ps = neural::trainable(m);
          for (size_t i = 0; i < ps.size(); ++i)
            stats.maximum_parameter_change =
                std::max(stats.maximum_parameter_change,
                         (ps[i] - initial.at(s)[i]).abs().max().item<double>());
        }
      }
      ++stats.refits;
    } catch (...) {
      release_cache(device);
      throw;
    }
    release_cache(device);
  }
};
FineTune::FineTune(const Params &z, const Tensor &w, torch::Device d,
                   const std::filesystem::path &r)
    : impl_(std::make_unique<Impl>(z, w, d, r)) {}
FineTune::~FineTune() = default;
FineTune::FineTune(const FineTune &x) : impl_(std::make_unique<Impl>(*x.impl_)) {}
FineTune &FineTune::operator=(const FineTune &x) {
  if (this != &x)
    impl_ = std::make_unique<Impl>(*x.impl_);
  return *this;
}
void FineTune::reset() { impl_->reset(); }
std::map<std::string, Tensor> FineTune::step(const NetworkStep &members, const Tensor &label,
                                             const RelabelFn &relabel) {
  auto &s = *impl_;
  check_label(label);
  int64_t count = s.labels.size();
  std::map<std::string, Tensor> result;
  if (!s.on) {
    s.labels.push_back(label.argmax().item<int64_t>());
    s.stats.windows = s.labels.size();
    return result;
  }
  if (count < s.cap)
    for (const auto &name : s.names)
      s.inputs.at(name).push_back(
          members.inputs.at(s.members.at(name)).to(torch::kCPU, torch::kFloat32).clone());
  if (count >= s.start) {
    if ((count - s.start) % s.every == 0) {
      if (count > s.cap)
        ++s.stats.capped;
      else if (s.stats.refit_seconds > s.seconds)
        ++s.stats.over;
      else {
        bool first = s.stats.refits == 0;
        auto begun = std::chrono::steady_clock::now();
        s.refit(count, relabel);
        if (!first)
          s.stats.refit_seconds +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
      }
    }
    s.load();
    torch::NoGradGuard guard;
    for (const auto &name : s.names) {
      auto x = members.inputs.at(s.members.at(name)).unsqueeze(0).to(s.device);
      result[name] =
          torch::softmax(
              s.nets.at(name)->forward({x}).toTensor().slice(1, 0, 3).to(torch::kFloat64), 1)[0]
              .cpu();
    }
  }
  s.labels.push_back(label.argmax().item<int64_t>());
  s.confidence.push_back(label.max().item<double>());
  s.stats.windows = s.labels.size();
  return result;
}
void FineTune::relabel(const std::vector<Tensor> &history) {
  auto &s = *impl_;
  require(history.size() == s.labels.size(), "fine-tuning relabel history differs");
  s.labels.clear();
  s.confidence.clear();
  for (const auto &p : history) {
    check_label(p);
    s.labels.push_back(p.argmax().item<int64_t>());
    s.confidence.push_back(p.max().item<double>());
  }
}
const std::map<std::string, std::vector<int64_t>> &FineTune::slots() const { return impl_->slots; }
const std::map<std::string, int64_t> &FineTune::members() const { return impl_->members; }
FineTuneStats FineTune::stats() const { return impl_->stats; }
} // namespace ni26
