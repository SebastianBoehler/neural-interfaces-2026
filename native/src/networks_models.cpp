#include "networks_internal.h"
#include <ATen/ops/fft_irfft.h>
#include <ATen/ops/fft_rfft.h>
#include <iomanip>
#include <openssl/evp.h>
#include <sstream>
#include <torch/torch.h>

namespace ni26::neural {
namespace {
Tensor field(const Params &p, const std::string &key, torch::Device dev, torch::ScalarType dtype) {
  return p.get(key).to(dev, dtype);
}
struct Shallow final : Network {
  Params t;
  Tensor weight, bias;
  Shallow(const Params &p, torch::Device dev) {
    for (const auto *name :
         {"time_w", "time_b", "spat_w", "bn_mean", "bn_var", "bn_w", "bn_b", "cls_w", "cls_b"})
      t.tensors[name] = p.get(name).to(dev);
    weight = (t.get("time_w") * t.get("spat_w").permute({1, 0, 2, 3})).sum(0).unsqueeze(1);
    bias =
        torch::matmul(t.get("spat_w").squeeze().sum(-1), t.get("time_b").unsqueeze(-1)).squeeze();
  }
  Tensor proba(const Tensor &x, WindowCache &) override {
    namespace F = torch::nn::functional;
    auto h = F::conv2d(x.unsqueeze(1).transpose(2, 3), weight, F::Conv2dFuncOptions().bias(bias));
    h = F::batch_norm(h, t.get("bn_mean"), t.get("bn_var"),
                      F::BatchNormFuncOptions()
                          .weight(t.get("bn_w"))
                          .bias(t.get("bn_b"))
                          .training(false)
                          .momentum(0)
                          .eps(1e-5));
    h = F::avg_pool2d(h.square(), F::AvgPool2dFuncOptions({75, 1}).stride({15, 1}))
            .clamp_min(1e-6)
            .log();
    h = F::conv2d(h, t.get("cls_w"), F::Conv2dFuncOptions().bias(t.get("cls_b")))
            .select(2, 0)
            .select(2, 0);
    return torch::softmax(h.to(torch::kFloat64), 1);
  }
  std::shared_ptr<Network> clone() const override { return std::make_shared<Shallow>(*this); }
};
struct FBC final : Network {
  Params t;
  std::vector<Tensor> fir, spectra;
  std::string key;
  int64_t stride;
  FBC(const Params &p, torch::Device dev) : stride(p.integer("fbc_stride")) {
    for (const auto &[k, v] : p.tensors)
      if (k.rfind("fbc_", 0) == 0)
        t.tensors[k] = v.to(dev, torch::kFloat64);
    auto bands = p.integer("fbc_n_bands");
    require(stride > 0 && 480 % stride == 0, "unsupported FBC stride");
    for (int64_t i = 0; i < bands; ++i) {
      auto f = t.get("fbc_fir" + std::to_string(i)).contiguous();
      fir.push_back(f);
      spectra.push_back(at::fft_rfft(f, 480 + f.numel() - 1));
      auto host = f.cpu();
      key.append(static_cast<const char *>(host.const_data_ptr()), host.nbytes());
    }
  }
  torch::ScalarType dtype() const override { return torch::kFloat64; }
  Tensor proba(const Tensor &x, WindowCache &cache) override {
    if (!cache.banks.count(key)) {
      std::vector<Tensor> y;
      for (size_t i = 0; i < fir.size(); ++i) {
        auto n = 480 + fir[i].numel() - 1;
        if (!cache.spectra.count(n))
          cache.spectra[n] = at::fft_rfft(x, n);
        auto start = (n - 480) / 2;
        y.push_back(at::fft_irfft(cache.spectra[n] * spectra[i], n).slice(-1, start, start + 480));
      }
      cache.banks[key] = torch::stack(y, 1);
    }
    auto y = cache.banks.at(key);
    auto k = y.size(1), c = y.size(2);
    auto z = torch::einsum("kfc,bkct->bkft", {t.get("fbc_W").view({k, -1, c}), y});
    z = z.reshape({x.size(0), -1, 480}) + t.get("fbc_bw").view({1, -1, 1});
    z = (z - t.get("fbc_rm").view({1, -1, 1})) /
        torch::sqrt(t.get("fbc_rv").view({1, -1, 1}) + t.get("fbc_eps"));
    z = torch::silu(z * t.get("fbc_g").view({1, -1, 1}) + t.get("fbc_be").view({1, -1, 1}));
    auto v = z.view({z.size(0), z.size(1), stride, 480 / stride})
                 .var(std::vector<int64_t>{3}, true, false)
                 .clamp(1e-6, 1e6)
                 .log()
                 .flatten(1);
    return torch::softmax(torch::matmul(v, t.get("fbc_L").t()) + t.get("fbc_lb"), 1);
  }
  std::shared_ptr<Network> clone() const override { return std::make_shared<FBC>(*this); }
};
std::string fir_digest(const torch::jit::Module &module) {
  auto bank = module.attr("spectral_filtering").toModule();
  auto filters = bank.attr("fir_list").toModule();
  const std::vector<int64_t> lengths{199, 199, 133, 99, 81, 67, 57, 51, 45};
  size_t i = 0;
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  require(bool(ctx), "cannot allocate FIR fingerprint context");
  require(EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1,
          "cannot initialize FIR fingerprint");
  for (const auto &p : filters.named_parameters(false)) {
    require(i < lengths.size() && p.name == std::to_string(i) &&
                p.value.scalar_type() == torch::kFloat32 &&
                p.value.sizes() == torch::IntArrayRef({lengths[i]}),
            "shared FIR parameter contract differs");
    auto x = p.value.detach().cpu().contiguous();
    require(EVP_DigestUpdate(ctx.get(), x.const_data_ptr(), x.nbytes()) == 1,
            "cannot hash FIR coefficients");
    ++i;
  }
  require(i == lengths.size(), "missing shared FIR filters");
  unsigned char data[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  require(EVP_DigestFinal_ex(ctx.get(), data, &length) == 1 && length == 32,
          "cannot finalize FIR fingerprint");
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (unsigned int j = 0; j < length; ++j)
    out << std::setw(2) << int(data[j]);
  return out.str();
}
struct Jit final : Network {
  std::shared_ptr<torch::jit::Module> module;
  bool stateful, shared = false, light = false;
  Jit(const Params &p, torch::Device dev, const std::filesystem::path &root)
      : stateful(p.has("sjit_file")) {
    auto file = p.text(stateful ? "sjit_file" : "jit_file");
    module = std::make_shared<torch::jit::Module>(torch::jit::load((root / file).string(), dev));
    module->eval();
    for (int seed = 0; seed < 6; ++seed) {
      auto s = "_s" + std::to_string(seed) + ".pt";
      if (file == "sbn_k2" + s || file == "sbn_k5" + s || file == "sbn_fbcnet_raw_k20" + s)
        shared = true;
      if (file == "sbn_fblight_ea_rot05_k5" + s)
        shared = light = true;
    }
    if (shared) {
      auto expected = light ? "FBLightConvNet" : "FBCNet";
      require(module->type()->name()->name() == expected, "shared FIR module architecture differs");
      require(fir_digest(*module) ==
                  "607a44ee25576803092c03d77feb67411fe7a2266475fc730ae7ae28c81506c6",
              "shared FIR fingerprint differs");
    }
  }
  void reset() override {
    if (stateful)
      zero_stream(*module);
  }
  Tensor proba(const Tensor &x, WindowCache &cache) override {
    Tensor logits;
    if (!shared)
      logits = module->forward({x}).toTensor();
    else {
      auto call = [&](const char *name, const Tensor &value) {
        return module->attr(name).toModule().forward({value}).toTensor();
      };
      if (!cache.shared_sbn.defined())
        cache.shared_sbn = call("spectral_filtering", x);
      auto y = cache.shared_sbn;
      auto h = call("spatial_conv", y);
      h = light ? call("attn_conv", call("temporal_layer",
                                         h.slice(-1, 0, 250).reshape({y.size(0), 32, -1, 250})))
                : call("temporal_layer", h.view({h.size(0), h.size(1), 4, 120}));
      logits = call("final_layer", call("flatten_layer", h));
    }
    return torch::softmax(logits.to(torch::kFloat64), 1);
  }
  std::shared_ptr<Network> clone() const override {
    auto out = std::make_shared<Jit>(*this);
    if (stateful)
      out->module = std::make_shared<torch::jit::Module>(module->clone());
    return out;
  }
};
} // namespace
std::shared_ptr<Network> build(const Params &p, torch::Device dev,
                               const std::filesystem::path &root) {
  if (p.has("sjit_file") || p.has("jit_file"))
    return std::make_shared<Jit>(p, dev, root);
  if (p.has("fbc_n_bands") && !p.has("fbl_win"))
    return std::make_shared<FBC>(p, dev);
  if (p.has("time_w"))
    return std::make_shared<Shallow>(p, dev);
  throw std::runtime_error("unsupported neural member export");
}
} // namespace ni26::neural
