#include "internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

T from_float(const float *p, std::vector<int64_t> shape, torch::Device device) {
    if (!p) throw std::invalid_argument("null float input");
    return torch::from_blob(const_cast<float *>(p), shape, torch::kFloat32).clone().to(device);
}
T from_labels(const int64_t *p, int64_t n, torch::Device device) {
    if (!p) throw std::invalid_argument("null labels");
    return torch::from_blob(const_cast<int64_t *>(p), {n}, torch::kInt64).clone().to(device);
}
void copy_float(const T &x, float *dst) {
    if (!dst) throw std::invalid_argument("null output");
    T y = x.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    std::memcpy(dst, y.data_ptr<float>(), y.numel() * sizeof(float));
}
void finite(const T &x, const char *what) {
    if (!torch::isfinite(x).all().item<bool>()) throw std::invalid_argument(what);
}

static T weights(at::Generator gen, std::vector<int64_t> shape, int64_t fan, torch::Device dev) {
    auto v = (at::rand(shape, gen, torch::TensorOptions().dtype(torch::kFloat32)) * 2 - 1) / std::sqrt(double(fan));
    return v.to(dev).set_requires_grad(true);
}

Model::Model(int a, int d, int64_t s)
    : arm(a), device(d == 0 ? torch::Device(torch::kCPU) : torch::Device(torch::kCUDA)), seed(s) {
    if (a < 0 || a > 3 || (d != 0 && d != 1)) throw std::invalid_argument("invalid arm/device");
    auto gen = at::detail::createCPUGenerator(seed);
    int in = a == 0 ? 246 : a == 1 ? 89 : a == 2 ? 221 : 269;
    p.push_back(weights(gen, {64, in}, in, device)); p.push_back(weights(gen, {64}, in, device));
    p.push_back(weights(gen, {3, 64}, 64, device)); p.push_back(weights(gen, {3}, 64, device));
    if (a == 1 || a == 3) {
        p.push_back(weights(gen, {16, 41, 1}, 41, device)); p.push_back(weights(gen, {16}, 41, device));
        for (int k : {15, 31, 63}) {
            p.push_back(weights(gen, {16, 1, k}, k, device)); p.push_back(weights(gen, {16}, k, device));
        }
    }
    if (a == 2 || a == 3) p.push_back(weights(gen, {8, 41}, 41, device));
    mu = torch::zeros({246}, torch::TensorOptions().device(device));
    sd = torch::ones({246}, torch::TensorOptions().device(device));
    raw_scale = torch::ones({41}, torch::TensorOptions().device(device));
    for (const auto &v : p) anchor.push_back(v.detach().clone());
}

Model::Model(const Model &o) : arm(o.arm), device(o.device), seed(o.seed) {
    for (const auto &v : o.p) p.push_back(v.detach().clone().set_requires_grad(true));
    for (const auto &v : o.anchor) anchor.push_back(v.clone());
    mu = o.mu.clone(); sd = o.sd.clone(); raw_scale = o.raw_scale.clone();
    for (const auto &v : o.prefix) prefix.push_back(v.clone());
    windows = o.windows; updates = o.updates; last_prefix = o.last_prefix; max_grad = o.max_grad;
    fit_steps = o.fit_steps; consistency_sum = o.consistency_sum; consistency_grad = o.consistency_grad;
    projection_grad_max = o.projection_grad_max; shuffle_hash = o.shuffle_hash;
}
void Model::reset() {
    torch::NoGradGuard guard;
    for (size_t i = 0; i < p.size(); ++i) p[i].copy_(anchor[i]);
    prefix.clear(); windows = updates = last_prefix = 0; max_grad = 0;
}

T reference_features(const T &raw) {
    auto x = raw.to(torch::kFloat64);
    auto mean = x.mean(-1);
    auto f = at::fft_rfft(x - mean.unsqueeze(-1), 480, -1);
    auto power = f.abs().square() / (480.0 * 480.0);
    std::vector<T> out{mean};
    for (auto b : {std::pair<int,int>{4,8}, {8,13}, {13,20}, {20,30}, {30,45}})
        out.push_back(power.slice(-1, b.first * 4, b.second * 4).mean(-1).clamp_min(1e-20).log());
    return torch::cat(out, 1).to(torch::kFloat32);
}

T head(Model &m, const T &x) {
    return at::linear(at::gelu(at::linear(x, m.p[0], m.p[1]), "none"), m.p[2], m.p[3]);
}

T embedding(Model &m, const T &raw) {
    if (m.arm == 0) return ((reference_features(raw) - m.mu) / m.sd).clamp(-8, 8);
    T x = raw / m.raw_scale.view({1, 41, 1});
    T means = x.mean(-1);
    std::vector<T> pieces;
    if (m.arm == 1 || m.arm == 3) {
        T spatial = at::conv1d(x, m.p[4], m.p[5], {1}, at::IntArrayRef({0}), {1}, 1);
        for (int j = 0; j < 3; ++j) {
            int k = 15 + (j == 1 ? 16 : j == 2 ? 48 : 0);
            T z = at::conv1d(spatial, m.p[6 + 2*j], m.p[7 + 2*j], {1}, at::IntArrayRef({k/2}), {1}, 16);
            pieces.push_back(z.square().mean(-1).clamp_min(1e-8).log());
        }
    }
    if (m.arm == 2 || m.arm == 3) {
        int wi = m.arm == 2 ? 4 : 12;
        T centered = x - means.unsqueeze(-1);
        T projected = torch::matmul(m.p[wi], centered);
        T f = at::fft_rfft(projected, 480, -1);
        std::vector<int64_t> ids;
        for (int i = 0; i < 8; ++i) for (int j = i; j < 8; ++j) ids.push_back(i * 8 + j);
        T index = torch::tensor(ids, torch::TensorOptions().dtype(torch::kInt64).device(m.device));
        T eye = torch::eye(8, torch::TensorOptions().device(m.device));
        for (auto b : {std::pair<int,int>{4,8}, {8,13}, {13,20}, {20,30}, {30,45}}) {
            T fb = f.slice(-1, b.first * 4, b.second * 4);
            T re = torch::real(fb), im = torch::imag(fb);
            T c = (torch::matmul(re, re.transpose(1,2)) + torch::matmul(im, im.transpose(1,2))) /
                  (480.0 * 480.0 * (b.second - b.first) * 4);
            T diag = c.diagonal(0, 1, 2).clamp_min(1e-8);
            T corr = c / torch::sqrt(diag.unsqueeze(2) * diag.unsqueeze(1));
            T v = corr * (1 - eye) + torch::diag_embed(diag.log());
            pieces.push_back(v.flatten(1).index_select(1, index));
        }
    }
    pieces.push_back(means);
    return torch::cat(pieces, 1);
}
T predict(Model &m, const T &raw) { return head(m, embedding(m, raw)); }
T weighted_loss(Model &m, const T &x, const T &y, const T &w, const T &mask) {
    auto logp = torch::log_softmax(head(m, x * mask), 1);
    return (-logp.gather(1, y.unsqueeze(1)).squeeze(1) * w).mean();
}
