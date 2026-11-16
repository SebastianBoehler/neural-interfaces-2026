#include "internal.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <torch/serialize.h>
#include <unistd.h>

T step(Model &m, const T &raw, bool update) {
    if (m.windows >= 40 && m.windows % 20 == 0 && update) {
        if (m.prefix.empty()) throw std::logic_error("empty adaptation prefix");
        T encoded;
        {
            torch::NoGradGuard no_grad;
            std::vector<T> rows(m.prefix.begin(), m.prefix.end());
            encoded = embedding(m, torch::cat(rows, 0)).detach();
        }
        m.last_prefix = static_cast<int64_t>(m.prefix.size());
        for (int iter = 0; iter < 3; ++iter) {
            T logp = torch::log_softmax(head(m, encoded), 1), p = logp.exp();
            T entropy = -(p * logp).sum(1).mean();
            T mean = p.mean(0).clamp_min(1e-12);
            T diversity = -(mean * mean.log()).sum();
            T penalty = (m.p[2] - m.anchor[2]).square().sum() + (m.p[3] - m.anchor[3]).square().sum();
            T loss = entropy - 0.1 * diversity + 0.1 * penalty;
            auto grads = torch::autograd::grad({loss}, {m.p[2], m.p[3]});
            double gn = torch::sqrt(grads[0].square().sum() + grads[1].square().sum()).item<double>();
            if (!std::isfinite(gn)) throw std::runtime_error("nonfinite adaptation gradient");
            m.max_grad = std::max(m.max_grad, gn);
            torch::NoGradGuard no_grad;
            for (int j = 0; j < 2; ++j) m.p[j + 2].sub_(0.001 * grads[j]);
            T delta = (m.p[2] - m.anchor[2]).square().sum() + (m.p[3] - m.anchor[3]).square().sum();
            double norm = delta.sqrt().item<double>();
            double scale = std::min(1.0, 0.1 / std::max(norm, 1e-12));
            for (int j = 2; j < 4; ++j) m.p[j].copy_(m.anchor[j] + scale * (m.p[j] - m.anchor[j]));
        }
        ++m.updates;
    }
    T result;
    { torch::NoGradGuard no_grad; result = predict(m, raw).detach(); }
    m.prefix.push_back(raw.detach().clone());
    if (m.prefix.size() > 64) m.prefix.pop_front();
    ++m.windows;
    return result;
}

void save(Model &m, const char *path) {
    if (!path || !*path) throw std::invalid_argument("empty checkpoint path");
    torch::serialize::OutputArchive archive;
    archive.write("arm", torch::tensor(m.arm)); archive.write("seed", torch::tensor(m.seed));
    archive.write("mu", m.mu); archive.write("sd", m.sd); archive.write("raw_scale", m.raw_scale);
    for (size_t i = 0; i < m.anchor.size(); ++i) archive.write("p" + std::to_string(i), m.anchor[i]);
    std::filesystem::path target(path);
    auto parent = target.has_parent_path() ? target.parent_path() : std::filesystem::path(".");
    std::string pattern = (parent / "lane5-checkpoint-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
    int fd = ::mkstemp(name.data());
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "checkpoint mkstemp");
    struct Cleanup { const char *name; ~Cleanup() { ::unlink(name); } } cleanup{name.data()};
    if (::close(fd) != 0) throw std::system_error(errno, std::generic_category(), "checkpoint close");
    archive.save_to(name.data());
    if (::link(name.data(), path) != 0) throw std::system_error(errno, std::generic_category(), "checkpoint publish");
}

void load(Model &m, const char *path) {
    if (!path) throw std::invalid_argument("null checkpoint path");
    torch::serialize::InputArchive archive; archive.load_from(path, m.device);
    T arm, seed, mu, sd, scale;
    archive.read("arm", arm); archive.read("seed", seed);
    if (arm.item<int>() != m.arm) throw std::invalid_argument("checkpoint arm differs");
    archive.read("mu", mu); archive.read("sd", sd); archive.read("raw_scale", scale);
    if (mu.sizes() != m.mu.sizes() || sd.sizes() != m.sd.sizes() || scale.sizes() != m.raw_scale.sizes())
        throw std::invalid_argument("checkpoint normalization shape differs");
    finite(mu, "checkpoint mu"); finite(sd, "checkpoint sd"); finite(scale, "checkpoint scale");
    if ((sd <= 0).any().item<bool>() || (scale <= 0).any().item<bool>())
        throw std::invalid_argument("checkpoint normalization nonpositive");
    std::vector<T> next;
    for (size_t i = 0; i < m.p.size(); ++i) {
        T v; archive.read("p" + std::to_string(i), v);
        if (v.sizes() != m.p[i].sizes()) throw std::invalid_argument("checkpoint parameter shape differs");
        finite(v, "checkpoint parameter nonfinite"); next.push_back(v.detach().clone());
    }
    m.seed = seed.item<int64_t>(); m.mu = mu; m.sd = sd; m.raw_scale = scale; m.anchor = std::move(next); m.reset();
}
