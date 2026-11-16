#include "internal.h"
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

void fit(Model &m, const float *raw, const int64_t *labels, const int32_t *people,
         int64_t n, int32_t heldout, int epochs, int consistency) {
    if (n < 27 || epochs < 1 || (consistency != 0 && consistency != 1) ||
        (consistency && m.arm != 3) || !people || !labels)
        throw std::invalid_argument("invalid training arguments");
    std::set<int32_t> subjects;
    std::map<std::pair<int32_t,int64_t>, int64_t> counts;
    for (int64_t i = 0; i < n; ++i) {
        if (people[i] == heldout || labels[i] < 0 || labels[i] > 2)
            throw std::invalid_argument("heldout row or invalid label in fit");
        subjects.insert(people[i]); ++counts[{people[i], labels[i]}];
    }
    if (subjects.size() != 9) throw std::invalid_argument("fit requires nine training people");
    for (auto p : subjects) for (int c = 0; c < 3; ++c)
        if (!counts[{p,c}]) throw std::invalid_argument("missing person/class in fit");
    std::vector<float> weights(n); double sum = 0;
    for (int64_t i = 0; i < n; ++i) { weights[i] = 1.0 / counts[{people[i], labels[i]}]; sum += weights[i]; }
    for (auto &w : weights) w *= n / sum;
    T x = from_float(raw, {n, 41, 480}, m.device); finite(x, "nonfinite training input");
    T y = from_labels(labels, n, m.device), w = from_float(weights.data(), {n}, m.device);
    {
        torch::NoGradGuard no_grad;
        if (m.arm == 0) {
            T f = reference_features(x);
            m.mu = f.mean(0); m.sd = f.std(0, false).clamp_min(1e-4);
        } else {
            m.raw_scale = x.square().mean({0, 2}).sqrt().clamp_min(1e-8);
        }
    }
    auto shuffle = at::detail::createCPUGenerator(m.seed + 1000);
    auto augment = at::detail::createCPUGenerator(m.seed + 2000);
    auto dropout = at::detail::createCPUGenerator(m.seed + 3000);
    m.fit_steps = 0; m.consistency_sum = 0; m.consistency_grad = 0; m.projection_grad_max = 0;
    m.shuffle_hash = 14695981039346656037ULL;
    torch::optim::AdamW optim(m.p, torch::optim::AdamWOptions(0.001).weight_decay(0.001));
    for (int epoch = 0; epoch < epochs; ++epoch) {
        T order = at::randperm(n, shuffle, torch::TensorOptions().dtype(torch::kInt64));
        auto bytes = reinterpret_cast<const unsigned char *>(order.data_ptr<int64_t>());
        for (int64_t i = 0; i < n * static_cast<int64_t>(sizeof(int64_t)); ++i) {
            m.shuffle_hash ^= bytes[i]; m.shuffle_hash *= 1099511628211ULL;
        }
        for (int64_t start = 0; start < n; start += 128) {
            T ids = order.slice(0, start, std::min<int64_t>(start + 128, n)).to(m.device);
            T batch = x.index_select(0, ids), by = y.index_select(0, ids), bw = w.index_select(0, ids);
            T f = embedding(m, batch);
            if (m.arm == 0) {
                T mask = (at::rand(f.sizes(), dropout, torch::TensorOptions().dtype(torch::kFloat32)).to(m.device) >= 0.1).to(torch::kFloat32);
                f = f * mask;
            }
            T logits = head(m, f), lp = torch::log_softmax(logits, 1);
            T loss = (-lp.gather(1, by.unsqueeze(1)).squeeze(1) * bw).mean();
            if (consistency) {
                T gain = (at::randn({batch.size(0), 41, 1}, augment, torch::TensorOptions().dtype(torch::kFloat32)) * 0.1).exp().to(m.device);
                T lq = torch::log_softmax(predict(m, batch * gain), 1);
                T kl = ((lp.exp() * (lp - lq) + lq.exp() * (lq - lp)).sum(1) * 0.5).mean();
                m.consistency_sum += kl.detach().item<double>();
                if (m.fit_steps == 0) {
                    auto grads = torch::autograd::grad({0.1 * kl}, m.p, {}, true, false, false);
                    std::vector<T> squares;
                    for (const auto &g : grads) squares.push_back(g.square().sum());
                    m.consistency_grad = torch::stack(squares).sum().sqrt().item<double>();
                    if (!std::isfinite(m.consistency_grad) || m.consistency_grad <= 0)
                        throw std::runtime_error("consistency term has no finite gradient");
                }
                loss = loss + 0.1 * kl;
            }
            optim.zero_grad(); loss.backward();
            std::vector<T> checks{torch::isfinite(loss)};
            for (const auto &v : m.p) {
                if (!v.grad().defined()) throw std::runtime_error("missing parameter gradient");
                checks.push_back(torch::isfinite(v.grad()).all());
            }
            if (!torch::stack(checks).all().item<bool>()) throw std::runtime_error("nonfinite loss or parameter gradient");
            if (m.arm == 2 || m.arm == 3) {
                auto g = m.p.back().grad();
                if (!g.defined()) throw std::runtime_error("connectivity projection has no gradient");
                double norm = g.detach().norm().item<double>();
                if (!std::isfinite(norm)) throw std::runtime_error("nonfinite projection gradient");
                m.projection_grad_max = std::max(m.projection_grad_max, norm);
            }
            optim.step(); ++m.fit_steps;
        }
    }
    for (const auto &v : m.p) finite(v, "nonfinite fitted parameters");
    m.anchor.clear(); for (const auto &v : m.p) m.anchor.push_back(v.detach().clone());
    m.reset();
}
