#include "internal.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
struct Repair {
    Model model;
    const int mode;
    int64_t attempts = 0, skips = 0, selected_total = 0;
    std::vector<int64_t> selections;
    double max_grad = 0.0;
    Repair(const Model &source, int policy) : model(source), mode(policy) {
        if ((mode != 1 && mode != 2) || source.windows != 0 || !source.prefix.empty())
            throw std::invalid_argument("repair requires a fresh population handle and fixed policy");
        if (source.p.size() != source.anchor.size()) throw std::invalid_argument("anchor size mismatch");
        for (size_t i = 0; i < source.p.size(); ++i)
            if (!torch::equal(source.p[i], source.anchor[i])) throw std::invalid_argument("population is not anchored");
    }
};
Repair &handle(void *p) {
    if (!p) throw std::invalid_argument("null repair handle");
    return *static_cast<Repair *>(p);
}
T teacher(Model &m, const T &encoded) {
    T hidden = at::gelu(at::linear(encoded, m.anchor[0], m.anchor[1]), "none");
    return at::linear(hidden, m.anchor[2], m.anchor[3]);
}
double distance(Model &m) {
    torch::NoGradGuard guard;
    return ((m.p[2] - m.anchor[2]).square().sum() + (m.p[3] - m.anchor[3]).square().sum()).sqrt().item<double>();
}
void update(Repair &r) {
    Model &m = r.model;
    if (m.prefix.empty()) throw std::logic_error("empty adaptation prefix");
    T encoded, logq, q, chosen;
    {
        torch::NoGradGuard guard;
        std::vector<T> rows(m.prefix.begin(), m.prefix.end());
        encoded = embedding(m, torch::cat(rows, 0)).detach();
        logq = torch::log_softmax(teacher(m, encoded), 1).detach();
        q = logq.exp();
        chosen = std::get<0>(q.max(1)) >= 0.70;
    }
    int64_t count = chosen.sum().item<int64_t>();
    ++r.attempts; r.selected_total += count; r.selections.push_back(count);
    m.last_prefix = static_cast<int64_t>(m.prefix.size());
    if (count < 8) { ++r.skips; return; }
    T ids = torch::nonzero(chosen).squeeze(1);
    encoded = encoded.index_select(0, ids); q = q.index_select(0, ids); logq = logq.index_select(0, ids);
    for (int iter = 0; iter < 3; ++iter) {
        T logp = torch::log_softmax(head(m, encoded), 1), p = logp.exp();
        T entropy = -(p * logp).sum(1).mean();
        T mean = p.mean(0).clamp_min(1e-12);
        T diversity = -(mean * mean.log()).sum();
        T kl = (q * (logq - logp)).sum(1).mean();
        T penalty = (m.p[2] - m.anchor[2]).square().sum() + (m.p[3] - m.anchor[3]).square().sum();
        T loss = (r.mode == 1 ? entropy : -entropy) - 0.1 * diversity + kl + 0.3 * penalty;
        auto grad = torch::autograd::grad({loss}, {m.p[2], m.p[3]});
        double norm = (grad[0].square().sum() + grad[1].square().sum()).sqrt().item<double>();
        if (!std::isfinite(norm)) throw std::runtime_error("nonfinite repair gradient");
        r.max_grad = std::max(r.max_grad, norm);
        torch::NoGradGuard guard;
        for (int j = 0; j < 2; ++j) m.p[j + 2].sub_(0.001 * grad[j]);
        double length = distance(m);
        double scale = std::min(1.0, 0.03 / std::max(length, 1e-12));
        for (int j = 2; j < 4; ++j) m.p[j].copy_(m.anchor[j] + scale * (m.p[j] - m.anchor[j]));
    }
    ++m.updates;
}
T advance(Repair &r, const T &raw) {
    Model &m = r.model;
    if (m.windows >= 40 && m.windows % 20 == 0) update(r);
    T result;
    { torch::NoGradGuard guard; result = predict(m, raw).detach(); }
    m.prefix.push_back(raw.detach().clone());
    if (m.prefix.size() > 64) m.prefix.pop_front();
    ++m.windows;
    return result;
}
}

#define N5R_BEGIN try { error.clear();
#define N5R_END return 0; } catch (const std::exception &e) { error = e.what(); return -1; }
extern "C" const char *n5r_error() { return error.c_str(); }
extern "C" int n5r_abi() { return 1; }
extern "C" void *n5r_create(void *population, int mode) {
    try { error.clear(); if (!population) throw std::invalid_argument("null population");
        return new Repair(*static_cast<Model *>(population), mode);
    } catch (const std::exception &e) { error = e.what(); return nullptr; }
}
extern "C" void n5r_destroy(void *p) { delete static_cast<Repair *>(p); }
extern "C" int n5r_reset(void *p) {
    N5R_BEGIN auto &r = handle(p); r.model.reset(); r.attempts = r.skips = r.selected_total = 0;
    r.selections.clear(); r.max_grad = 0; N5R_END
}
extern "C" int n5r_step(void *p, const float *raw, float *out) {
    N5R_BEGIN auto &r = handle(p); T x = from_float(raw, {1, 41, 480}, r.model.device);
    finite(x, "nonfinite repair window"); copy_float(advance(r, x), out); N5R_END
}
extern "C" int n5r_teacher(void *p, const float *raw, float *out) {
    N5R_BEGIN auto &r = handle(p); T x = from_float(raw, {1, 41, 480}, r.model.device);
    finite(x, "nonfinite teacher window"); torch::NoGradGuard guard;
    copy_float(teacher(r.model, embedding(r.model, x)), out); N5R_END
}
extern "C" int n5r_status(void *p, int64_t *windows, int64_t *attempts, int64_t *updates,
                            int64_t *skips, int64_t *selected, double *max_grad, double *delta) {
    N5R_BEGIN auto &r = handle(p);
    if (!windows || !attempts || !updates || !skips || !selected || !max_grad || !delta)
        throw std::invalid_argument("null status output");
    *windows = r.model.windows; *attempts = r.attempts; *updates = r.model.updates;
    *skips = r.skips; *selected = r.selected_total; *max_grad = r.max_grad; *delta = distance(r.model); N5R_END
}
extern "C" int n5r_selections(void *p, int64_t *out, int64_t count) {
    N5R_BEGIN auto &r = handle(p);
    if (count != static_cast<int64_t>(r.selections.size()) || (count && !out))
        throw std::invalid_argument("selection count mismatch");
    std::copy(r.selections.begin(), r.selections.end(), out); N5R_END
}
