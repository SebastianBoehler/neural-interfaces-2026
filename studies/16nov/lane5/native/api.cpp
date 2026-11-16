#include "api.h"
#include "internal.h"
#include <exception>
#include <stdexcept>

static thread_local std::string error_message;
static Model &model(void *p) {
    if (!p) throw std::invalid_argument("null handle");
    return *static_cast<Model *>(p);
}
static void valid_n(int64_t n) { if (n <= 0) throw std::invalid_argument("n must be positive"); }
#define N5_BEGIN try { error_message.clear();
#define N5_END } catch (const std::exception &e) { error_message = e.what(); return -1; } catch (...) { error_message = "unknown native exception"; return -1; } return 0;

extern "C" const char *n5_error(void) { return error_message.c_str(); }
extern "C" void *n5_create(int arm, int device, int64_t seed) {
    try { error_message.clear(); return new Model(arm, device, seed); }
    catch (const std::exception &e) { error_message = e.what(); return nullptr; }
}
extern "C" void *n5_clone(void *p) {
    try { error_message.clear(); return new Model(model(p)); }
    catch (const std::exception &e) { error_message = e.what(); return nullptr; }
}
extern "C" void n5_destroy(void *p) { delete static_cast<Model *>(p); }

extern "C" int n5_features(const float *raw, int64_t n, float *out) {
    N5_BEGIN valid_n(n); T x = from_float(raw, {n,41,480}, torch::kCPU); finite(x, "nonfinite raw input");
    copy_float(reference_features(x), out); N5_END
}
extern "C" int n5_set_norm(void *p, const float *mu, const float *sd) {
    N5_BEGIN auto &m = model(p); T a = from_float(mu, {246}, m.device), b = from_float(sd, {246}, m.device);
    finite(a, "nonfinite mean"); finite(b, "nonfinite std");
    if ((b < 0).any().item<bool>()) throw std::invalid_argument("negative std");
    m.mu = a; m.sd = b.clamp_min(1e-4); N5_END
}
extern "C" int64_t n5_param_count(void *p) {
    try { error_message.clear(); int64_t n = 0; for (auto &v : model(p).p) n += v.numel(); return n; }
    catch (const std::exception &e) { error_message = e.what(); return -1; }
}
extern "C" int n5_get_params(void *p, float *out) {
    N5_BEGIN auto &m = model(p); if (!out) throw std::invalid_argument("null parameters output");
    int64_t off = 0; for (auto &v : m.p) { copy_float(v, out + off); off += v.numel(); } N5_END
}
extern "C" int n5_set_params(void *p, const float *in, int64_t count) {
    N5_BEGIN auto &m = model(p); if (!in || count != n5_param_count(p)) throw std::invalid_argument("parameter count mismatch");
    T values = from_float(in, {count}, m.device); finite(values, "nonfinite parameters");
    torch::NoGradGuard no_grad; int64_t off = 0;
    for (auto &v : m.p) { v.copy_(values.slice(0, off, off + v.numel()).view(v.sizes())); off += v.numel(); }
    m.anchor.clear(); for (auto &v : m.p) m.anchor.push_back(v.detach().clone()); m.reset(); N5_END
}
extern "C" int n5_head_forward(void *p, const float *x, int64_t n, float *out) {
    N5_BEGIN auto &m = model(p); valid_n(n); if (m.arm != 0) throw std::invalid_argument("head parity requires baseline");
    T v = from_float(x, {n,246}, m.device); finite(v, "nonfinite features"); copy_float(head(m, v), out); N5_END
}
static T loss_inputs(Model &m, const float *x, const int64_t *y, const float *w,
                     const float *mask, int64_t n, T &input) {
    valid_n(n); if (m.arm != 0) throw std::invalid_argument("head parity requires baseline");
    input = from_float(x, {n,246}, m.device).set_requires_grad(true);
    T labels = from_labels(y, n, m.device), weights = from_float(w, {n}, m.device), keep = from_float(mask, {n,246}, m.device);
    finite(input, "nonfinite features"); finite(weights, "nonfinite weights"); finite(keep, "nonfinite mask");
    if ((labels < 0).any().item<bool>() || (labels > 2).any().item<bool>()) throw std::invalid_argument("invalid label");
    return weighted_loss(m, input, labels, weights, keep);
}
extern "C" int n5_head_grads(void *p, const float *x, const int64_t *y, const float *w,
                             const float *mask, int64_t n, float *loss, float *gx, float *gp) {
    N5_BEGIN auto &m = model(p); if (!loss || !gx || !gp) throw std::invalid_argument("null gradient output");
    T input, objective = loss_inputs(m, x, y, w, mask, n, input);
    std::vector<T> vars{input}; vars.insert(vars.end(), m.p.begin(), m.p.end());
    auto grads = torch::autograd::grad({objective}, vars); *loss = objective.item<float>(); copy_float(grads[0], gx);
    int64_t off = 0; for (size_t i = 1; i < grads.size(); ++i) { copy_float(grads[i], gp + off); off += grads[i].numel(); } N5_END
}
extern "C" int n5_adam_step(void *p, const float *x, const int64_t *y, const float *w,
                            const float *mask, int64_t n, double lr, double wd) {
    N5_BEGIN auto &m = model(p); if (lr <= 0 || wd < 0) throw std::invalid_argument("invalid AdamW hyperparameters");
    T input; auto objective = loss_inputs(m, x, y, w, mask, n, input);
    torch::optim::AdamW opt(m.p, torch::optim::AdamWOptions(lr).weight_decay(wd));
    opt.zero_grad(); objective.backward(); opt.step(); N5_END
}
extern "C" int n5_predict(void *p, const float *raw, int64_t n, float *out) {
    N5_BEGIN auto &m = model(p); valid_n(n); T x = from_float(raw, {n,41,480}, m.device); finite(x, "nonfinite raw input");
    torch::NoGradGuard no_grad; copy_float(predict(m, x), out); N5_END
}
extern "C" int n5_fit(void *p, const float *x, const int64_t *y, const int32_t *people,
                      int64_t n, int32_t heldout, int epochs, int consistency) {
    N5_BEGIN fit(model(p), x, y, people, n, heldout, epochs, consistency); N5_END
}
extern "C" int n5_reset(void *p) { N5_BEGIN model(p).reset(); N5_END }
extern "C" int n5_step(void *p, const float *raw, int update, float *out) {
    N5_BEGIN auto &m = model(p); if (update != 0 && update != 1) throw std::invalid_argument("invalid update flag");
    T x = from_float(raw, {1,41,480}, m.device); finite(x, "nonfinite raw input"); copy_float(step(m, x, update), out); N5_END
}
extern "C" int n5_status(void *p, int64_t *windows, int64_t *updates, int64_t *prefix,
                         double *max_grad, double *delta) {
    N5_BEGIN auto &m = model(p); if (!windows || !updates || !prefix || !max_grad || !delta) throw std::invalid_argument("null status output");
    *windows = m.windows; *updates = m.updates; *prefix = m.last_prefix; *max_grad = m.max_grad;
    torch::NoGradGuard no_grad; *delta = ((m.p[2] - m.anchor[2]).square().sum() + (m.p[3] - m.anchor[3]).square().sum()).sqrt().item<double>(); N5_END
}
extern "C" int n5_fit_status(void *p, int64_t *steps, double *consistency_sum,
                             double *projection_grad_max, uint64_t *shuffle_hash) {
    N5_BEGIN auto &m = model(p);
    if (!steps || !consistency_sum || !projection_grad_max || !shuffle_hash) throw std::invalid_argument("null fit status output");
    *steps = m.fit_steps; *consistency_sum = m.consistency_sum;
    *projection_grad_max = m.projection_grad_max; *shuffle_hash = m.shuffle_hash; N5_END
}
extern "C" int n5_consistency_grad(void *p, double *out) {
    N5_BEGIN if (!out) throw std::invalid_argument("null consistency gradient output");
    *out = model(p).consistency_grad; N5_END
}
extern "C" int n5_save(void *p, const char *path) { N5_BEGIN save(model(p), path); N5_END }
extern "C" int n5_load(void *p, const char *path) { N5_BEGIN load(model(p), path); N5_END }
