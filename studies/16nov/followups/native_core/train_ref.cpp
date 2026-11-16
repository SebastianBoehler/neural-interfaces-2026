// Native whole-training-session references, distinct from causal test state.
#include <torch/torch.h>
#include <cmath>
#include <cstring>
#include <string>

using T = torch::Tensor;
static thread_local std::string err;

struct TrainingReference {
    torch::Device device;
    T sum;
    double slow[328] = {};
    int64_t n = 0;
    explicit TrainingReference(int cuda) : device(cuda ? torch::kCUDA : torch::kCPU) {
        sum = torch::zeros({51, 41, 41}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    }
};

static T eigen(const T& x, int mode) {
    auto pair = torch::linalg_eigh(x, "L");
    T w = std::get<0>(pair), v = std::get<1>(pair);
    if (mode == 0) w = w.clamp_min(1e-12).log();
    else if (mode == 1) w = w.exp();
    else w = w.clamp_min(1e-12).rsqrt();
    return torch::matmul(v * w.unsqueeze(-2), v.transpose(-2, -1));
}

extern "C" const char* nt_error() { return err.c_str(); }
extern "C" void* nt_create(int cuda) {
    if (cuda != 0 && cuda != 1) return nullptr;
    try { return new TrainingReference(cuda); }
    catch (const std::exception& e) { err = e.what(); return nullptr; }
}
extern "C" void nt_destroy(void* ptr) { delete static_cast<TrainingReference*>(ptr); }
extern "C" int nt_add(void* ptr, const double* cov, const float* slow) {
    if (!ptr || !cov || !slow) return -1;
    try {
        auto& s = *static_cast<TrainingReference*>(ptr);
        T x = torch::from_blob(const_cast<double*>(cov), {51, 41, 41}, torch::kFloat64).clone().to(s.device);
        if (!torch::isfinite(x).all().item<bool>()) throw std::runtime_error("nonfinite covariance");
        for (int i = 0; i < 328; ++i) if (!std::isfinite(slow[i])) throw std::runtime_error("nonfinite slow feature");
        T next = s.sum + eigen(x, 0);
        s.sum = next; s.n++;
        for (int i = 0; i < 328; ++i) s.slow[i] += slow[i];
        return 0;
    } catch (const std::exception& e) { err = e.what(); return -1; }
}
extern "C" int nt_prior(void* ptr, double* output) {
    if (!ptr || !output) return -1;
    try {
        auto& s = *static_cast<TrainingReference*>(ptr);
        if (!s.n) throw std::runtime_error("empty training reference");
        T out = (s.sum / s.n).to(torch::kCPU).contiguous();
        std::memcpy(output, out.data_ptr<double>(), 51 * 41 * 41 * sizeof(double));
        return 0;
    } catch (const std::exception& e) { err = e.what(); return -1; }
}
extern "C" int nt_feature(void* ptr, const double* cov, const float* slow, double* output) {
    if (!ptr || !cov || !slow || !output) return -1;
    try {
        auto& s = *static_cast<TrainingReference*>(ptr);
        if (!s.n) throw std::runtime_error("empty training reference");
        T x = torch::from_blob(const_cast<double*>(cov), {51, 41, 41}, torch::kFloat64).clone().to(s.device);
        T inv = eigen(eigen(s.sum / s.n, 1), 2);
        T tangent = eigen(torch::matmul(torch::matmul(inv, x), inv), 0).to(torch::kCPU).contiguous();
        const double* values = tangent.data_ptr<double>();
        int i = 0;
        for (int b = 0; b < 51; ++b)
            for (int c = 0; c < 41; ++c)
                for (int d = c; d < 41; ++d)
                    output[i++] = values[(b * 41 + c) * 41 + d] * (c == d ? 1. : std::sqrt(2.));
        for (int j = 0; j < 328; ++j) output[i++] = slow[j] - s.slow[j] / s.n;
        return 0;
    } catch (const std::exception& e) { err = e.what(); return -1; }
}
// Combine ten person-level means without using any held-person observations.
extern "C" int nt_combine(const double* priors, const int64_t* counts, int held, double* output) {
    if (!priors || !counts || !output || held < 0 || held >= 10) return -1;
    constexpr int D = 51 * 41 * 41;
    int64_t total = 0;
    for (int p = 0; p < 10; ++p) {
        if (counts[p] <= 0) return -1;
        if (p != held) total += counts[p];
    }
    if (total <= 0) return -1;
    for (int i = 0; i < D; ++i) {
        double sum = 0;
        for (int p = 0; p < 10; ++p)
            if (p != held) sum += priors[p * D + i] * counts[p];
        output[i] = sum / total;
    }
    return 0;
}
