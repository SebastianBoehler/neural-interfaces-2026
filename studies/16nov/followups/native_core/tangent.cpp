// Double-precision tangent features and session-local causal reference.
#include <torch/torch.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>

using T = torch::Tensor;
static thread_local std::string error;

struct Session {
    torch::Device device;
    T prior, sum;
    double slow_sum[328] = {};
    int64_t n = 0;
    Session(const double* p, int cuda) : device(cuda ? torch::kCUDA : torch::kCPU) {
        prior = torch::from_blob(const_cast<double*>(p), {51, 41, 41}, torch::kFloat64).clone().to(device);
        sum = torch::zeros_like(prior);
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

static void features(const T& cov, const T& logref, const double* slow, double* output) {
    T ref = eigen(logref, 1), inv = eigen(ref, 2);
    T tangent = eigen(torch::matmul(torch::matmul(inv, cov), inv), 0);
    T host = tangent.to(torch::kCPU).contiguous();
    const double* src = host.data_ptr<double>();
    int i = 0;
    for (int b = 0; b < 51; ++b)
        for (int c = 0; c < 41; ++c)
            for (int d = c; d < 41; ++d)
                output[i++] = src[(b * 41 + c) * 41 + d] * (c == d ? 1. : std::sqrt(2.));
    std::memcpy(output + i, slow, 328 * sizeof(double));
}

extern "C" const char* nc_error() { return error.c_str(); }
extern "C" void* nc_create(const double* prior, int cuda) {
    if (!prior || (cuda != 0 && cuda != 1)) return nullptr;
    try { return new Session(prior, cuda); }
    catch (const std::exception& e) { error = e.what(); return nullptr; }
}
extern "C" void nc_destroy(void* ptr) { delete static_cast<Session*>(ptr); }
extern "C" int nc_reset(void* ptr) {
    if (!ptr) return -1;
    try {
        auto& s = *static_cast<Session*>(ptr);
        s.sum.zero_(); std::memset(s.slow_sum, 0, sizeof(s.slow_sum)); s.n = 0;
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
extern "C" int nc_step(void* ptr, const double* input, const float* slow, double* output) {
    if (!ptr || !input || !slow || !output) return -1;
    try {
        auto& s = *static_cast<Session*>(ptr);
        T cov = torch::from_blob(const_cast<double*>(input), {51, 41, 41}, torch::kFloat64).clone().to(s.device);
        if (!torch::isfinite(cov).all().item<bool>()) throw std::runtime_error("nonfinite covariance");
        T sum = s.sum + eigen(cov, 0);
        int64_t n = s.n + 1;
        double centered[328];
        for (int i = 0; i < 328; ++i) {
            if (!std::isfinite(slow[i])) throw std::runtime_error("nonfinite slow feature");
            centered[i] = static_cast<double>(slow[i]) - (s.slow_sum[i] + slow[i]) / (2. + n);
        }
        features(cov, (2. * s.prior + sum) / (2. + n), centered, output);
        s.sum = sum; s.n = n;
        for (int i = 0; i < 328; ++i) s.slow_sum[i] += slow[i];
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
