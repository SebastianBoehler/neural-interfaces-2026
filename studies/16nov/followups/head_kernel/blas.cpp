// Explicit LP64 system OpenBLAS kernel; no runtime fallback or data copies.
#include <torch/torch.h>
#include <torch/optim/lbfgs.h>
#include <dlfcn.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
using Gemm = void (*)(int,int,int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
using Config = const char* (*)();
using SetThreads = void (*)(int);
using GetThreads = int (*)();
struct Blas {
    void* handle;
    Gemm gemm;
    Config config;
    GetThreads threads;
    std::string path;
    Blas() {
        handle = dlopen("/usr/lib64/libopenblasp.so.0", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (!handle) throw std::runtime_error(dlerror());
        gemm = reinterpret_cast<Gemm>(dlsym(handle, "cblas_dgemm"));
        config = reinterpret_cast<Config>(dlsym(handle, "openblas_get_config"));
        threads = reinterpret_cast<GetThreads>(dlsym(handle, "openblas_get_num_threads"));
        auto set = reinterpret_cast<SetThreads>(dlsym(handle, "openblas_set_num_threads"));
        if (!gemm || !config || !threads || !set) throw std::runtime_error("missing OpenBLAS LP64 symbol");
        std::string c(config());
        if (c.find("USE64BITINT") != std::string::npos || c.find("SINGLE_THREADED") != std::string::npos)
            throw std::runtime_error("unexpected OpenBLAS ABI or single-threaded library");
        set(6);
        if (threads() != 6) throw std::runtime_error("OpenBLAS did not set six threads");
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void*>(gemm), &info) || !info.dli_fname) throw std::runtime_error("BLAS path unavailable");
        path = info.dli_fname;
    }
};
Blas& blas() { static Blas instance; return instance; }
void require(bool ok, const char* why) { if (!ok) throw std::invalid_argument(why); }

void validate(const double* x, const int64_t* y, int64_t n, int64_t d, double c) {
    require(x && y && n > 0 && d > 0 && n <= INT_MAX && d < INT_MAX && std::isfinite(c) && c > 0, "invalid input");
    for (int64_t i = 0; i < n * d; ++i) require(std::isfinite(x[i]), "nonfinite matrix");
    for (int64_t i = 0; i < n; ++i) require(y[i] >= 0 && y[i] < 3, "invalid labels");
}

double compute(const double* x, const int64_t* y, int n, int d, const double* theta,
               double c, double* grad) {
    auto& b = blas();
    std::vector<double> residual(static_cast<size_t>(n) * 3);
    // Row-major X[n,d] × weights[3,d]^T; theta's last column is the intercept.
    b.gemm(101, 111, 112, n, 3, d, 1.0, x, d, theta, d + 1, 0.0, residual.data(), 3);
    double loss = 0.0, bias_grad[3]{};
    for (int i = 0; i < n; ++i) {
        double z[3], mx = -INFINITY, denom = 0.0;
        for (int k = 0; k < 3; ++k) { z[k] = residual[3*i+k] + theta[k*(d+1)+d]; mx = std::max(mx, z[k]); }
        for (double v : z) denom += std::exp(v - mx);
        loss += mx + std::log(denom) - z[y[i]];
        for (int k = 0; k < 3; ++k) {
            double v = (std::exp(z[k] - mx) / denom - (y[i] == k)) / n;
            residual[3*i+k] = v;
            bias_grad[k] += v;
        }
    }
    b.gemm(101, 112, 111, 3, d, n, 1.0, residual.data(), 3, x, d, 0.0, grad, d + 1);
    double penalty = 0.0;
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < d; ++j) {
            double w = theta[k*(d+1)+j];
            penalty += w*w;
            grad[k*(d+1)+j] += w / (c*n);
        }
        grad[k*(d+1)+d] = bias_grad[k];
    }
    return loss / n + penalty / (2*c*n);
}

template<class F> int checked(F f) {
    try { error.clear(); f(); return 0; }
    catch (const std::exception& e) { error = e.what(); return -1; }
    catch (...) { error = "unknown BLAS kernel error"; return -1; }
}
}

extern "C" const char* s311b_error() { return error.c_str(); }
extern "C" const char* s311b_config() { try { return blas().config(); } catch (...) { return "unavailable"; } }
extern "C" const char* s311b_path() { try { return blas().path.c_str(); } catch (...) { return "unavailable"; } }
extern "C" int s311b_threads() { try { return blas().threads(); } catch (...) { return -1; } }

extern "C" int s311b_objective(const double* x, const int64_t* y, int64_t n, int64_t d,
    const double* theta, double c, double* loss, double* gradient) {
    return checked([&] {
        validate(x, y, n, d, c);
        require(theta && loss && gradient, "missing output");
        for (int64_t i = 0; i < 3*(d+1); ++i) require(std::isfinite(theta[i]), "nonfinite theta");
        *loss = compute(x, y, static_cast<int>(n), static_cast<int>(d), theta, c, gradient);
        require(std::isfinite(*loss), "nonfinite loss");
        for (int64_t i = 0; i < 3*(d+1); ++i) require(std::isfinite(gradient[i]), "nonfinite gradient");
    });
}

extern "C" int s311b_fit(const double* x, const int64_t* y, int64_t n, int64_t d, double c,
    double* result, double* loss, double* maximum, int64_t* calls) {
    return checked([&] {
        validate(x, y, n, d, c);
        require(result && loss && maximum && calls, "missing fit output");
        bool seen[3]{};
        for (int64_t i = 0; i < n; ++i) seen[y[i]] = true;
        require(seen[0] && seen[1] && seen[2], "training requires three classes");
        auto opt = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
        auto theta = torch::zeros({3, d+1}, opt.requires_grad(true));
        torch::optim::LBFGS optimizer({theta}, torch::optim::LBFGSOptions(1.0)
            .max_iter(2000).max_eval(2500).history_size(10)
            .tolerance_grad(1e-8).tolerance_change(1e-12).line_search_fn("strong_wolfe"));
        *calls = 0;
        auto closure = [&]() {
            optimizer.zero_grad();
            auto grad = torch::empty({3, d+1}, opt);
            double value = compute(x, y, static_cast<int>(n), static_cast<int>(d), theta.data_ptr<double>(), c, grad.data_ptr<double>());
            theta.mutable_grad() = grad;
            ++*calls;
            return torch::tensor(value, opt);
        };
        optimizer.step(closure);
        *loss = closure().item<double>();
        *maximum = theta.grad().abs().max().item<double>();
        require(std::isfinite(*loss) && std::isfinite(*maximum) && *maximum <= 1e-5, "BLAS fit did not converge");
        std::memcpy(result, theta.data_ptr<double>(), 3*(d+1)*sizeof(double));
    });
}
