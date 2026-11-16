// Independent closed-form multinomial gradient; CPU float64 only.
#include <torch/torch.h>
#include <torch/optim/lbfgs.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
const auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

torch::Tensor matrix(const double* data, int64_t n, int64_t d) {
    require(data && n > 0 && d > 0, "missing matrix");
    auto x = torch::from_blob(const_cast<double*>(data), {n, d}, opts);
    require(torch::isfinite(x).all().item<bool>(), "nonfinite matrix");
    return x;
}

torch::Tensor labels(const int64_t* data, int64_t n) {
    require(data != nullptr, "missing labels");
    auto y = torch::from_blob(const_cast<int64_t*>(data), {n}, torch::kInt64);
    require(y.min().item<int64_t>() >= 0 && y.max().item<int64_t>() < 3, "invalid labels");
    return y;
}

torch::Tensor analytical(const torch::Tensor& x, const torch::Tensor& y,
                         const torch::Tensor& theta, double c, torch::Tensor& grad) {
    torch::NoGradGuard guard;
    const int64_t n = x.size(0), d = x.size(1);
    auto weights = theta.slice(1, 0, d);
    auto logits = torch::matmul(x, weights.t()) + theta.select(1, d);
    auto loss = (torch::logsumexp(logits, 1) - logits.gather(1, y.unsqueeze(1)).squeeze(1)).mean()
        + weights.square().sum() / (2 * c * n);
    auto residual = torch::softmax(logits, 1);
    residual.scatter_add_(1, y.unsqueeze(1), -torch::ones({n, 1}, opts));
    residual.div_(n);
    auto dw = torch::matmul(residual.t(), x) + weights / (c * n);
    auto db = residual.sum(0).unsqueeze(1);
    grad = torch::cat({dw, db}, 1).contiguous();
    return loss;
}

template<class F> int checked(F f) {
    try { error.clear(); f(); return 0; }
    catch (const std::exception& e) { error = e.what(); return -1; }
    catch (...) { error = "unknown analytical kernel error"; return -1; }
}
}

extern "C" const char* s311_error() { return error.c_str(); }

extern "C" int s311_objective(const double* data, const int64_t* ys, int64_t n,
    int64_t d, const double* params, double c, double* loss, double* gradient) {
    return checked([&] {
        require(loss && gradient && std::isfinite(c) && c > 0, "invalid objective outputs or C");
        auto x = matrix(data, n, d), y = labels(ys, n), theta = matrix(params, 3, d + 1);
        torch::Tensor grad;
        auto value = analytical(x, y, theta, c, grad);
        *loss = value.item<double>();
        require(std::isfinite(*loss) && torch::isfinite(grad).all().item<bool>(), "nonfinite output");
        std::memcpy(gradient, grad.data_ptr<double>(), 3 * (d + 1) * sizeof(double));
    });
}

extern "C" int s311_fit(const double* data, const int64_t* ys, int64_t n, int64_t d,
    double c, double* result, double* loss, double* max_gradient, int64_t* evaluations) {
    return checked([&] {
        require(result && loss && max_gradient && evaluations && std::isfinite(c) && c > 0,
                "invalid fit outputs or C");
        auto x = matrix(data, n, d), y = labels(ys, n);
        for (int k = 0; k < 3; ++k) require((y == k).any().item<bool>(), "training requires three classes");
        auto theta = torch::zeros({3, d + 1}, opts.requires_grad(true));
        torch::optim::LBFGS optimizer({theta}, torch::optim::LBFGSOptions(1.0)
            .max_iter(2000).max_eval(2500).history_size(10)
            .tolerance_grad(1e-8).tolerance_change(1e-12).line_search_fn("strong_wolfe"));
        *evaluations = 0;
        auto closure = [&]() {
            optimizer.zero_grad();
            torch::Tensor grad;
            auto value = analytical(x, y, theta, c, grad);
            theta.mutable_grad() = grad;
            ++*evaluations;
            return value;
        };
        optimizer.step(closure);
        auto final = closure();
        *loss = final.item<double>();
        *max_gradient = theta.grad().abs().max().item<double>();
        require(std::isfinite(*loss) && std::isfinite(*max_gradient) && *max_gradient <= 1e-5,
                "analytical fit did not converge");
        auto out = theta.detach().contiguous();
        std::memcpy(result, out.data_ptr<double>(), 3 * (d + 1) * sizeof(double));
    });
}
