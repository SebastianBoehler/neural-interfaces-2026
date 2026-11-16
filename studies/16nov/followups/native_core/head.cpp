// Matched multinomial logistic head. LibTorch supplies only tensor math/L-BFGS.
#include "head.h"
#include <torch/torch.h>
#include <torch/optim/lbfgs.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(message);
}

torch::Tensor matrix(const double* x, std::int64_t n, std::int64_t d) {
    require(x && n > 0 && d > 0, "missing or empty float64 matrix");
    auto result = torch::from_blob(const_cast<double*>(x), {n, d}, options);
    require(torch::isfinite(result).all().item<bool>(), "nonfinite matrix");
    return result;
}

torch::Tensor labels(const std::int64_t* y, std::int64_t n) {
    require(y != nullptr, "missing labels");
    auto result = torch::from_blob(const_cast<std::int64_t*>(y), {n}, torch::kInt64);
    require(result.min().item<std::int64_t>() >= 0 && result.max().item<std::int64_t>() < 3,
            "labels must be 0,1,2");
    return result;
}

torch::Tensor objective(const torch::Tensor& x, const torch::Tensor& y,
                        const torch::Tensor& theta, double c) {
    require(std::isfinite(c) && c > 0, "C must be finite and positive");
    const auto d = x.size(1);
    auto weights = theta.slice(1, 0, d);
    auto logits = torch::matmul(x, weights.t()) + theta.select(1, d);
    return (torch::logsumexp(logits, 1) - logits.gather(1, y.unsqueeze(1)).squeeze(1)).mean()
           + weights.square().sum() / (2 * c * x.size(0));
}

template<class Function> int checked(Function operation) {
    try { error.clear(); operation(); return 0; }
    catch (const std::exception& exception) { error = exception.what(); return -1; }
    catch (...) { error = "unknown native head error"; return -1; }
}
}

extern "C" const char* nc_head_error() { return error.c_str(); }

extern "C" int nc_head_standardize(const double* x, std::int64_t n, std::int64_t d,
    const double* mean, const double* scale, double* output) {
    return checked([&] {
        require(output != nullptr, "missing standardized output");
        torch::NoGradGuard guard;
        auto features = matrix(x, n, d);
        auto means = matrix(mean, 1, d), scales = matrix(scale, 1, d);
        require((scales > 0).all().item<bool>(), "scaler scale must be positive");
        auto result = ((features - means) / scales).contiguous();
        require(torch::isfinite(result).all().item<bool>(), "nonfinite standardized features");
        std::memcpy(output, result.data_ptr<double>(), n * d * sizeof(double));
    });
}

extern "C" int nc_head_predict(const double* x, std::int64_t n, std::int64_t d,
    const double* mean, const double* scale, const double* coefficients,
    const double* intercepts, double* logits, double* probabilities) {
    return checked([&] {
        require(logits && probabilities, "missing prediction outputs");
        torch::NoGradGuard guard;
        auto features = matrix(x, n, d);
        auto means = matrix(mean, 1, d), scales = matrix(scale, 1, d);
        require((scales > 0).all().item<bool>(), "scaler scale must be positive");
        auto weights = matrix(coefficients, 3, d), biases = matrix(intercepts, 1, 3);
        auto z = (torch::matmul((features - means) / scales, weights.t()) + biases).contiguous();
        auto p = torch::softmax(z, 1).contiguous();
        require(torch::isfinite(z).all().item<bool>() && torch::isfinite(p).all().item<bool>(),
                "nonfinite prediction");
        std::memcpy(logits, z.data_ptr<double>(), n * 3 * sizeof(double));
        std::memcpy(probabilities, p.data_ptr<double>(), n * 3 * sizeof(double));
    });
}

extern "C" int nc_head_objective(const double* x, const std::int64_t* y,
    std::int64_t n, std::int64_t d, const double* theta, double c,
    double* loss, double* gradient) {
    return checked([&] {
        require(loss && gradient, "missing objective outputs");
        auto features = matrix(x, n, d), target = labels(y, n);
        auto parameters = matrix(theta, 3, d + 1).clone().set_requires_grad(true);
        auto value = objective(features, target, parameters, c);
        value.backward();
        auto grad = parameters.grad().contiguous();
        *loss = value.item<double>();
        require(std::isfinite(*loss) && torch::isfinite(grad).all().item<bool>(),
                "nonfinite objective or gradient");
        std::memcpy(gradient, grad.data_ptr<double>(), 3 * (d + 1) * sizeof(double));
    });
}

namespace {
void fit_impl(const double* x, const std::int64_t* y,
    std::int64_t n, std::int64_t d, double c, double* theta,
    double* loss, double* max_gradient, std::int64_t* evaluations, torch::Device device) {
        require(theta && loss && max_gradient && evaluations, "missing fit outputs");
        auto features = matrix(x, n, d).to(device), target = labels(y, n).to(device);
        for (std::int64_t k = 0; k < 3; ++k)
            require((target == k).any().item<bool>(), "training requires all three classes");
        auto parameters = torch::zeros({3, d + 1}, options.device(device).requires_grad(true));
        torch::optim::LBFGS optimizer({parameters}, torch::optim::LBFGSOptions(1.0)
            .max_iter(2000).max_eval(2500).history_size(10)
            .tolerance_grad(1e-8).tolerance_change(1e-12).line_search_fn("strong_wolfe"));
        *evaluations = 0;
        auto closure = [&]() {
            optimizer.zero_grad();
            auto value = objective(features, target, parameters, c);
            value.backward();
            ++*evaluations;
            return value;
        };
        optimizer.step(closure);
        auto final = closure();
        *loss = final.item<double>();
        *max_gradient = parameters.grad().abs().max().item<double>();
        require(std::isfinite(*loss) && std::isfinite(*max_gradient), "nonfinite fitted head");
        require(*max_gradient <= 1e-5, "native logistic solver did not converge");
        auto result = parameters.detach().to(torch::kCPU).contiguous();
        std::memcpy(theta, result.data_ptr<double>(), 3 * (d + 1) * sizeof(double));
}
}

extern "C" int nc_head_fit(const double* x, const std::int64_t* y,
    std::int64_t n, std::int64_t d, double c, double* theta,
    double* loss, double* max_gradient, std::int64_t* evaluations) {
    return checked([&] { fit_impl(x, y, n, d, c, theta, loss, max_gradient, evaluations, torch::kCPU); });
}

extern "C" int nc_head_fit_device(const double* x, const std::int64_t* y,
    std::int64_t n, std::int64_t d, double c, int device, double* theta,
    double* loss, double* max_gradient, std::int64_t* evaluations) {
    return checked([&] {
        require(device == 0 || device == 1, "device must be 0=CPU or 1=CUDA");
        const auto selected = device == 1 ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
        fit_impl(x, y, n, d, c, theta, loss, max_gradient, evaluations, selected);
    });
}
