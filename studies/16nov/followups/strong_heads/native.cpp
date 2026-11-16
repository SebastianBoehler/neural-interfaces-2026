// Immutable TorchScript classifier with explicit, independent native handles.
#include <torch/script.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace {
thread_local std::string error_text;
struct Head {
    torch::jit::script::Module model;
    torch::Device device;
    explicit Head(const char* path, int gpu)
        : model(torch::jit::load(path)), device(gpu ? torch::kCUDA : torch::kCPU) {
        model.to(device);
        model.eval();
    }
};
}

extern "C" const char* s308_error() { return error_text.c_str(); }

extern "C" void* s308_load(const char* path, int device) {
    try {
        if (!path || (device != 0 && device != 1)) throw std::runtime_error("invalid load arguments");
        return new Head(path, device);
    } catch (const std::exception& e) { error_text = e.what(); return nullptr; }
}

extern "C" int s308_predict(void* ptr, const double* x, int64_t n, int64_t d, double* out) {
    try {
        if (!ptr || !x || !out || n <= 0 || d <= 0) throw std::runtime_error("invalid prediction arguments");
        auto& h = *static_cast<Head*>(ptr);
        torch::InferenceMode guard;
        auto input = torch::from_blob(const_cast<double*>(x), {n, d}, torch::kFloat64).to(h.device);
        auto result = h.model.forward({input}).toTensor().to(torch::kCPU).to(torch::kFloat64).contiguous();
        if (result.dim() != 2 || result.size(0) != n || result.size(1) != 3 ||
            !torch::isfinite(result).all().item<bool>()) throw std::runtime_error("invalid model output");
        std::memcpy(out, result.data_ptr<double>(), n * 3 * sizeof(double));
        return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}

extern "C" void s308_free(void* ptr) { delete static_cast<Head*>(ptr); }
