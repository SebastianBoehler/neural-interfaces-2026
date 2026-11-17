// Exact source-span ridge logistic fit, with original-space convergence checks.
#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

extern "C" int nc_head_fit(const double*, const int64_t*, int64_t, int64_t, double,
                            double*, double*, double*, int64_t*);
extern "C" const char* nc_head_error();
namespace {
thread_local std::string error;
using Clock = std::chrono::steady_clock;
const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
double seconds(Clock::time_point tick) {
    return std::chrono::duration<double>(Clock::now() - tick).count();
}
}
extern "C" const char* s343_error() { return error.c_str(); }

// stats: factor/solve/reconstruct/check wall, rank, eigmin/max, cutoff,
// discarded absolute sum, original objective/gradient, reduced objective/gradient.
extern "C" int s343_fit(const double* data, const int64_t* labels, int64_t n,
    int64_t d, double c, double* result, double* stats, int64_t* evaluations) {
    try {
        error.clear();
        torch::NoGradGuard guard;
        require(data && labels && result && stats && evaluations && n > 3 && d >= n,
                "missing outputs or source matrix must be wide with N>3");
        require(std::isfinite(c) && c > 0, "C must be finite and positive");
        auto tick = Clock::now();
        for (int64_t i = 0; i < n * d; ++i)
            require(std::isfinite(data[i]), "nonfinite source matrix");
        for (int64_t i = 0; i < n; ++i)
            require(labels[i] >= 0 && labels[i] < 3, "invalid source labels");
        auto x = torch::from_blob(const_cast<double*>(data), {n, d}, options);
        auto y = torch::from_blob(const_cast<int64_t*>(labels), {n}, torch::kInt64);
        auto gram = torch::matmul(x, x.t());
        auto decomposition = torch::linalg_eigh(gram);
        auto eigenvalues = std::get<0>(decomposition);
        auto eigenvectors = std::get<1>(decomposition);
        double largest = eigenvalues.max().item<double>();
        double smallest = eigenvalues.min().item<double>();
        double cutoff = std::numeric_limits<double>::epsilon() * n * largest;
        require(largest > 0 && smallest >= -cutoff, "Gram matrix has a materially negative eigenvalue");
        auto retained = eigenvalues > cutoff;
        auto values = eigenvalues.index({retained});
        auto vectors = eigenvectors.index({torch::indexing::Slice(), retained});
        auto roots = torch::sqrt(values);
        auto z = (vectors * roots).contiguous();
        int64_t rank = z.size(1);
        stats[13] = (gram - torch::matmul(z, z.t())).norm().item<double>()
                    / gram.norm().item<double>();
        require(stats[13] <= 1e-10, "source-span Gram reconstruction failed");
        require(rank > 0, "empty source span");
        stats[0] = seconds(tick);
        stats[4] = static_cast<double>(rank);
        stats[5] = smallest;
        stats[6] = largest;
        stats[7] = cutoff;
        stats[8] = eigenvalues.index({~retained}).abs().sum().item<double>();
        auto reduced = torch::empty({3, rank + 1}, options);
        double loss, gradient;
        tick = Clock::now();
        // The original fit creates its own autograd scope; release NoGradGuard.
        {
            torch::AutoGradMode enable(true);
            int rc = nc_head_fit(z.data_ptr<double>(), labels, n, rank, c,
                                 reduced.data_ptr<double>(), &loss, &gradient, evaluations);
            if (rc) throw std::runtime_error(nc_head_error());
        }
        stats[1] = seconds(tick);
        stats[11] = loss;
        stats[12] = gradient;
        tick = Clock::now();
        auto dual = torch::matmul(vectors / roots, reduced.slice(1, 0, rank).t());
        auto weights = torch::matmul(x.t(), dual).t();
        auto theta = torch::cat({weights, reduced.select(1, rank).unsqueeze(1)}, 1).contiguous();
        stats[2] = seconds(tick);
        tick = Clock::now();
        auto logits = torch::matmul(x, weights.t()) + theta.select(1, d);
        auto residual = torch::softmax(logits, 1);
        residual.scatter_add_(1, y.unsqueeze(1), -torch::ones({n, 1}, options));
        residual.div_(n);
        auto full_gradient = torch::matmul(residual.t(), x) + weights / (c * n);
        auto bias_gradient = residual.sum(0);
        stats[9] = ((torch::logsumexp(logits, 1) - logits.gather(1, y.unsqueeze(1)).squeeze(1)).mean()
                    + weights.square().sum() / (2 * c * n)).item<double>();
        stats[10] = std::max(full_gradient.abs().max().item<double>(), bias_gradient.abs().max().item<double>());
        stats[3] = seconds(tick);
        require(std::isfinite(stats[9]) && std::isfinite(stats[10]) && stats[10] <= 1e-5,
                "reconstructed original-space fit did not converge");
        require(std::abs(stats[9] - loss) <= 1e-9, "factorization changed original objective");
        require(torch::isfinite(theta).all().item<bool>(), "nonfinite reconstructed parameters");
        std::memcpy(result, theta.data_ptr<double>(), 3 * (d + 1) * sizeof(double));
        return 0;
    } catch (const std::exception& exception) {
        error = exception.what();
        return -1;
    } catch (...) {
        error = "unknown source-span error";
        return -1;
    }
}
