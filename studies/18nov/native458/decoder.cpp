// One arriving EEG window. The independent seed axis is the only batch axis.
#include <ATen/ATen.h>
#include <torch/library.h>

namespace {
using at::Tensor;
Tensor feature(const Tensor& window, const Tensor& hann, const Tensor& mean, const Tensor& scale) {
    auto power = at::fft_rfft(window * hann).abs().square();
    std::vector<Tensor> bands;
    int edges[] = {4,16,32,52,80,120,180};
    for (int i = 0; i < 6; ++i)
        bands.push_back(power.slice(-1, edges[i], edges[i+1]).mean(-1).clamp_min(1e-8).log());
    return ((at::cat(bands) - mean) / scale).clamp(-8,8);
}

Tensor step(const Tensor& x, const c10::List<Tensor>& w, Tensor state, const Tensor& centroid) {
    auto dense = [&](const Tensor& v, int i) {
        return at::bmm(w.get(i),v.unsqueeze(-1)).squeeze(-1) + w.get(i+1);
    };
    auto hidden = dense(x.expand({state.size(0),-1}),0);
    hidden = at::gelu(at::layer_norm(hidden,{64}) * w.get(2) + w.get(3));
    auto query = (at::elu(dense(hidden,6)) + 1) / 12;
    auto key = at::elu(dense(hidden,8)) + 1;
    auto value = dense(hidden,10);
    auto write = at::cat({(key.unsqueeze(-1) * value.unsqueeze(1)).flatten(1), key},1);
    auto table = state.slice(-1,0,36).reshape({-1,2,12,3});
    auto mass = state.slice(-1,36,48);
    auto projected = (query.unsqueeze(1).unsqueeze(-1)*table).sum(2);
    auto divisor = 20 + (query.unsqueeze(1)*mass).sum(2,true);
    auto logits = dense(hidden,4).unsqueeze(1) + projected/divisor;
    state.add_(at::stack({write,centroid},1));
    return logits;
}
}
TORCH_LIBRARY(ni458, m) {
    m.def("feature(Tensor window, Tensor hann, Tensor mean, Tensor scale) -> Tensor",feature);
    m.def("step(Tensor x, Tensor[] weights, Tensor(a!) state, Tensor centroid) -> Tensor",step);
}
