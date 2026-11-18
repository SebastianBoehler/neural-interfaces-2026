// Complete frozen §439/441 decoder. All mutable state belongs to one stream call.
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <cstdio>
#include <cstring>

namespace {
using at::Tensor;
struct Decoder {
    Tensor spatial, temporal, writer, wb, reader, rb, adapt, ab, head, hb, label, lb;
    Tensor mean, scale;

    explicit Decoder(const float* flat, int n) {
        int offset = 0;
        auto take = [&](at::IntArrayRef shape) {
            int size = 1;
            for (auto v : shape) size *= v;
            TORCH_CHECK(offset + size <= n, "incomplete parameter buffer");
            auto t = at::from_blob(const_cast<float*>(flat + offset), shape, at::kFloat).to(at::kCUDA);
            offset += size;
            return t;
        };
        spatial = take({48,41,1}); temporal = take({48,1,33});
        writer = take({48,96}); wb = take({48});
        reader = take({64,49}); rb = take({64}); adapt = take({96,64}); ab = take({96});
        head = take({64,192}); hb = take({64}); label = take({3,64}); lb = take({3});
        mean = take({48});
        scale = (take({48}) - mean.square()).clamp_min(1e-6).rsqrt();
        TORCH_CHECK(offset == n, "unexpected parameter count");
    }

    Tensor classify(const Tensor& normalized, const Tensor& context) const {
        auto hidden = at::linear(context, reader, rb).tanh();
        auto chunks = at::linear(hidden, adapt, ab).chunk(2, -1);
        auto adjusted = normalized * chunks[1].tanh().exp().unsqueeze(-1) + chunks[0].unsqueeze(-1);
        auto bins = adjusted.reshape({1,48,4,28}).square().mean(-1).clamp_min(1e-8).log();
        return at::linear(at::gelu(at::linear(bins.flatten(1), head, hb)), label, lb);
    }
};
}

// rows are in actual delivery order. Outputs [window, genuine/blank, class].
extern "C" int decode452(const float* weights, int nw, const float* rows, int count,
                         float rms, float* out, char* error, int capacity) {
    try {
        c10::InferenceMode guard;
        TORCH_CHECK(count > 0 && rms > 0, "invalid stream");
        Decoder net(weights, nw);
        auto sum = at::zeros_like(net.mean).unsqueeze(0);
        auto age = at::zeros({1,1}, sum.options());
        auto blank = at::zeros({1,49}, sum.options());
        auto logits = at::empty({count,2,3}, sum.options());
        for (int i = 0; i < count; ++i) {
            auto x = at::from_blob(const_cast<float*>(rows+i*41*480), {1,41,480}, at::kFloat).to(at::kCUDA) / rms;
            auto spatial = at::conv1d(x, net.spatial);
            auto features = at::conv1d(spatial, net.temporal, std::nullopt, at::IntArrayRef{4}, at::IntArrayRef{0}, at::IntArrayRef{1}, 48);
            auto normalized = (features - net.mean.unsqueeze(-1)) * net.scale.unsqueeze(-1);
            age.fill_(i);
            auto context = at::cat({sum / age.clamp_min(1), age / (age+20)}, -1);
            logits[i][0].copy_(net.classify(normalized, context).view({3}));
            logits[i][1].copy_(net.classify(normalized, blank).view({3}));
            auto input = at::cat({features.mean(-1), features.square().mean(-1).clamp_min(1e-8).log()}, -1);
            sum.add_(at::linear(input, net.writer, net.wb).tanh());
        }
        auto host = logits.cpu().contiguous();
        std::memcpy(out, host.const_data_ptr<float>(), count * 6 * sizeof(float));
        return 0;
    } catch (const std::exception& exc) {
        std::snprintf(error, capacity, "%s", exc.what());
        return 1;
    }
}
