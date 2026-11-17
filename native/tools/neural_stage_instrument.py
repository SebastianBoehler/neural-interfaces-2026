"""Observation-only copies of actual native source graph; no production API/global changes."""
import argparse
import hashlib
import json
from pathlib import Path


def instrument(root):
    replacements={
      'training_fbc.cpp':[
       ('    spectra_.push_back(at::fft_rfft(f, 480 + f.numel() - 1));','    spectra_.push_back(at::fft_rfft(f, 480 + f.numel() - 1));\n    source_observe("fir_spectrum_" + std::to_string(i), spectra_.back().view({1, 1, -1}).expand({1, 41, spectra_.back().numel()}));'),
       ('  std::map<int64_t, Tensor> cache;','  source_observe("input", x);\n  std::map<int64_t, Tensor> cache;'),
       ('    auto start = (n - 480) / 2;','    source_observe("input_spectrum_" + std::to_string(i), cache.at(n));\n    auto start = (n - 480) / 2;'),
       ('  auto h = F::conv2d(torch::stack(bands, 1), spatial_weight_.renorm(2, 0, 2.),','  auto filtered = torch::stack(bands, 1);\n  source_observe("fir", filtered);\n  auto spatial = spatial_weight_.renorm(2, 0, 2.);\n  source_observe("spatial_weight", spatial);\n  auto h = F::conv2d(filtered, spatial,'),
       ('  h = torch::silu(bn_.forward(h, training)).view({x.size(0), 288, 4, 120});','  source_observe("conv", h);\n  auto normalized = bn_.forward(h, training);\n  source_observe("bn", normalized);\n  auto activation = torch::silu(normalized);\n  source_observe("activation", activation);\n  h = activation.view({x.size(0), 288, 4, 120});\n  source_observe("segments", h);'),
       ('  auto features = h.var(std::vector<int64_t>{3}, true, true).clamp(1e-6, 1e6).log().flatten(1);','  auto logvar = h.var(std::vector<int64_t>{3}, true, true).clamp(1e-6, 1e6).log();\n  source_observe("logvar", logvar);\n  auto features = logvar.flatten(1);'),
       ('  return F::linear(features, head_weight_.renorm(2, 0, .5), head_bias_);','  source_observe("features", features);\n  auto head = head_weight_.renorm(2, 0, .5);\n  source_observe("head_weight", head);\n  auto logits = F::linear(features, head, head_bias_);\n  source_observe("logits", logits);\n  return logits;')],
      'session_bn.cpp':[
       ('    auto gm = r1.mean(1), gv = (r2.mean(1) - gm.square()).clamp_min(0);','    source_observe("bn_r1", r1);\n    source_observe("bn_r2", r2);\n    auto gm = r1.mean(1), gv = (r2.mean(1) - gm.square()).clamp_min(0);\n    source_observe("bn_mean", gm);\n    source_observe("bn_var", gv);')]}
    proof={}
    for name,changes in replacements.items():
        path=root/'native/src'/name;original=path.read_text();changed=original
        for before,after in changes:
            if changed.count(before)!=1:raise ValueError('native source observation anchor changed:'+before)
            changed=changed.replace(before,after)
        changed='#include "../tests/source_trace.h"\n'+changed
        output=path.with_name(path.stem+'_observed.cpp');output.write_text(changed)
        sha=lambda value:hashlib.sha256(value.encode()).hexdigest()
        proof[name]=dict(original_sha256=sha(original),observed_sha256=sha(changed),anchors=len(changes))
    (root/'instrumentation.json').write_text(json.dumps(proof,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);instrument(p.parse_args().root)
