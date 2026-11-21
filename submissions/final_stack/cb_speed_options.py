"""Compare eager seed vectorization and GPU whitening on the same public prefix."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from build_cb_speed import NAME, digest
from cb_speed_bench import ATOL, stats, timed
from check_cb_speed import DATA_SHA, load_models


class Seed(torch.nn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net, self.head = net, head

    def forward(self, z):
        f = self.net(z, return_features=True)["features"]
        return self.head(f.permute(0, 2, 1, 3).flatten(2)).mean(1).softmax(1)[0]


def vectorized(cb):
    if not cb.shared:
        raise ValueError("vectorization experiment requires the shared prior")
    seeds = [Seed(net, head).eval() for net, head in zip(cb.nets, cb.heads, strict=True)]
    params, buffers = torch.func.stack_module_state(seeds)

    def forward(params, buffers, z):
        return torch.func.functional_call(seeds[0], (params, buffers), (z,), strict=True)

    batched = torch.vmap(forward, in_dims=(0, 0, None))

    def run(aligned):
        z = torch.nn.functional.interpolate(aligned, size=800, mode="linear", align_corners=False)
        return batched(params, buffers, z.clamp(-15, 15) * cb.SCALE)

    return run


def gpu_alignment(cb):
    prior = torch.as_tensor(cb.priors[0] * cb.K, device=cb.dev)
    cov = torch.zeros_like(prior)
    count = 0

    def run(raw):
        nonlocal cov, count
        x = torch.from_numpy(raw).to(cb.dev, dtype=torch.float64)
        x = x - x.mean(-1, keepdim=True)
        cov += x @ x.T / x.shape[1]
        count += 1
        matrix = (cov + prior) / (cb.K + count)
        s, u = torch.linalg.eigh(matrix)
        s = torch.maximum(s, matrix.trace() * 1e-5 / 41)
        return ((((u * s.pow(-0.5)) @ u.T) @ x).float())[None]

    return run


def align_cpu(cb, raw):
    x = raw.astype(np.float64)
    x -= x.mean(-1, keepdims=True)
    cb.cov_sum += x @ x.T / x.shape[1]
    cb.n += 1
    return torch.from_numpy(cb.whiten(x, cb.priors[0]))[None]


@torch.no_grad()
def options(cb, windows):
    cb.reset()
    vector = vectorized(cb)
    cuda_align = gpu_alignment(cb)
    times = {key: [] for key in ("cpu_whitening", "gpu_whitening", "eager", "vectorized", "gpu_whitening_eager")}
    delta = {"gpu_whitening_eager": 0.0, "vectorized": 0.0}
    vector_error = None
    for age, raw in enumerate(windows):
        start = perf_counter()
        cpu_aligned = align_cpu(cb, raw)
        cpu_ms = 1e3 * (perf_counter() - start)
        gpu_aligned, gpu_ms = timed(cuda_align, raw)
        aligned = cpu_aligned.to(cb.dev)
        original, eager_ms = timed(cb._forward_seeds, aligned)
        tested = {}
        tested["gpu_whitening_eager"], gpu_forward_ms = timed(cb._forward_seeds, gpu_aligned)
        step = dict(cpu_whitening=cpu_ms, gpu_whitening=gpu_ms, eager=eager_ms + cpu_ms,
                    gpu_whitening_eager=gpu_ms + gpu_forward_ms)
        if vector_error is None:
            try:
                tested["vectorized"], vector_ms = timed(vector, aligned)
                step["vectorized"] = vector_ms + cpu_ms
            except (RuntimeError, NotImplementedError) as exc:
                vector_error = str(exc)
                torch.cuda.synchronize()
        for key, result in tested.items():
            diff = (result - original).abs().max().item()
            if not np.isfinite(diff):
                raise RuntimeError(f"{key} returned a non-finite posterior")
            delta[key] = max(delta[key], diff)
        if age >= 16:
            for key, elapsed in step.items():
                times[key].append(elapsed)
    return dict(windows=len(windows), warm={key: stats(values) for key, values in times.items() if values},
                max_seed_abs={key: val for key, val in delta.items() if key != "vectorized" or vector_error is None},
                atol=ATOL, vectorized_error=vector_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zips", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if digest(args.data) != DATA_SHA:
        raise ValueError("PROTEUS export digest differs")
    with np.load(args.data, allow_pickle=False) as data:
        channels, windows = data["ch_names"], data["X"]
    with ExitStack() as stack:
        models, hashes = load_models(stack, args.zips, NAME, channels, "cuda:0")
        result = options(models[1].cb, windows[:(24 if args.smoke else 160)])
    result.update(zip_sha256=hashes, torch=torch.__version__, gpu=torch.cuda.get_device_name())
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
