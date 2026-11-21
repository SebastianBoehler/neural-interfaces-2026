"""Where the Team B time goes inside the real Solver (sync-timed sub-steps of Encoder.posterior and a torch profiler table).

python profile_team_b_overhead.py ZIPS_DIR DATA OUT.json ZIP_NAME [--sessions N] [--threads N]
Streams the first N sessions (shuffle 71000) of the exact ZIP through the official fresh_model path; every wrapped
function is CUDA-synchronised around the call, so the per-call means are exact but add their own sync cost."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

import check_hard_team_b as C

ACC = defaultdict(lambda: [0, 0.0])


def timed(name, fn):
    def inner(*a, **k):
        if torch.cuda.is_current_stream_capturing():  # a sync inside a capture invalidates it (graph would fall back to eager)
            return fn(*a, **k)
        torch.cuda.synchronize()
        t = perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        ACC[name][0] += 1
        ACC[name][1] += perf_counter() - t
        return out
    return inner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zips", type=Path); p.add_argument("data", type=Path); p.add_argument("out", type=Path); p.add_argument("name")
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--prof", action="store_true")
    a = p.parse_args()
    import warnings
    warnings.filterwarnings("error", message="Team B CUDA graph capture failed")  # eager fallback would invalidate the profile
    data = C.D.load(a.data)
    plan = list(C.plans(data, "71000").items())[:a.sessions + 1]
    with TemporaryDirectory() as scratch:
        directory = Path(scratch) / a.name
        module = C.unpack(a.zips / f"{a.name}.zip", directory)
        model = C.load(module, directory, data, "cuda:0")
        if a.threads:
            torch.set_num_threads(a.threads)
        enc = type(model.team_b)
        g = module._TEAM_B
        for fn in ("detrend", "bandpass", "whitener", "sfn"):
            g[fn] = timed("fn_" + fn, g[fn])
        embed = enc._embed

        def embed_by_size(self, x):  # the graph path for batch 1 and 64, eager otherwise
            if torch.cuda.is_current_stream_capturing():
                return embed(self, x)
            torch.cuda.synchronize()
            t = perf_counter()
            out = embed(self, x)
            torch.cuda.synchronize()
            key = f"embed_size_{len(x) if len(x) in (1, 64) else 'other'}"
            ACC[key][0] += 1
            ACC[key][1] += perf_counter() - t
            return out
        enc._embed = embed_by_size
        for meth in ("posterior", "_embed_prefix"):
            setattr(enc, meth, timed("enc_" + meth, getattr(enc, meth)))
        eager_batch = enc._embed_batch  # runs only for eager forwards; graph replays and captures bypass or nest it
        def counted(self, x):
            if torch.cuda.is_current_stream_capturing() or any(g.graph is None and not g.failed and len(x) == n for n, g in self.feature_graphs.items()):
                return eager_batch(self, x)  # capture / warm-up forwards of a graph that is being built
            torch.cuda.synchronize()
            t = perf_counter()
            out = eager_batch(self, x)
            torch.cuda.synchronize()
            ACC[f"eager_forward_batch{'<=16' if len(x) <= 16 else '<64' if len(x) < 64 else '64'}"][0] += 1
            ACC[f"eager_forward_batch{'<=16' if len(x) <= 16 else '<64' if len(x) < 64 else '64'}"][1] += perf_counter() - t
            return out
        enc._embed_batch = counted
        hp = type(model).hard_probability
        type(model).hard_probability = timed("hard_probability", hp)
        mix6 = type(model).mix6
        type(model).mix6 = timed("mix6_total", mix6)
        res = []
        for k, (session, rows) in enumerate(plan):
            w = torch.from_numpy(data["X"][rows]).to("cuda:0")
            if k == 1:
                ACC.clear()
            t = perf_counter()
            if a.prof and k == 1:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
                    C.stream(model, w)
                print(pr.key_averages().table(sort_by="cpu_time_total", row_limit=25), flush=True)
            else:
                C.stream(model, w)
            torch.cuda.synchronize()
            if k:
                res.append((len(rows), perf_counter() - t))
    n = sum(r[0] for r in res)
    out = dict(gpu=torch.cuda.get_device_name(), threads=torch.get_num_threads(), windows=n,
               wall_ms_per_window=1e3 * sum(r[1] for r in res) / n,
               parts={k: dict(calls=c, ms_per_window=1e3 * s / n, ms_per_call=1e3 * s / max(c, 1)) for k, (c, s) in ACC.items()})
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
