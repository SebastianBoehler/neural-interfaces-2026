"""Witness the actual capped FBLight output on the pinned public stress rows."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from fbl2 import FBL2Hook, MEMBERS
from route_probe import RouteProbe
from settings import sha, stamp


def main():
    assert torch.__version__ == "2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    assert sha("/native/source.sha256") == "61feef26593423464c1e3d8935c0ef43ce01d737df84fa6c73035d0b3b663841"
    smoke = json.loads(Path("/native/smoke.json").read_text())
    assert smoke["passed"] and not smoke["held_labels_opened"] and smoke["source"] == sha("/native/source.sha256")
    reference = json.loads(Path("/reference.json").read_text())["streams"]["first"]["forced-unknown-378"]["activity"]
    x, _, channels = replay.read_source()
    rows = np.random.default_rng(36017800).permutation(len(x))[:378].astype("<i8")
    template = replay.load(Path("/package"), channels)
    model = fresh_model(template)
    assert not model.v5hist and model.cp.t == 0 and model.ft.cap == 250
    gates = model.route.identity, model.route.contrast
    original_tau = [g.tau for g in gates]
    assert all(0 <= t < 1e6 for t in original_tau)
    for g in gates:
        g.tau = 1e6
    hook, route = FBL2Hook(model, "fbl2"), RouteProbe(model)
    skips = []
    post_warm = 0
    frozen_error, mix_change, arithmetic = 0., 0., 0.
    started = perf_counter()
    with torch.jit.optimized_execution(False):
        for age, row in enumerate(rows):
            count = model.ft.capped
            y = model.predict(x[int(row):int(row)+1].to("cuda:0"))
            assert y.shape == (1,) and y.dtype == torch.int64 and y.device.type == "cpu"
            assert not any(getattr(p, "dead", False) for p in (model, model.cf, model.cp))
            trace = model.trace["v6"]
            hook.check(trace)
            assert hook.last_mix is route.last and hook.last_warm == (age >= 20)
            assert model.cp.t == age + 1 and model.cp.stopped is None
            if model.ft.capped != count:
                assert model.ft.capped == count + 1
                skips.append(age)
            if not skips:
                continue
            assert model.ft.steps == 480 and model.ft.refits == 24 and model.ft.relabels == 24
            assert all(len(a) == 250 for a in model.ft.X.values())
            frozen = np.mean([trace["members"][m] for m in MEMBERS], axis=0)
            adapted = np.maximum(frozen + trace["p_ft"]["fbl"] - trace["members"][MEMBERS[0]], 1e-6)
            adapted /= adapted.sum()
            expected = (16 * hook.last_stock + adapted) / 17
            arithmetic = max(arithmetic, float(np.max(np.abs(expected - route.last))))
            frozen_error = max(frozen_error, float(np.max(np.abs(frozen - hook.last_fbl))))
            delta = float(np.max(np.abs(route.last - hook.last_stock)))
            assert delta > 0 and arithmetic <= 2e-14
            mix_change = max(mix_change, delta)
            post_warm += 1
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    activity = replay.activity(model)
    for name in ("windows", "recognized", "ft_refits", "ft_steps", "ft_relabels", "ft_capped", "ft_over", "errors"):
        assert activity[name] == reference[name], (name, activity[name], reference[name])
    assert hook.warm == 358 and hook.frozen == 20 and post_warm == 118
    assert skips == list(range(260, 378, 10)) and frozen_error > .01
    assert activity["cp_refits"] > 0 and activity["cp_fused"] > 0
    assert not model.recB and not route.summary()["routed"]
    assert not template.v5hist and template.cp.t == 0
    out = Path("cap_witness.json")
    with out.open("x") as f:
        json.dump(dict(passed=True, held_labels_opened=False, rows=stamp(rows), seconds=elapsed,
                       activity=activity, archived_activity=reference, skips=skips, post_cap_warm=post_warm,
                       original_tau=original_tau, threshold=1e6, post_cap_warm_vs_frozen_max=frozen_error,
                       post_cap_mix_max=mix_change, arithmetic_error=arithmetic, hook=hook.summary(),
                       smoke=sha("/native/smoke.json"), source=sha("source.sha256"),
                       device=torch.cuda.get_device_name()), f, sort_keys=True, allow_nan=False)
        f.write("\n")
    out.chmod(0o444)
    print("PASS F-C21cap", sha(out), activity, frozen_error, arithmetic, flush=True)


if __name__ == "__main__":
    main()
