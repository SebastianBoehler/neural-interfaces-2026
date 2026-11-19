"""Verify bitwise all-off parity, reachable switches, and paired long RTX timing."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

from external import bank_case, windows
from make import digest
from native import assert_equal, bitwise, fresh_tail, load, stream
from orders import stamp


PUBLIC_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
LONG_ROWS = "c5b80556953e87b39a223e78e578eb666364fd0906c5b9741268a40eac39f0c7"


def public(path):
    if digest(path) != PUBLIC_SHA:
        raise ValueError("Unexpected PROTEUS export")
    with np.load(path, allow_pickle=False) as z:
        return torch.from_numpy(z["X"]), z["ch_names"].tolist(), z["subject"], z["session"]


def smoke(package, data, runs, public_data, device):
    x, _, visits, channels, order = windows(data, runs)
    p = order[0, 0]
    unknown = x[p[visits[p] == "Scherer2015/G|1"]].to(device)
    x, public_channels, people, visit = public(public_data)
    if channels != public_channels:
        raise ValueError("The two exports have different montages")
    known = x[(people == "Dreyer2026Proteus/P10") & (visit == "3")][:31].to(device)
    templates = {arm: load(package / f"{arm}.zip", channels, device) for arm in ("stock", "off", "B2", "B4")}
    templates["channels"] = channels
    tails = {arm: fresh_tail(package / f"{arm}.zip") for arm in ("off", "B1", "B2", "B3", "B5")}
    # Identical executor mode in both archives isolates the Python changes from graph warm-up.
    with torch.jit.optimized_execution(False):
        for batch, force in ((known, None), (unknown, lambda _: False)):
            stock = stream(templates["stock"], batch, force=force)
            off = stream(templates["off"], batch, force=force)
            assert_equal(stock, off)
            if force is None and stock["meta"]["rec"] < 0:
                raise ValueError("The recognized null fixture did not recognize")
            if force is not None and not (stock["meta"]["ft_refits"] and stock["meta"]["csp_fusions"]):
                raise ValueError("The unrecognized null fixture did not adapt")
        forced_window = lambda age: 30 <= age < 40 or 91 <= age < 100
        old = stream(templates["stock"], unknown[:104], force=forced_window)
        changed = stream(templates["B2"], unknown[:104], force=forced_window)
        if (old["warm"][40:].any() or old["fused"][-1] or not changed["warm"][40]
                or changed["warm"][30:40].any() or changed["meta"]["recognition_returns"] != 2
                or changed["fused"][99] != changed["fused"][90] or changed["fused"][100] <= changed["fused"][99]):
            raise ValueError("Forced recognition did not pause and resume both stages")
        short = bank_case(package, templates, tails, unknown)
        bitwise(short["stock"][0], short["B4"][0], "B4 short labels")
        assert_equal(stream(templates["stock"], unknown[:31]), stream(templates["B4"], unknown[:31]))
    csp = tails["B3"][2]
    csp.reset_state()
    scores = np.array([.63, .32, .05])
    for i in range(91):
        posterior = np.roll(scores, i % 2)
        csp.step(unknown[i].cpu().numpy(), posterior, posterior, False)
    if csp.log["err"] or set(csp.model[1].classes_) != {0, 1} or csp.fused != 1:
        raise ValueError("The real two-class CSP fit did not fuse")
    np.testing.assert_allclose(csp.correction(np.array([.8, .2, 0.])), np.log([1.6, .4, 1.]))
    if csp.correction(np.array([.5, .5, 0.])) is not None:
        raise ValueError("Uniform CSP counted as a correction")
    print("PASS bitwise all-off, forced B2 and B3, and short B4", flush=True)


def timing(package, public_data, device):
    x, channels, _, _ = public(public_data)
    rows = np.random.default_rng(36017800).permutation(len(x))[:378].astype("<i8")
    if stamp(rows) != LONG_ROWS:
        raise ValueError("Unexpected long runtime fixture rows")
    windows = x[rows].to(device)
    original, changed = (load(package / f"{name}.zip", channels, device) for name in ("stock", "B4"))
    with torch.jit.optimized_execution(False):
        stream(original, windows[:31], force=lambda _: False)
        pairs = []
        for first, second in ((original, changed), (changed, original)):
            pair = [stream(t, windows, force=lambda _: False) for t in (first, second)]
            stock, extended = sorted(pair, key=lambda t: t["meta"]["cap"])
            for key in ("labels", "raw", "terminal"):
                bitwise(stock[key][:260], extended[key][:260], f"pre-extension {key}")
            a, b = stock["meta"], extended["meta"]
            if (a["ft_refits"], b["ft_refits"], b["ft_steps"] - a["ft_steps"]) != (24, 31, 140):
                raise ValueError("The seven post-250 refits did not run")
            pairs.append(dict(stock=a, B4=b, added_s=b["wall_s"] - a["wall_s"]))
    print(json.dumps(dict(device=torch.cuda.get_device_name() if device != "cpu" else "cpu", pairs=pairs)), flush=True)


if __name__ == "__main__":
    mode, package, data, runs, public_data, device = sys.argv[1:]
    if mode == "smoke":
        smoke(*map(Path, (package, data, runs, public_data)), device)
    elif mode == "timing":
        timing(Path(package), Path(public_data), device)
    else:
        raise ValueError(mode)
