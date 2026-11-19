"""Prove online output and independent tail conformance, then measure warm RTX time."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import torch

import replay
from fbl2 import FBL2Hook
from live_gate import LiveSelector
from matched import private
from mirror import Outer
from native21 import stream
from route_probe import RouteProbe
from scope import external, known
from settings import sha
from worker import authentic_stock
from benchmark_utils.streaming import fresh_model


def live(template, eeg, rows, audit=False):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    model = fresh_model(template)
    live_path = LiveSelector(model, audit)
    restored = perf_counter()
    y, dt = [], []
    for row in rows:
        tick = perf_counter()
        label = live_path.predict(eeg[int(row):int(row)+1].to("cuda:0"))
        dt.append(perf_counter() - tick)
        if label.dtype != torch.int64 or label.shape != (1,) or label.device.type != "cpu":
            raise ValueError("the normal single-window interface changed")
        y.append(int(label[0]))
    torch.cuda.synchronize()
    end = perf_counter()
    info = dict(labels=y, activity=replay.activity(model), alternate=live_path.summary(), seconds=end-start,
                restore_s=restored-start, first_s=dt[0], p95_s=float(np.percentile(dt,95)),
                peak_reserved=torch.cuda.max_memory_reserved())
    return info, live_path


def audit(template, run, eeg, rows, expected, eligible):
    info, gate = run
    stock, alternative, choice = gate.stages()
    np.testing.assert_array_equal(stock[:,2].argmax(1), expected)
    np.testing.assert_array_equal(np.stack([e[0] for e in gate.events]), eeg[rows].numpy())
    np.testing.assert_array_equal(choice.argmax(1), info["labels"])
    use = (np.arange(len(rows)) >= 20) & ~np.array([event[7] for event in gate.events])
    if int(use.sum()) != eligible or info["alternate"]["eligible"] != eligible:
        raise ValueError("fixed native gate did not activate at the expected ages")
    np.testing.assert_array_equal(choice[use], alternative[use,2])
    np.testing.assert_array_equal(choice[~use], stock[~use,2])
    solver = fresh_model(template)
    original = Outer(solver)
    np.testing.assert_array_equal(stock, private(original, gate.events, [event[2] for event in gate.events]))
    hook = FBL2Hook(solver, "fbl2")
    alt = Outer(solver)
    route = RouteProbe(SimpleNamespace(routeB=alt.route))
    q = [hook.project(r[2], r[-2], r[-1]) for r in gate.events]
    np.testing.assert_array_equal(alternative, private(alt, gate.events, q))
    if (original.cp.refits != gate.model.cp.refits or original.cp.fused != gate.model.cp.fused
            or alt.cp.refits != gate.outer.cp.refits or alt.cp.fused != gate.outer.cp.fused
            or alt.fit != gate.outer.fit or hook.summary() != gate.hook.summary()):
        raise ValueError("exact independent tail work differs from live branch")
    return dict(info, selected_changes=int((choice.argmax(1)!=stock[:,2].argmax(1)).sum()),
                max_alternative_terminal=float(np.abs(alternative[:,2]-stock[:,2]).max()),
                exact_private_native_tails=True, exact_protected_and_eligible_emissions=True,
                route=route.summary(), rows=rows.tolist())


def equal_prefix(full, short, n):
    for x, y in zip(full.stages(), short.stages(), strict=True):
        np.testing.assert_array_equal(x[:n], y)


def check(template, eeg, who, orders, channels, anchor):
    rows = orders[0][who+"|1"]
    cold = stream(template, eeg, rows, "stock")
    authentic_stock(cold, anchor)
    g = live(template, eeg, rows, audit=True)
    p = audit(template, g, eeg, rows, anchor["labels"], 100)
    authentic_stock((dict(p, labels=anchor["labels"]),None), anchor)
    if p["alternate"]["recognized"] != -1 or p["alternate"]["alternate_fused"] <= 0 or p["alternate"]["hook"]["warm_rows"] != 100:
        raise ValueError("naturally unrecognized G did not do warm alternate native work")
    px, r, expected = known(channels)
    known_run = live(template, px, r, audit=True)
    v = audit(template, known_run, px, r, expected, 0)
    if v["alternate"]["recognized"] != 10 or v["route"]["routed"] <= 0:
        raise ValueError("fixture did not exercise true native recognition")
    live(template, eeg, orders[0][who+"|2"][:17])
    equal_prefix(g[1], live(template, eeg, rows[:31], audit=True)[1], 31)
    equal_prefix(known_run[1], live(template, px, r[:17], audit=True)[1], 17)
    return dict(unrecognized=p, known=v, exact_interleaved_prefix=True)


def timed(template, eeg, who, orders, channels, index):
    smoke = json.loads(Path("smoke.json").read_text())
    delayed = json.loads(Path("delayed.json").read_text())
    if (any(p["source"] != sha("source.sha256") or p["held_labels_opened"] or not p["passed"]
            for p in (smoke, delayed)) or delayed["accepted"] != list(range(20,28))
            or not delayed["exact_archive_stock_fbl2"] or not delayed["exact_prefix"]):
        raise ValueError("measured runtime needs both signed native conformance receipts")
    if index == 0:
        x, rows = eeg, orders[0][who+"|1"]
        name, arms = "unrecognized", ("stock","selector","selector","stock")
    else:
        x, rows, _ = known(channels)
        name, arms = "known", ("selector","stock","stock","selector")
    source = smoke[name]
    warmed = stream(template,x,rows,"stock",inspect=False)[0]
    runs = [stream(template,x,rows,"stock",inspect=False)[0] if arm == "stock" else live(template,x,rows)[0] for arm in arms]
    base, gate = ([r for r, arm in zip(runs, arms, strict=True) if arm == key] for key in ("stock", "selector"))
    if any(x["activity"] != warmed["activity"] or x["activity"] != source["activity"] for x in runs):
        raise ValueError("protected stock state differs in timed runs")
    work = ("rows", "recognized", "eligible", "alternate_refits", "alternate_fused", "alternate_fit_ages", "native_stop")
    if (any(x["labels"] != warmed["labels"] for x in base)
            or any(x["labels"] != source["labels"]
                   or any(x["alternate"][k] != source["alternate"][k] for k in work) for x in gate)):
        raise ValueError("timed prototype differs from the verified live selector")
    a, b = (np.asarray([r["seconds"] for r in arm]) for arm in (base,gate))
    return dict(fixture=name, blocks=arms, warmed=warmed, runs=runs, ratio=float(b.mean()/a.mean()),
                paired=(b/a).tolist(), worst=float((b/a).max()))


def main(mode,index):
    if torch.__version__ != "2.14.1+cu130" or "RTX 2080 Ti" not in torch.cuda.get_device_name():
        raise ValueError("pinned RTX worker is required")
    eeg, who, channels, orders, anchor, archive = external(5)
    template = replay.load(Path("/package"),channels)
    payload = check(template,eeg,who,orders,channels,anchor) if mode == "smoke" else timed(template,eeg,who,orders,channels,index)
    path = Path("smoke.json" if mode == "smoke" else f"timing_{index}.json")
    with path.open("x") as f:
        json.dump(dict(passed=True,source=sha("source.sha256"),device=torch.cuda.get_device_name(),archive=archive,
                       held_labels_opened=False,**payload), f, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS F-C21gn",mode,index,sha(path),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("smoke", "timing"))
    p.add_argument("index", type=int, choices=(0,1))
    args = p.parse_args()
    main(args.mode,args.index)
