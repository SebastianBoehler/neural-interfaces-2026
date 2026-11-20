"""Replay waveform experts on verified schedules with a perfect-ID diagnostic."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch

from waveform import Fingerprint, Traces, WaveExpert


def load_inputs(args):
    data, windows = {}, {}
    for f in sorted(args.features.glob("*.npz")):
        with np.load(f) as z:
            data[f.stem] = {k: z[k] for k in ("idx", "day", "spec", "y1")}
    for corpus, path in (("pub", args.public), ("sch", args.scherer)):
        with np.load(path) as z:
            x = z["X"]
        for k, d in data.items():
            if k.startswith(corpus):
                windows[k] = x[d["idx"]]
    for i in range(19):
        with np.load(args.b10 / "ext" / f"p{i}.npz") as z:
            windows[f"net_p{i}"] = z["X"]
    assert len(windows) == len(data) == 38
    return data, windows


def main(args):
    ds, waves = load_inputs(args)
    gate = Fingerprint(args.identity, ds)
    pubs = sorted(k for k in ds if k.startswith("pub_"))

    @lru_cache(maxsize=2)
    def load_fold(path):
        return Traces(path)

    @lru_cache(maxsize=40)
    def train_expert(pkg, person):
        d = ds[person]
        return WaveExpert(load_fold(pkg), waves[person][d["day"] == 1], d["y1"])

    start = time.perf_counter()
    paths = sorted(list(args.pred.glob("*.npz")), key=lambda p: (p.stem.split("_")[1], p.stem))
    paths = [p for p in paths if not p.name.startswith("pred_pop")]
    assert len(paths) == 249, len(paths)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(exist_ok=True)
    for path in paths:
        mode, corpus, person, *pair = path.stem.split("_")
        key = corpus + "_" + person
        if args.smoke and path.stem != "return_pub_P04":
            continue
        names = sorted(set(pubs + [k for k in ds if k.startswith(corpus + "_")]))
        if mode in ("new", "publiconly", "wrong"):
            names.remove(key)
        if mode == "publiconly":
            names = pubs
        elif mode == "single":
            names = pubs + [key]
        elif mode == "wrong":
            names = pubs + ["_".join(pair[:2])]
        assert (key in names) == (mode in ("return", "single"))
        pkg = args.zip
        if corpus == "pub":
            role = "ret" if mode == "return" else "new"
            pkg = args.stock / f"{role}_{person}" / "pkg"
        with np.load(path) as z:
            tags = json.loads(str(z["routing"]))
            output, routing = {}, {}
            for tag in tags:
                get = lambda k: z[tag + "|" + k]
                rows, base, stock = get("rows"), get("new"), get("stock")
                np.testing.assert_array_equal(base, get("zero"))
                ix = np.searchsorted(ds[key]["idx"], rows)
                np.testing.assert_array_equal(ds[key]["idx"][ix], rows)
                projection = gate.proj[key][ix]
                at, name = gate.route(names, projection)
                x = waves[key][ix]
                pred, protected, oracle = base.copy(), stock.copy(), base.copy()
                probs = {}
                for target in dict.fromkeys([name] + ([key] if mode in ("return", "single") else [])):
                    if target:
                        expert = train_expert(pkg, target)
                        probs[target] = expert.predict(x)
                        if args.smoke:
                            np.testing.assert_allclose(probs[target][:18], expert.predict(x[:18]), atol=4e-5)
                            # A second call after an unrelated prefix must reset every StreamBN buffer.
                            expert.predict(x[-8:])
                            np.testing.assert_allclose(probs[target][:1], expert.predict(x[:1]), atol=4e-5)
                if name:
                    p = probs[name]
                    mean = p.mean(1)
                    pred[at:] = mean[at:].argmax(1)
                    same = (p[:, 0].argmax(1) == p[:, 1].argmax(1))
                    mask = (np.arange(len(x)) >= at) & same & (mean.max(1) >= .70)
                    protected[mask] = mean[mask].argmax(1)
                if mode in ("return", "single"):
                    oracle[10:] = probs[key][10:].mean(1).argmax(1)
                if args.smoke:
                    a, j = gate.route(names, projection[:50])
                    assert (a, j) == ((at, name) if 0 <= at < 50 else (-1, ""))
                    from timing import timing
                    timing(train_expert(pkg, key), x, names, gate, args.out / "warm.json")
                    print("WAVEFORM SMOKE OK", path.stem, tag, at, name, flush=True)
                    return
                for k, v in dict(rows=rows, stock=stock, new=base, heldnew=get("heldnew"), zero=base,
                                 wave=pred, protected=protected, oracle=oracle).items():
                    output[tag+"|"+k] = v
                routing[tag] = {"wave": [at, name]}
        output["routing"] = np.array(json.dumps(routing))
        np.savez_compressed(args.out / path.name, **output)
        print("wave", path.stem, round(time.perf_counter()-start, 1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("features", "public", "scherer", "stock", "b10", "identity", "pred", "out", "zip"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
