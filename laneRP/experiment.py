"""Blind outer-system replay. Labels from evaluation visits are opened only by score.py."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from model import Population
from recognition import gallery, recognize
from streams import netbci, public, read_labels, scherer


class Engine:
    def __init__(self, records, people):
        self.ds = records
        self.pop = Population([records[k] for k in people])
        self.experts = {}
        self.templates = {k: [v[0] for v in gallery([d])] for k, d in records.items()}

    def evaluate(self, s, names, smoke=False):
        # Person, day and row IDs select the offline stream here. Neither gate nor expert receives them.
        d = self.ds[s.person]
        local = np.searchsorted(d["idx"], s.rows)
        np.testing.assert_array_equal(d["idx"][local], s.rows)
        assert (d["day"][local] == s.day).all() and len(np.unique(local)) == len(local)
        x = d["f"][local]
        probes = [d[k][local] for k in ("spec", "conn")]
        templates = [np.stack([self.templates[k][i] for k in names]) for i in range(2)]
        routes = recognize(probes, templates)
        predictions = {a: s.new.copy() for a in ("strict", "loose", "ridge", "neural", "zero")}
        predictions["hybrid"] = s.stock.copy()
        mapping = {}
        cache = {}
        for gate, (at, who) in routes.items():
            mapping[gate] = [at, names[who] if who >= 0 else ""]
            if who < 0:
                continue
            name = names[who]
            if name not in self.experts:
                self.experts[name] = self.pop.personalize(self.ds[name])
            if name not in cache:
                cache[name] = self.experts[name].predict(x)
            pair = cache[name]
            assert np.isfinite(pair).all() and np.allclose(pair.sum(2), 1, atol=1e-5)
            prob = pair.mean(1)
            predictions[gate][at:] = prob[at:].argmax(1)
            if gate == "strict":
                votes = pair.argmax(2)
                predictions["ridge"][at:] = votes[at:, 0]
                predictions["neural"][at:] = votes[at:, 1]
                mask = (np.arange(len(x)) >= at) & (votes[:, 0] == votes[:, 1]) & (prob.max(1) >= .70)
                predictions["hybrid"][mask] = votes[mask, 0]
            if smoke:
                end = min(len(x) - 1, max(at + 1, 45))
                p = self.experts[name].predict(x[:end])
                np.testing.assert_allclose(pair[:end], p, atol=2e-5)
        np.testing.assert_array_equal(predictions["zero"], s.new)
        if smoke:
            end = len(x) // 2
            short = recognize([v[:end] for v in probes], templates)
            for gate, (at, who) in routes.items():
                assert short[gate] == ((at, who) if 0 <= at < end else (-1, -1))
        return mapping, predictions


def run(args):
    ds = {}
    for path in sorted(args.features.glob("*.npz")):
        with np.load(path) as z:
            ds[path.stem] = {k: z[k] for k in z.files}
    pubs = sorted(k for k in ds if k.startswith("pub_"))
    assert len(pubs) == 10 and len(ds) == 38
    for k in ds:
        np.testing.assert_array_equal(ds[k]["channels"], ds[pubs[0]]["channels"])
        assert ds[k]["f"].shape[0] == len(ds[k]["idx"]) and np.all(np.diff(ds[k]["idx"]) > 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    tasks = {"pub": public, "sch": scherer, "net": netbci}
    modes = {"pub": ("return", "new"), "sch": ("return", "new", "publiconly", "single", "wrong"),
             "net": ("return", "new", "single")}[args.corpus]
    if args.smoke:
        modes = ("return",)
    full = Engine(ds, pubs)
    for mode in modes:
        subset = sorted(k for k in ds if k.startswith(args.corpus + "_"))
        plan = list(tasks[args.corpus](args, ds, mode in ("return", "single")))
        byperson = sorted({s.person for s in plan})
        for person in byperson:
            engine = full
            if args.corpus == "pub" and mode == "new":
                engine = Engine(ds, [k for k in pubs if k != person])
            names = sorted(set(pubs + (subset if mode != "publiconly" else [])))
            if mode not in ("return", "single") and person in names:
                names.remove(person)
            if mode == "single":
                names = pubs + [person]
            assert (person in names) == (mode in ("return", "single"))
            if mode == "wrong":
                # Genuine wrong-enrollment archives exist for full:0 on both visits, every ordered pair.
                donors = [k for k in subset if k != person]
                for donor in donors:
                    i = subset.index(donor)
                    source = args.niv / f"fa_cap250_k{i}" / f"p{i}_c0.npz"
                    with np.load(source) as z:
                        streams = [s for s in plan if s.person == person and s.scheme == "full" and s.seed == 71000]
                        assert len(streams) == 2
                        for s in streams:
                            tag = f"full|0|Scherer2015/{person[4:]}|{s.day}"
                            from dataclasses import replace
                            wrong = read_labels(z, tag, s.rows)
                            control = replace(s, stock=wrong)
                            route, pred = engine.evaluate(control, pubs + [donor])
                            suffix = f"full_71000_{s.day}"
                            pair = {suffix + "|" + k: a for k, a in
                                    dict(rows=s.rows, stock=wrong, new=s.new, heldnew=s.heldnew, **pred).items()}
                            pair["routing"] = np.array(json.dumps({suffix: route}))
                            np.savez_compressed(args.out / f"wrong_{person}_{donor}_{s.day}.npz", **pair)
                continue
            out, routing = {}, {}
            for s in (s for s in plan if s.person == person):
                tag = f"{s.scheme}_{s.seed}_{s.day}"
                route, pred = engine.evaluate(s, names, args.smoke)
                routing[tag] = route
                for k, a in dict(rows=s.rows, stock=s.stock, new=s.new, heldnew=s.heldnew, **pred).items():
                    out[tag + "|" + k] = a
                if args.smoke:
                    # Exercise trained personal decoders even if the gate correctly abstains here.
                    expert = engine.pop.personalize(ds[person])
                    x = ds[person]["f"][np.searchsorted(ds[person]["idx"], s.rows)]
                    p = expert.predict(x[:40])
                    np.testing.assert_allclose(p[:20], expert.predict(x[:20]), atol=2e-5)
                    assert np.max(p.std(0)) > 1e-3 and np.max(abs(p[:, 0] - p[:, 1])) > 1e-3
                    print("SMOKE OK", person, tag, route, flush=True)
                    return
            out["routing"] = np.array(json.dumps(routing))
            path = args.out / f"{mode}_{person}.npz"
            np.savez_compressed(path, **out)
            print("blind", mode, person, len(routing), round(time.monotonic() - start, 1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus", choices=("pub", "net", "sch"))
    for arg in ("features", "stock", "population", "public", "scherer", "manifest", "niv", "b10", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())
