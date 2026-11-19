"""Consume the original fixed-mass waveform substitution in two live matched tails."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from adapter import Adapter
from benchmark_utils.streaming import fresh_model
from fixture import ARMS, chosen
from mirror import Outer, Tap
from settings import sha, stamp
from worker import authentic_stock, stream


class PairedTap(Tap):
    """Use F-C20s native attachment and arithmetic for both fixed pupil pairs."""

    def __init__(self, model, pupils):
        self.pairs = (pupils[:, 0], pupils[:, 2])
        self.tape, self.current = [], None
        super().__init__(model, self.pairs[0])

    def mixing(self, mix, members, warm):
        assert self.current is self.raw_frame is None
        stock, trials = None, []
        for pair in self.pairs:
            self.pair = pair
            reference = super().mixing(mix, members, warm)
            origin, alternative = self.raw_frame
            assert reference is origin
            if stock is None:
                stock = origin
            else:
                np.testing.assert_array_equal(origin, stock)
            self.raw_frame = None
            trials.append(alternative)
        self.current = (stock, *trials)
        return stock

    def stepping(self, x, u, mix, core, extra, gc, gi):
        stock, hard, joint = self.current
        assert mix is stock and self.raw_frame is None
        self.current = None
        self.consumed.append(None)
        self.tape.append(deepcopy((x, u, stock, hard, joint, core, extra, gc, gi, self.model.recB)))
        return self.route(x, u, mix, core, extra, gc, gi)


def mirror(template, events, arm):
    ref = Outer(template)
    index = dict(stock=2, hard=3, joint=4)[arm]
    for event in events:
        x, u, p, _, _, core, extra, gc, gi, recognized = event
        ref.step((x, u, p, event[index], core, extra, gc, gi, recognized), "hard")
    return np.stack((ref.raw, ref.count, ref.terminal), axis=1), ref


def one_session(template, eeg, rows, pupils, smoke):
    start = perf_counter()
    model = fresh_model(template)
    assert not model.v5hist and model.cp.t == 0 and not model.dead
    protected = Adapter(model, "stock", ())
    tap = PairedTap(model, pupils)
    recorded, y = [], []
    native_cp = model.cp.step

    def observe(x, raw, count, recognized):
        p = native_cp(x, raw, count, recognized)
        recorded.append(np.stack((raw, count, p)))
        return p

    model.cp.step = observe
    for r in rows:
        decision = model.predict(eeg[int(r):int(r)+1].to("cuda:0"))
        assert decision.shape == (1,) and decision.dtype == torch.long and decision.device.type == "cpu"
        protected.record()
        y.append(int(decision.item()))
    torch.cuda.synchronize()
    stock = np.asarray(recorded)
    np.testing.assert_array_equal(stock[:, 2].argmax(1), y)
    np.testing.assert_array_equal(np.stack([e[0] for e in tap.tape]), eeg[rows].numpy())
    activity = replay.activity(model)
    assert not any(stage.dead for stage in (model, model.cf, model.cp))
    assert activity["windows"] == len(rows) == len(tap.tape) and not model.ft.over and not activity["errors"]
    assert activity["ft_steps"] == 20*activity["ft_refits"]
    assert len(protected.refits) == activity["ft_refits"] and protected.trace
    assert np.asarray(tap.warm).reshape(-1, 2).sum(0).tolist() == [model.log6["ft"]]*2
    before = protected.summary()
    zero, observer = mirror(template, tap.tape, "stock")
    np.testing.assert_array_equal(zero, stock)
    assert observer.cp.stopped == model.cp.stopped
    assert (observer.cp.refits, observer.cp.fused) == (model.cp.refits, model.cp.fused)

    labels = {"stock": y}
    stages = {}
    for arm in ARMS[1:]:
        candidate, path = mirror(template, tap.tape, arm)
        assert path.cp.stopped == model.cp.stopped
        if smoke:
            prefix = min(31, len(rows)-1)
            short, _ = mirror(template, tap.tape[:prefix], arm)
            np.testing.assert_array_equal(short, candidate[:prefix])
        labels[arm] = candidate[:, 2].argmax(1).tolist()
        stages[arm] = dict(fit_ages=path.fit, fusions=path.cp.fused,
                           terminal_change=float(np.abs(candidate[:, 2]-stock[:, 2]).max()))
    assert protected.summary() == before
    motion = np.asarray(tap.movement).reshape(len(rows), 2)
    assert np.isfinite(motion).all() and min(motion.max(0)) > 1e-6
    return dict(rows=rows.tolist(), **labels, activity=activity, stop=model.cp.stopped, native_zero_exact=True,
                exact_private_prefix=smoke, stages=stages, owner=before,
                warm=int(model.log6["ft"]), maximum_mix=dict(zip(ARMS[1:], motion.max(0).tolist(), strict=True)),
                seconds=perf_counter()-start)


def pupil_bank(path, mode, index, choices, who):
    assert not path.stat().st_mode & 0o222
    output = {}
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        assert meta == dict(source=sha("source.sha256"), mode=mode, index=index, who=who, actors=sha("actors.json"),
                            keys=list(choices), exact_detach=True, reset_prefix=True, held_labels_opened=False)
        for key, (x, rows) in choices.items():
            np.testing.assert_array_equal(z[key+"|rows"], rows)
            assert str(z[key+"|x"]) == stamp(x[rows].numpy())
            post = z[key+"|p"]
            assert post.shape == (len(rows), 3, 2, 3) and np.isfinite(post).all() and (post >= 0).all()
            np.testing.assert_allclose(post.sum(-1), 1., rtol=0, atol=1e-12)
            np.testing.assert_array_equal(post[:, 0], post[:, 1])
            output[key] = post
    return output


def run(mode, index):
    assert torch.__version__ == "2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    scope, (eeg, canonical, anchor, archive), channels, who, expected = chosen(mode, index)
    package = replay.load(Path("/package"), channels)
    assert package.routed == 1 and not package.v6r and package.r15 and package.ft.cap == 250
    cold = stream(package, (), eeg, canonical, "stock")
    authentic_stock(cold, anchor)
    source = sha("source.sha256")
    if mode == "produce":
        proof = json.loads(Path("smoke.json").read_text())
        assert proof["passed"] and proof["source"] == source and proof["actors"] == sha("actors.json")
    p = Path(f"pupils_{mode}_{index}.npz")
    pupils = pupil_bank(p, mode, index, scope, who)
    results = {}
    for key, (x, rows) in scope.items():
        record = one_session(package, x, rows, pupils[key], mode == "smoke")
        if key in expected:
            assert record["stock"] == expected[key]
        if key == "P10-3":
            assert record["stop"] == 10 and record["activity"]["recognized"] == 10
            assert record["warm"] == 0 and record["activity"]["ft_steps"] == 0
        else:
            assert record["stop"] is None and record["activity"]["recognized"] == -1
            assert record["warm"] > 0 and all(r["fusions"] > 0 for r in record["stages"].values())
        results[key] = record
        print("PASS F32x actual native", key, record["maximum_mix"], flush=True)
    assert package.cp.t == 0 and not package.v5hist
    name = Path("smoke.json" if mode == "smoke" else f"person_{index}.json")
    with name.open("x") as file:
        json.dump(dict(passed=True, mode=mode, index=index, who=who, source=source, actors=sha("actors.json"),
                       archive=archive, cold=cold[0], cold_rows=canonical.tolist(), pupil=sha(p),
                       held_labels_opened=False, device=torch.cuda.get_device_name(), results=results),
                  file, sort_keys=True, allow_nan=False)
        file.write("\n")
    name.chmod(0o444)
    print("PASS F32x worker", mode, index, sha(name), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "produce"))
    parser.add_argument("index", type=int, choices=range(9))
    args = parser.parse_args()
    run(args.mode, args.index)
