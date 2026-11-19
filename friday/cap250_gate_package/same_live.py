"""Prove the shipped selector ZIP against exact current inputs on a clean worker."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from smoke_gate import REFERENCE_SHA, digest, frames, fresh_model, load_paths, pinned


def exact(left, right, stage):
    np.testing.assert_array_equal(left, right, err_msg=stage)


def direct_trace(model, windows):
    selector = model.predict.__func__.__globals__["SharedCovRetiringSelector"]
    initialize = selector.__init__

    def capture(instance, *args, **kwargs):
        if kwargs:
            raise ValueError("unexpected shipped selector initialization options")
        initialize(instance, *args, audit=True)

    with patch.object(selector, "__init__", capture):
        outputs = [model.predict(frame[None]) for frame in windows]
    assert all(x.shape == (1,) and x.device.type == "cpu" and x.dtype == torch.long for x in outputs)
    return torch.cat(outputs).numpy(), model.selector


def native_mirrors(namespace, stock_model, events):
    original = namespace["Outer"](fresh_model(stock_model))
    alt_model = fresh_model(stock_model)
    hook = namespace["FBL2Hook"](alt_model, "fbl2")
    alternate = namespace["Outer"](alt_model)
    for x, u, p, core, extra, contrast, identity, recognized, members, warm in events:
        boosted = hook.project(p, members, warm)
        original.step((x, u, p, p, core, extra, contrast, identity, recognized), "stock")
        alternate.step((x, u, p, boosted, core, extra, contrast, identity, recognized), "hard")
    a = np.stack((original.raw, original.count, original.terminal), axis=1)
    b = np.stack((alternate.raw, alternate.count, alternate.terminal), axis=1)
    return a, b, original, alternate


def prove_fixture(reference, stock, deployed, name, windows, peer):
    trial = fresh_model(deployed)
    labels, live = direct_trace(trial, windows)
    own, private, selected = live.stages()
    proof = live.summary()

    exact(np.stack([e[0] for e in live.events]),
          windows.cpu().numpy(), "real raw input")
    old, full_alt, old_work, alt_work = native_mirrors(reference, stock, live.events)
    length = len(windows)
    rec = 10 if name == "recognized" else -1
    kept = rec if rec >= 0 else length
    expected_choice = old[:, 2].copy()
    expected_choice[20:kept] = full_alt[20:kept, 2]
    exact(own, old, "native stock vectors")
    exact(private, full_alt[:kept], "live alternate prefix")
    exact(selected, expected_choice, "shipped selected vectors")
    exact(labels, expected_choice.argmax(1), "shipped decisions")
    assert (old_work.cp.refits, old_work.cp.fused) == (trial.stock.cp.refits, trial.stock.cp.fused)
    assert proof["alternate_rows"] == proof["shared_calls"] == proof["verified_covariances"] == kept
    assert proof["retired_at"] == (rec if rec >= 0 else None)
    assert proof["eligible"] == max(0, kept-20) and proof["recognized"] == rec
    assert trial.stock.ft.steps == (0 if rec >= 0 else 200)
    if rec < 0:
        assert kept == 120
        assert (alt_work.cp.refits, alt_work.cp.fused) == (5, 30)
    for reset in (fresh_model(trial), trial):
        reset.reset_state()
        reset.predict(peer)
        reset.reset_state()
        label = reset.predict(windows[:1])
        assert reset.stock.cp.t == reset.selector.age == 1
        exact(label.numpy(), labels[:1], "interleaved reset decision")
        exact(reset.selector.probability, expected_choice[0], "interleaved reset vector")
    return dict(rows=length, recognized=rec, kept=kept, eligible=proof["eligible"], shared=proof["shared_calls"],
                exact_stock_alternate_and_selected=True, first=int(labels[0]), actual_work=proof["alternate_refits"],
                fuse=proof["alternate_fused"])


def repeated_stock(reference, stock, windows):
    runs = []
    for _ in range(2):
        gate = reference["LiveSelector"](fresh_model(stock), audit=True)
        y = [int(gate.predict(row[None]).item()) for row in windows]
        runs.append((gate.stages(), np.asarray(y), gate.summary()))
    old, new = runs
    changed = int(np.count_nonzero(old[1] != new[1]))
    for key in ("rows", "recognized", "eligible", "alternate_refits", "alternate_fused", "alternate_fit_ages"):
        assert old[2][key] == new[2][key]
    effects = {}
    for label, a, b in zip(("stock", "alternate", "selected"), old[0], new[0], strict=True):
        distances = np.abs(a-b).reshape(len(windows), -1).max(1)
        effects[label] = dict(max=float(distances.max()), age=int(distances.argmax()), age106=float(distances[106]))
    return dict(vectors=effects, changed_labels=changed)


def main(args):
    assert torch.__version__ == "2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    ref = {"np": np, "torch": torch}
    exec(compile(pinned(args.reference, REFERENCE_SHA).read_text(), "<signed-original-live-source>", "exec"), ref)
    proof = {}
    with TemporaryDirectory() as scratch:
        stock, deployed = load_paths(args.stock, args.trial, scratch, "cuda:0")
        pairs = [(name, windows.to("cuda:0")) for name, windows in
                 frames(args.public, args.external, args.known, args.unknown, external_rows=120)]
        assert [len(x) for _, x in pairs] == [31, 120]
        warm = fresh_model(stock)
        for row in pairs[1][1]:
            warm.predict(row[None])
        del warm
        for index, (name, x) in enumerate(pairs):
            proof[name] = prove_fixture(ref, stock, deployed, name, x, pairs[1-index][1][:1])
        print(json.dumps(dict(same_live_passed=True, proof=proof, held_labels_opened=False),
                         sort_keys=True, allow_nan=False), flush=True)
        repeat = repeated_stock(ref, stock, pairs[1][1])
    result = dict(archive=digest(args.trial), proof=proof, repeat=repeat, held_labels_opened=False)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("stock", "trial", "public", "external", "known", "unknown", "reference"):
        parser.add_argument(field, type=Path)
    main(parser.parse_args())
