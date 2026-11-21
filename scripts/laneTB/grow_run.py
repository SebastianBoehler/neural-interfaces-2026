"""Real clean fold streams, pinned smoke, and counterbalanced A100 timing."""
import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from grow_sources import (ARMS, CONFIRM_ORDERS, REFERENCE, SELECT_ORDERS, banks, once,
                          team_a, team_c)
from run import check_zero_mixer, stream_session, sync, timed_load
from verify import B, D, DEVICE, FOLDS, HARD_SHA, HARD_ZIP, STORED, W, digest, fold_fixture, revised

HERE = Path(__file__).resolve().parent
DECISION = "labels.append(int(self.cp.step(x, p, self.cf.step(x, p), self.recB if self.routed else not self.active).argmax()))"
PIN = W / "laneTB-verify-21nov/out/hard/new_P01/nine25/pred_new_P01_71124.npz"


def source(original, mix, members, blob):
    text = revised(original, mix, members, blob, "HardWaveEnsemble")
    start, rest = text.split("    def step62(self, x):", 1)
    routed, end = rest.split("    def step61(self, x):", 1)
    routed = once(routed, "        mix = mix + self.w[-1] * core2",
                  "        mix = self.grow_native(mix + self.w[-1] * core2, x)")
    text = start + "    def step62(self, x):" + routed + "    def step61(self, x):" + end
    text = once(text, "            m = m - fam + w.sum() * F / F.sum()",
                "            if self.native_weight:\n"
                "                m = m + (1 - self.native_weight) * (w.sum() * F / F.sum() - fam)\n"
                "            else:\n"
                "                m = m - fam + w.sum() * F / F.sum()")
    text = once(text, "        return super().mix6(mix + delta, altered, p_ft)",
                "        return super().mix6(mix + (1 - self.native_weight) * delta, altered, p_ft)")
    text = once(text, DECISION, "labels.append(self.grow_label(x, p))")
    text = once(text, B.SOLVER, (HERE / "grow_stack.py.inc").read_text() + "\n\n" + B.SOLVER)
    return once(text, "model = TeamBlendStack(z,", "model = GrowStack(z,")


class Fixtures:
    def __init__(self, folder, data, data_sha, run, fold):
        if digest(HARD_ZIP) != HARD_SHA:
            raise ValueError("pinned hard ZIP changed")
        self.root, self.data, self.data_sha, self.run, self.fold = folder, data, data_sha, run, fold
        self.parent = folder / "hard"
        self.parent.mkdir()
        with ZipFile(HARD_ZIP) as z:
            raw = z.extract("submission.py", folder)
        fold_fixture(self.parent, Path(raw), FOLDS / fold / "pkg", "hard")
        self.original = (self.parent / "submission.py").read_text()
        self.receipt = {p.name: digest(p) for p in self.parent.iterdir()}
        self.blobs = {}

    def load(self, arm):
        w, members, stage, partner = ARMS[arm]
        if w and members not in self.blobs:
            self.blobs[members] = banks(self.run, self.fold, self.data, members)
        blob, audit = self.blobs[members] if w else (b"", {})
        pkg = self.root / arm
        pkg.mkdir(exist_ok=True)
        mix = w if stage == "mixer" else (1.0 if w else 0.0)
        text = source(self.original, mix, members, blob)
        if stage != "mixer":
            # Load the existing signed encoder, then remove the legacy hook's output mass.
            text = once(text, "    def _grow_posterior(self, raw):",
                        f"    {stage}_weight = {w!r}\n\n    def _grow_posterior(self, raw):")
        (pkg / "submission.py").write_text(text)
        (pkg / "team_b.pt").write_bytes(blob)
        for p in self.parent.iterdir():
            if p.name != "submission.py" and not (pkg / p.name).exists():
                (pkg / p.name).symlink_to(p.resolve())
        start = time.perf_counter()
        model, _ = timed_load(pkg, self.data, DEVICE)
        assert [c.__name__ for c in type(model).__mro__[:3]] == ["GrowStack", "TeamBlendStack", "HardWaveEnsemble"]
        assert type(model.ft).__name__ == "FineTune" and model.cf is not None and model.cp is not None
        if stage != "mixer":
            model.TEAM_B_MIX = 0.0
        if arm == "base":
            check_zero_mixer(model)
        receipt = dict(source_sha=B.sha(text.encode()), bank_sha=B.sha(blob), team_b=audit)
        if partner == "a":
            model.partner, receipt["partner"] = team_a(self.fold, self.data, self.data_sha)
            model.partner_kind, model.terminal_weight = "a", 0.10
        elif partner:
            model.partner, receipt["partner"] = team_c(self.run, self.fold, self.data, partner)
            model.partner_kind = "c"
        sync(DEVICE)
        assert (model.team_b is not None) == bool(w)
        return model, time.perf_counter() - start, receipt


def play(template, signals, rows, *, capture=False):
    _, official = D.official()
    sync(DEVICE)
    start = time.perf_counter()
    model = official.fresh_model(template)
    labels, mixes = [], []
    for age, row in enumerate(rows):
        result = model.predict(signals[row:row + 1].to(DEVICE))
        if result.dtype != torch.int64 or result.shape != (1,):
            raise ValueError("invalid official Solver output")
        label = int(result[0])
        assert 0 <= label <= 2
        labels.append(label)
        if capture:
            mixes.append(model.trace["v6"]["mix6"].copy())
        if age == 0:
            sync(DEVICE)
            warm_start = time.perf_counter()
    sync(DEVICE)
    assert model.routed == 1 and not model.dead and model.log6["err"] == 0 and model.log6["v6"] == len(rows)
    assert not model.cp.dead and not model.cf.dead and model.cp.log["err"] == model.cf.log["err"] == 0
    assert model.grow_calls == (len(rows) if model.team_b is not None else 0)
    if model.team_b is not None:
        assert len(model.team_b.windows) == len(rows), "Team B must consume each window exactly once"
    if model.partner_kind == "c":
        assert model.partner.n == len(rows)
        if len(rows) >= model.partner.cfg["n0"]:
            assert (model.partner.opts is not None) == bool(model.partner.cfg["steps"])
    elif model.partner_kind == "a":
        assert model.partner.age == len(rows)
    return (np.asarray(labels, dtype=np.int8), np.asarray(mixes),
            dict(setup_s=warm_start - start, warm_s=time.perf_counter() - warm_start,
                 recognition=model.log6["rec"], move=model.grow_move))


def smoke(fixture, signals):
    stream = "P01|1"
    base_path, base_order = STORED["hard"]
    cases = [("base", base_path, base_order), (REFERENCE, PIN, 71124)]
    for arm, path, order in cases:
        model, _, _ = fixture.load(arm)
        rows = D.plans(fixture.data, order, [stream])[stream]
        with np.load(path, allow_pickle=False) as pinned:
            np.testing.assert_array_equal(rows, pinned[f"rows_{order}_1"])
            got, _, _ = play(model, signals, rows)
            np.testing.assert_array_equal(got, pinned[f"labels_{order}_1"])
        print(f"{arm} equals pinned labels: {len(rows)}", flush=True)
    rows = D.plans(fixture.data, 71124, [stream])[stream][:36]
    base, _, _ = fixture.load("base")
    pristine, _ = timed_load(fixture.parent, fixture.data, DEVICE)
    old, _, _ = stream_session(pristine, signals, rows, DEVICE)
    base_y, base_m, _ = play(base, signals, rows, capture=True)
    np.testing.assert_array_equal(base_y, old)
    for arm in ARMS:
        if arm == "base":
            continue
        model, _, _ = fixture.load(arm)
        labels, mixes, meta = play(model, signals, rows, capture=True)
        fresh, fresh_m, _ = play(model, signals, rows[:30], capture=True)
        np.testing.assert_array_equal(fresh, labels[:30])
        np.testing.assert_allclose(fresh_m, mixes[:30], rtol=0, atol=2e-6)
        assert not model.team_b.windows
        if model.partner_kind == "c":
            assert model.partner.n == 0
        elif model.partner_kind == "a":
            assert model.partner.age == 0
        if arm == "terminal25":
            np.testing.assert_allclose(mixes, base_m, rtol=0, atol=2e-6)
        if arm in ("native25", "terminal25", "ba_terminal") or arm.startswith("bc_"):
            assert meta["move"] > 1e-4, (arm, meta)
        else:
            assert np.max(np.abs(mixes - base_m)) > 1e-4, arm
        print("PASS", arm, "same prefix and fresh-state parity", meta, flush=True)


def returning_smoke(fixture, signals):
    held = fixture.fold.split("_")[1]
    stream = f"{held}|2"
    rows = D.plans(fixture.data, 71124, [stream])[stream][:24]
    for arm in ("bc_a1", "bc_light", "bc_a2", "ba_terminal"):
        model, _, _ = fixture.load(arm)
        labels, mixes, meta = play(model, signals, rows, capture=True)
        prefix, prefix_mixes, _ = play(model, signals, rows[:18], capture=True)
        np.testing.assert_array_equal(labels[:18], prefix)
        np.testing.assert_allclose(mixes[:18], prefix_mixes, rtol=0, atol=2e-6)
        assert meta["move"] > 1e-4
        print("PASS", fixture.fold, arm, meta, flush=True)


def score(fixture, signals, arm_names, orders, out, resume):
    mode, held = fixture.fold.split("_")
    d = fixture.data
    sessions = ["2", "3"] if mode == "ret" else sorted(set(d["session"][d["person"] == held]))
    for arm in arm_names:
        out_arm = out / fixture.fold / arm
        target = out_arm / f"pred_{mode}_{held}_{orders[0]}.npz"
        if resume and target.is_file() and target.with_suffix(".json").is_file():
            print("Reuse completed replay", fixture.fold, arm, orders[0], flush=True)
            continue
        assert not target.exists(), target
        model, load, receipt = fixture.load(arm)
        res = {}
        for order in orders:
            for stream, rows in D.plans(d, order, [f"{held}|{s}" for s in sessions]).items():
                suffix = f"{order}_{stream.split('|')[1]}"
                labels, _, meta = play(model, signals, rows)
                res.update({f"rows_{suffix}": rows, f"labels_{suffix}": labels})
                print(fixture.fold, arm, suffix, len(rows), meta, flush=True)
        out_arm.mkdir(parents=True, exist_ok=True)
        res.update(source_sha=receipt["source_sha"], weights_sha=receipt["bank_sha"])
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **res)
        tmp.rename(target)
        target.with_suffix(".json").write_text(json.dumps(dict(receipt, base_fixture=fixture.receipt, load_s=load)) + "\n")


def timing(fixture, signals, out):
    data = {}
    for arm in ARMS:
        model, _, _ = fixture.load(arm)
        rows = D.plans(fixture.data, 71124, ["P01|1"])["P01|1"]
        play(model, signals, rows)
    # Close pair on the same node. Reverse order on the second pinned stream.
    for k, order in enumerate((71124, 71125)):
        rows = D.plans(fixture.data, order, ["P01|1"])["P01|1"]
        for arm in ARMS:
            shots = []
            for a in ((REFERENCE, arm) if k == 0 else (arm, REFERENCE)):
                model, load, _ = fixture.load(a)
                _, _, m = play(model, signals, rows)
                shots.append(dict(m, load_s=load, windows=len(rows)))
                del model
            data.setdefault(arm, []).append(shots if k == 0 else shots[::-1])
    out.write_text(json.dumps(dict(device=torch.cuda.get_device_name(), pairs=data), indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("smoke", "returning-smoke", "timing", "select", "confirm"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--chunk", type=int, choices=range(4), default=0)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    assert not a.resume or (a.phase == "select" and a.fold.startswith("ret_"))
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    d = D.load(a.data)
    signals = torch.from_numpy(d["X"])
    with TemporaryDirectory(dir=a.run) as folder:
        fixture = Fixtures(Path(folder), d, digest(a.data), a.run, a.fold)
        if a.phase == "smoke":
            assert a.fold == "new_P01"
            smoke(fixture, signals)
        elif a.phase == "returning-smoke":
            assert a.fold.startswith("ret_")
            returning_smoke(fixture, signals)
        elif a.phase == "timing":
            timing(fixture, signals, a.run / "timing.json")
        else:
            arms = tuple(ARMS)
            if a.phase == "confirm":
                chosen = json.loads((a.run / "select.json").read_text())["chosen"]
                assert chosen in ARMS and chosen != "base"
                arms = tuple(dict.fromkeys(("base", REFERENCE, chosen)))
            orders = SELECT_ORDERS if a.phase == "select" else CONFIRM_ORDERS
            score(fixture, signals, arms, orders[2 * a.chunk:2 * a.chunk + 2], a.run / a.phase, a.resume)


if __name__ == "__main__":
    main()
