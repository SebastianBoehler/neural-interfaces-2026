"""Capture one live forward and score both A1 readouts through the quarter tail."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from manas_head_patch import manas_slot, patch_main, route_slots
from returning_diet_worker import DATA_SHA, DEVICE, ROSTER, V, check, diet, fixture_assets, fresh, load, open_fold
from returning_diet_worker import modified, pin, stage, tap
from returning_diet import patch as patch_diet
from returning_em import patch as patch_em
from tape import replay


def capture(template, signals, rows, names):
    alternate, baseline_people, main_people = [], [], []

    def fresh_with_readouts(parent):
        m = fresh(parent)
        actor = m.team_a
        original = m._team_a_probability
        selected = None
        slots = route_slots(names, tuple(sorted(names)))

        @torch.inference_mode()
        def probability(raw):
            nonlocal selected
            stock = original(raw)
            baseline_people.append(actor.person if actor.person is not None else -1)
            if selected is None and m.recB:
                if m.log6["rec"] != actor.age - 1:
                    raise ValueError("MANAS did not see the first MAIN latch")
                gate = m.route.last
                selected = manas_slot(slots, gate["contrast"], gate["identity"])
            main_people.append(selected if selected is not None else -1)
            if selected is None or selected == actor.person:
                alternate.append(stock.copy())
            else:
                # A1's personal term is read-only. The adapted features and generic logits stay unchanged.
                f = actor.adapter.features[-1]
                b = actor.bank
                pure = torch.nn.functional.linear(f, b["personal_weight"][selected], b["personal_bias"][selected])
                logits = 0.30 * actor.adapter.read(f) + 0.70 * pure
                q = logits.softmax(-1)[0].cpu().numpy()
                m._check_probability(q, "latched MANAS")
                alternate.append(q.copy())
            return stock

        m._team_a_probability = probability
        return m

    items = tap(template, signals, rows, fresh_with_readouts, check, DEVICE)
    alternate_items = [replace(item, team_a=q) for item, q in zip(items, alternate, strict=True)]
    return items, alternate_items, np.array(baseline_people), np.array(main_people)


def live_smoke(folder, data, signals, rows, receipt, base, candidate, quarter_route, latch, slot, quarter_source):
    prefix = rows[:max(84, latch + 8)]
    with TemporaryDirectory() as tmp:
        for name, source, result in (("base", quarter_source, base), ("main", patch_main(quarter_source), candidate)):
            root = Path(tmp) / name
            stage(root / ROSTER, folder, source)
            model, _ = load(data, root, ROSTER, receipt)

            def live_check(m, n):
                check(m, n)
                if m.log6["rec"] != latch - 1:
                    raise ValueError("live patch changed MAIN's recognition age")
                if name == "main" and m.team_a.person != slot:
                    raise ValueError("live patch did not engage the MAIN head")

            seen = tap(model, signals, prefix, fresh, live_check, DEVICE)
            np.testing.assert_array_equal([i.label for i in seen], result[0][:len(prefix)])
            np.testing.assert_allclose([i.terminal for i in seen], result[1][:len(prefix)], atol=2e-5, rtol=0)
            np.testing.assert_allclose([i.route_out for i in seen], quarter_route[:len(prefix)], atol=2e-5, rtol=0)
            del model
    print(f"PASS live quarter and MAIN source parity on {len(prefix)} windows", flush=True)


def run(a):
    if not torch.cuda.is_available() or V.digest(a.data) != DATA_SHA:
        raise ValueError("CUDA and signed PROTEUS data required")
    blocks = json.loads(a.manifest.read_text())["active"]
    block, = [b for b in blocks if b["roster"] == ROSTER and b["task"] == a.task]
    fold, phase = block["fold"], block["phase"]
    active = block["recognition"]
    first = min(int(key.split("_")[0]) for key in active)
    first -= first % 4
    if a.smoke:
        key = min(active)
        active = {key: active[key]}
    data = V.D.load(a.data)
    signals = torch.from_numpy(data["X"])
    pkg, receipts = open_fold(a.fixtures, fold, "ba-confirm")
    receipt = receipts[ROSTER]
    folder = pkg / ROSTER
    names = fixture_assets(folder, receipt)
    model, _ = load(data, pkg, ROSTER, receipt)
    if model.team_a.recipe != "A1" or len(model.team_a.bank["gallery"]) != len(names):
        raise ValueError("expected a matching A1 gallery and MAIN enrolment roster")
    payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
    with np.load(a.archive / f"{ROSTER}-{fold}-{first}.npz", allow_pickle=False) as old, \
            np.load(a.diet / phase / f"{fold}-{first}.npz", allow_pickle=False) as anchor:
        for ref in (old, anchor):
            if str(ref["receipt"]) != str(payload["receipt"]):
                raise ValueError("different assets in reference capture")
        for key, age in sorted(active.items()):
            order, visit = key.split("_")
            stream = f"{fold.split('_')[1]}|{visit}"
            rows = V.D.plans(data, int(order), [stream])[stream]
            items, arm, native_people, main_people = capture(model, signals, rows, names)
            native = np.array([item.label for item in items])
            pin(a.fixtures, fold, ROSTER, int(order), stream, rows, native, receipt)
            np.testing.assert_array_equal(replay(model, items, "w25")[0], native)
            np.testing.assert_array_equal(old[f"rows_{key}"], rows)
            np.testing.assert_array_equal(anchor[f"rows_{key}"], rows)
            np.testing.assert_array_equal(old[f"native_{key}"], native)
            np.testing.assert_array_equal(modified(model, items, "half")[0], old[f"half_{key}"])
            identity = str(old[f"identity_{key}"])
            slot = sorted(names).index(identity)
            flags = np.arange(len(rows)) >= age - 1
            np.testing.assert_array_equal(flags, [item.recognised for item in items])
            np.testing.assert_array_equal(main_people, np.where(flags, slot, -1))
            if int(old[f"rec_{key}"]) != age:
                raise ValueError("pinned recognition age differs")
            base = diet(model, items, "quarter")
            candidate = diet(model, arm, "quarter")
            np.testing.assert_array_equal(base[0], anchor[f"quarter_{key}"])
            np.testing.assert_array_equal(candidate[1][~flags], base[1][~flags])
            np.testing.assert_array_equal(candidate[2], base[2])
            n = min(len(rows), max(84, age + 8))
            for cut, full in zip(diet(model, arm[:n], "quarter"), candidate, strict=True):
                np.testing.assert_array_equal(cut, full[:n])
            if a.smoke:
                if np.max(np.abs(candidate[1] - base[1])) <= 2e-5:
                    raise ValueError("the candidate did not change the terminal posterior")
                quarter = patch_diet(patch_em((folder / "submission.py").read_text(), 0.5), 0.25)
                live_smoke(folder, data, signals, rows, receipt, base, candidate, base[2], age, slot, quarter)
            payload.update({f"rows_{key}": rows, f"quarter_{key}": base[0], f"main_{key}": candidate[0],
                            f"native_person_{key}": native_people, f"main_person_{key}": main_people})
            print(f"PASS {fold} {key}: {len(rows)} pinned labels, MAIN latched from {age} {identity}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.out) as tmp:
        target = Path(tmp) / f"{fold}-{first}.npz"
        np.savez_compressed(target, **payload)
        target.rename(a.out / target.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "fixtures", "archive", "diet", "manifest", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("task", type=int)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
