"""Replay all active refits from the matched fresh-Sinc full0 shadow pair."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from paired_refit_smoke import TARGETS, target_table
from refit_smoke import HERE, ROOT, TOL, changed, file_sha, refit_api, waveform_data
from restore_refit import DEVICES, FAMILIES, RELEASE, old_meta, restore


FRESH = ROOT.parent / "ni26_friday_sinc_seed0_fresh_full0_v3_20261118"


def read_checked(path, expected):
    if file_sha(path) != expected:
        raise ValueError(f"unexpected artifact {path}")
    return json.loads(path.read_text())


def assets(family, bank):
    if family == "sinc":
        meta = read_checked(FRESH / "frozen.json", bank["receipt_sha256"])
        if file_sha(FRESH / "source.sha256") != bank["source_sha256"] or meta["folds"] != bank["folds"]:
            raise ValueError("fresh Sinc source proof differs from the causal bridge")
        files = {p: (FRESH / "folds" / f"{p}.pt", m["checkpoint_sha256"], m["fit_rows_sha256"])
                 for p, m in meta["folds"].items()}
        ref = FRESH / "frozen.npz"
        if file_sha(ref) != bank["bank_sha256"] or file_sha(ref) != meta["bank_sha256"]:
            raise ValueError("fresh Sinc frozen bank differs from the causal bridge")
        return ref, "full0", files
    meta, reference = old_meta(family)
    files = {p: (RELEASE / "checkpoints" / f"{family}_s0_{p}.pt", m["sha256"], m["fit_rows_sha"])
             for p, m in meta["fit_states"].items()}
    return reference, "base", files


def run(family, bridge, bridge_sha, smoke):
    proof = read_checked(bridge, bridge_sha)
    if (proof["order"] != "full0" or not proof["matched_refits_allowed"] or proof["held_labels_opened"]
            or proof["status"] != "ready_for_matched_refits"
            or any(v["routed_sessions"] for v in proof["recognition"]["routing"].values())):
        raise ValueError("missing a qualified unrouted fresh Sinc bridge")
    api, stream, source = refit_api(DEVICES[family])
    if family == "sinc":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    root, key, paths = assets(family, proof["bank"])
    with np.load(root, allow_pickle=False) as bank:
        expected, delivery = bank[key], bank["delivery"]
        np.testing.assert_array_equal(bank["index"], np.arange(2655))
    inputs = {arm: proof["inputs"]["corrected" if arm == "causal" else arm] for arm in TARGETS}
    data = waveform_data()
    arrays = {name: np.full((2655, 3), np.nan) for name in ("base", *TARGETS)}
    people = ("P01",) if smoke else tuple(sorted(paths))
    frozen_error = 0.
    for person in people:
        model, fit, held = restore(api, data, family, person, *paths[person][:2])
        if hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest() != paths[person][2]:
            raise ValueError("refit source exclusion differs from checkpoint receipt")
        roster = delivery[held[delivery]]
        np.testing.assert_array_equal(np.sort(roster), np.flatnonzero(held))
        for session in dict.fromkeys(data["stream"][roster]):
            rows = roster[data["stream"][roster] == session]
            x = torch.from_numpy(model._aligned(rows, causal=True)).to(model.dev)
            with torch.no_grad():
                base = torch.softmax(model.net(x)[:, :3].double(), 1).cpu().numpy()
            err = changed(base, expected[rows])
            if err["changed"] or err["max_abs"] > TOL:
                raise ValueError(f"{family} {person} restored frozen output differs: {err}")
            frozen_error = max(err["max_abs"], frozen_error)
            arrays["base"][rows] = base
            for arm, bank in inputs.items():
                target = target_table(Path(bank["path"]), bank["sha256"], rows, (len(rows), 58))
                replay = stream(model, x, target, 0)["r15c250"]
                np.testing.assert_array_equal(replay[:20], base[:20])
                if float(np.abs(replay - base).max()) <= 1e-3:
                    raise ValueError(f"inactive {family} {session} {arm} refit")
                if smoke:
                    compare = changed(stream(model, x[:80], target[:80], 0)["r15c250"], replay[:80])
                    if compare["changed"] or compare["max_abs"] > TOL:
                        raise ValueError(f"independent new-target prefix differs: {compare}")
                arrays[arm][rows] = replay
        print(f"PASS matched restored {family} {person}", flush=True)
    ids = np.flatnonzero(np.isfinite(arrays["base"]).all(1))
    if len(ids) != (190 if smoke else 2655):
        raise ValueError("restored source bank is incomplete")
    arrays = {name: value[ids] for name, value in arrays.items()}
    delta = changed(arrays["causal"], arrays["historical"])
    out = HERE / ("smoke" if smoke else "members") / f"{family}_s0.npz"
    with out.open("xb") as handle:
        np.savez_compressed(handle, index=ids, delivery=delivery[np.isin(delivery, ids)], **arrays)
    with out.with_suffix(".json").open("x") as handle:
        json.dump(dict(member=family, seed=0, rows=len(ids), order="full0", smoke=smoke,
                       held_labels_opened=False, actor_sha256=source,
                       bridge_sha256=proof["bridge"]["sha256"], bridge_status_sha256=bridge_sha,
                       output_sha256=file_sha(out), archived_frozen_max_abs=frozen_error,
                       paired_refit_delta=delta, targets={a: v["sha256"] for a, v in inputs.items()},
                       fit_states={p: dict(sha256=paths[p][1], fit_rows_sha=paths[p][2]) for p in people}),
                  handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("bridge_proof", type=Path)
    parser.add_argument("bridge_sha")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.family, args.bridge_proof, args.bridge_sha, args.smoke)
