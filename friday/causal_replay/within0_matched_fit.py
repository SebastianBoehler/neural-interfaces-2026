"""Restore each within0 seed-0 actor and replay both real causal refit targets."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from fresh_sinc import state_hash
from member_inputs import ORDERS, pinned_delivery
from paired_refit_smoke import target_table
from refit_smoke import HERE, ROOT, TOL, changed, file_sha, waveform_data
from restore_refit import RELEASE, restore
from within0_frozen_probe import PEOPLE, rows_sha
from within0_pair_inputs import ACTOR_SHA, BRIDGE_SHA, FAMILIES, ORDER, authenticate
from within0_warm import FTREL, REFIT_NAME, actor, reference, run_stream


ORIGIN = ROOT.parent / "ni26_friday_sinc_seed0_fresh_full0_v3_20261118"


def pin(path, expected):
    if file_sha(path) != expected:
        raise ValueError(f"digest differs {path}")


def fit_inputs(family, proof, p):
    if family != "sinc":
        reference_rows = reference(family)["base"]
        states = json.loads((RELEASE / "members" / f"{family}_s0.json").read_text())["fit_states"]
        paths = {person: (RELEASE / "checkpoints" / f"{family}_s0_{person}.pt",
                           row["sha256"], row["fit_rows_sha"]) for person, row in states.items()}
        if family == "fbl":
            root = HERE.parent / "within0_warm_recovery_v1"
            for person, selected in proof["recovered_fbl"].items():
                checkpoint = root / f"fbl_s0_{person}.pt"
                pin(checkpoint.with_suffix(".json"), selected["receipt_sha256"])
                if (Path(selected["checkpoint"]) != checkpoint
                        or selected["fit_rows_sha"] != states[person]["fit_rows_sha"]):
                    raise ValueError("recovered FBLight a different source fold")
                paths[person] = checkpoint, selected["checkpoint_sha256"], selected["fit_rows_sha"]
        return reference_rows, paths
    bank = proof["bank"]
    pin(ORIGIN / "frozen.json", bank["origin_receipt_sha256"])
    origin = json.loads((ORIGIN / "frozen.json").read_text())
    if origin["held_labels_opened"] or bank["folds"] != origin["folds"]:
        raise ValueError("fresh Sinc source folds differ from the common input")
    path = Path(bank["path"])
    pin(path, bank["bank_sha256"])
    order = ORDERS.index(ORDER)
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["index"], np.arange(len(p)))
        np.testing.assert_array_equal(z["delivery"][order], p)
        if tuple(z["orders"]) != ORDERS:
            raise ValueError("fresh Sinc bank does not match the order")
        base = z["proba"][order]
    paths = {person: (ORIGIN / "folds" / f"{person}.pt", row["checkpoint_sha256"], row["fit_rows_sha256"])
             for person, row in origin["folds"].items()}
    return base, paths


def run(family, warm_sha, smoke):
    proof = authenticate(pin, warm_sha)
    if family == "sinc":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("fresh Sinc requires its deterministic workspace")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    api, stream, _ = actor("A100" if family == "sinc" else "RTX 2080 Ti")
    pin(FTREL, ACTOR_SHA)
    p = pinned_delivery(ROOT / "runs/shuffle_perms.npz")[ORDERS.index(ORDER)]
    with np.load(proof["bridge"]["path"], allow_pickle=False) as old:
        np.testing.assert_array_equal(p, old["delivery"])
    base, paths = fit_inputs(family, proof, p)
    arrays = {name: np.full((len(p), 3), np.nan) for name in ("base", "historical", "causal")}
    inputs = {"historical": proof["inputs"]["historical"], "causal": proof["inputs"]["corrected"]}
    data = waveform_data()
    ids_expected = np.flatnonzero(data["subject"] == "Dreyer2026Proteus/P01") if smoke else np.arange(len(p))
    source_sha = file_sha(HERE / "source.sha256")
    fits, per_visit = {}, {}
    for person in PEOPLE[:1] if smoke else PEOPLE:
        checkpoint, expected, fit_sha = paths[person]
        model, fit, held = restore(api, data, family, person, checkpoint, expected)
        if rows_sha(fit) != fit_sha:
            raise ValueError("source fold excludes the wrong rows")
        state = state_hash(model.net)
        delivery = p[held[p]]
        for visit in dict.fromkeys(data["stream"][delivery]):
            rows = delivery[data["stream"][delivery] == visit]
            x = torch.from_numpy(model._aligned(rows, causal=True)).to(model.dev)
            details = {}
            for arm, target in inputs.items():
                v = target_table(Path(target["path"]), target["sha256"], rows, (len(rows), 58))
                got, steps = run_stream(stream, model, x, v)
                delta = changed(got["base"], base[rows])
                if delta["changed"] or delta["max_abs"] > TOL:
                    raise ValueError(f"frozen input differs for {family}/{visit}")
                if arm == "causal":
                    np.testing.assert_array_equal(got["base"], arrays["base"][rows])
                arrays["base"][rows], arrays[arm][rows] = got["base"], got[REFIT_NAME]
                if float(np.abs(got[REFIT_NAME] - got["base"]).max()) <= 1e-3:
                    raise ValueError(f"inactive matched {arm} refit for {visit}")
                details[arm] = dict(steps=steps, base=delta)
                if smoke:
                    cut = v[:80].copy()
                    cut[:, 3:-1].reshape(80, 18, 3)[:, 20 + 10 * np.arange(18) >= 80] = np.nan
                    xp = torch.from_numpy(model._aligned(rows[:80], causal=True)).to(model.dev)
                    torch.testing.assert_close(xp, x[:80], rtol=0, atol=0)
                    prefix, _ = run_stream(stream, model, xp, cut)
                    check = changed(prefix[REFIT_NAME], got[REFIT_NAME][:80])
                    if check["changed"] or check["max_abs"] > TOL:
                        raise ValueError(f"noncausal matched {arm} prefix {check}")
                    details[arm]["prefix"] = check
            per_visit[visit] = details
        if state_hash(model.net) != state:
            raise ValueError("matched warm runs mutate the shared frozen model")
        fits[person] = dict(sha256=expected, fit_rows_sha=fit_sha)
    ids = np.flatnonzero(np.isfinite(arrays["base"]).all(1))
    np.testing.assert_array_equal(ids, ids_expected)
    values = {k: v[ids] for k, v in arrays.items()}
    delta = changed(values["causal"], values["historical"])
    if delta["max_abs"] <= 1e-4:
        raise ValueError("paired fitted targets did not reach the real output")
    path = HERE / ("smoke" if smoke else "members") / f"{family}_s0.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, index=ids, delivery=p[np.isin(p, ids)], **values)
    path.chmod(0o444)
    receipt = dict(member=family, seed=0, smoke=smoke, order=ORDER, rows=len(ids), held_labels_opened=False,
                   actor_sha256=ACTOR_SHA, source=source_sha, targets={a: v["sha256"] for a, v in inputs.items()},
                   warm_qualification_sha256=warm_sha, bridge_sha256=proof["bridge"]["sha256"],
                   bridge_status_sha256=BRIDGE_SHA, fit_states=fits, visits=per_visit, paired_refit_delta=delta,
                   output_sha256=file_sha(path), smoke_sha256=os.environ.get("NI26_WITHIN0_PAIR_SMOKE"))
    with path.with_suffix(".json").open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.with_suffix(".json").chmod(0o444)
    print("PASS matched", family, file_sha(path.with_suffix(".json")), flush=True)


def check_smoke(family, warm_sha):
    proof = authenticate(pin, warm_sha)
    path = HERE / "smoke" / f"{family}_s0.npz"
    receipt = path.with_suffix(".json")
    got = json.loads(receipt.read_text())
    expected = dict(member=family, seed=0, smoke=True, order=ORDER, rows=190, held_labels_opened=False,
                    actor_sha256=ACTOR_SHA, source=file_sha(HERE / "source.sha256"),
                    targets={"historical": proof["inputs"]["historical"]["sha256"],
                             "causal": proof["inputs"]["corrected"]["sha256"]},
                    warm_qualification_sha256=warm_sha, bridge_sha256=proof["bridge"]["sha256"],
                    bridge_status_sha256=BRIDGE_SHA)
    if (any(got[k] != v for k, v in expected.items()) or got["paired_refit_delta"]["max_abs"] <= 1e-4
            or set(got["fit_states"]) != {"P01"} or set(got["visits"]) != {"Dreyer2026Proteus/P01|1"}
            or path.stat().st_mode & 0o222 or receipt.stat().st_mode & 0o222):
        raise ValueError("matched refit smoke receipt differs")
    for details in got["visits"].values():
        for arm in ("historical", "causal"):
            r = details[arm]
            if (r["steps"] != 170 or r["base"]["changed"] or r["base"]["max_abs"] > TOL
                    or r["prefix"]["changed"] or r["prefix"]["max_abs"] > TOL):
                raise ValueError("active matched refit smoke did not qualify")
    pin(path, got["output_sha256"])
    return file_sha(receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("warm_sha")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-smoke", action="store_true")
    args = parser.parse_args()
    if args.verify_smoke:
        print(check_smoke(args.family, args.warm_sha), flush=True)
    else:
        run(args.family, args.warm_sha, args.smoke)
