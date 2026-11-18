"""Pin the conditional §321 bundle, both EA banks, and native reference outputs."""

import hashlib
from pathlib import Path

import numpy as np

from assets import ANCHORS, BRIDGE, load_rows, meta, sha, source_fit


BASE = Path("/mnt/lustre/work/gehler/gfh527")
BUNDLE = BASE / "ni26_js321_20261117_returning_bundle2"
PRIOR = BASE / "ni26_js321_20261117_returning_pair_cpu1"
EA = BASE / "ni26_friday_ea_returning_fresh_v2_20261118"
FRESH_WARM = BASE / "ni26_friday_returning_fresh_warm_v1_20261118"
ARM_NAMES = ("native_zero", "recording", "causal")
GROUPS = {1: (20, 21), 3: (0, 1, 2, 3, 4)}
ZERO = {"P04": (4, "90f4243071c790f8fe571d4b8caf90442adcc0c19dcbc7bc594982bd92422fd1",
                "81ff3c3d77c91f7978470e06442b72821d675825a693f5feea430ffc0254a619"),
        "P10": (12, "416043ee6e06e408a3553d3d0915ab53ff1fa343121efa60b39bae18e58f8a56",
                "33366f2751cce8c0e91c0504c966b2d5231b1b228344f2e9164d1a57b32e2b51")}
BASELINES = {"P04|2": 10, "P04|3": 10, "P10|2": 28, "P10|3": 10}
BUNDLE_SHA = "4141e13721928b21248af7c1eedcd10a18f20c2293631aeaa529bad619d13aaa"
PRIOR_SHA = "933e49cc29bc74526e39bff707fab9887e4590c018d4a681f6e835071cef3f17"


def pin(path, want):
    got = sha(path)
    if got != want:
        raise ValueError(f"asset differs: {path}")
    return got


def ea_means(order):
    idx = np.sort(order)
    values = {}
    for slot, family, seeds in ((1, "fbcnet_ea_tf", 5), (3, "shallow_ea", 6)):
        with np.load(EA / f"{family}_seen_session_orders.npz", allow_pickle=False) as bank:
            k = bank["orders"].tolist().index("full0")
            np.testing.assert_array_equal(bank["index"], idx)
            np.testing.assert_array_equal(bank["delivery"][k], order)
            np.testing.assert_array_equal(bank["seeds"], np.arange(seeds))
            values[slot] = {"recording": bank["recording"].mean(0), "causal": bank["proba"][:, k].mean(0)}
    with np.load(BRIDGE / "matched.npz", allow_pickle=False) as checked:
        np.testing.assert_array_equal(checked["index"], idx)
        for slot, arms in values.items():
            for arm, p in arms.items():
                np.testing.assert_array_equal(p, checked[f"fresh_{arm}|slot{slot}"])
                if p.shape != (762, 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=2e-6):
                    raise ValueError("invalid same-fit EA mean")
    return values


def bundle_and_fit(who, people, day):
    fit = source_fit(who, people, day)
    path = BUNDLE / fit["fold"] / "base"
    pin(BUNDLE / "bundle_complete.json", BUNDLE_SHA)
    bundle = meta(BUNDLE / "bundle_complete.json")
    expected = next(a for a in bundle["artifacts"] if a["path"] == fit["fold"] + "/base")
    pin(path / "submission.py", expected["module_sha256"])
    for name, digest in expected["files"].items():
        pin(path / name, digest)
    with np.load(path / "weights.npz", allow_pickle=False) as weights:
        for key, value in fit["core"].items():
            np.testing.assert_array_equal(weights[f"ts_{key}"], value)
        for key, value in fit["params"].items():
            np.testing.assert_array_equal(weights[key], value)
        for key, value in (("tsem_m", fit["m"]), ("tsem_V", fit["low"]), ("route_V_hi", fit["high"])):
            np.testing.assert_array_equal(weights[key], value)
        names = weights["route_names"].tolist()
        native_w = weights["member_weights"]
        shallow = tuple(i for i in range(65) if f"net{i}_time_w" in weights and weights[f"net{i}_align"] and native_w[i + 1] > 0)
        fbc = tuple(i for i in range(65) if f"net{i}_fbc_n_bands" in weights and f"net{i}_fbl_win" not in weights
                    and weights[f"net{i}_align"] and native_w[i + 1] > 0)
        if (fbc, shallow) != (GROUPS[1], GROUPS[3]):
            raise ValueError("declared live native EA groups differ")
    return path, fit, names, native_w


def rows_and_zero(who, day, data, fit):
    x, people, visit, order, _ = data
    selector = (people[order] == f"Dreyer2026Proteus/{who}") & (visit[order] == day)
    rows = order[selector]
    if rows[0] != ANCHORS[f"{who}|{day}"]:
        raise ValueError("native returning delivery mismatch")
    number, digest, bank_sha = ZERO[who]
    root = PRIOR / f"cell_{number}"
    pin(root / "proof.json", digest)
    pin(root / "predictions.npz", bank_sha)
    parent = meta(root / "proof.json")
    if (parent["fold"] != fit["fold"] or parent["order"] != "full0" or parent["query_labels_read"]
            or parent["accuracy_computed"] or parent["bundle_manifest_sha256"] != BUNDLE_SHA
            or parent["source_sha256"] != PRIOR_SHA or parent["bank_sha256"] != fit["view_sha"]
            or parent["predictions_sha256"] != bank_sha):
        raise ValueError("native §321 conditional zero differs")
    loc = np.searchsorted(fit["held"], rows)
    np.testing.assert_array_equal(fit["held"][loc], rows)
    with np.load(root / "predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["index"], fit["held"])
        case = dict(rows=rows, x=x[rows], score=z["base_scores"][loc], labels=z["base_labels"][loc],
                    u=fit["reference"]["view256"][loc], core=fit["reference"]["proba"][loc])
    p = np.searchsorted(np.sort(order), rows)
    means = ea_means(order)
    case["ea"] = {arm: {s: means[s][arm][p] for s in GROUPS} for arm in ARM_NAMES[1:]}
    return case


def native_ft_proof(model, who):
    if set(model.ft.member) != {"fb", "fbraw", "sinc", "fbl"}:
        raise ValueError("native package no longer has four warm families")
    if who != "P10":
        return
    for family, i in (("fbraw", 47), ("sinc", 53)):
        old = meta(FRESH_WARM / family / "fresh_warm.json")
        if (not old["no_gold"] or not old["same_fresh_actor"] or model.ft.member[family] != i
                or hashlib.sha256(model.ft.blobs[family]).hexdigest() != old["trace"]):
            raise ValueError(f"the native {family} blob differs from D-EArW")


def load(who):
    data = load_rows()
    path, fit, names, weights = bundle_and_fit(who, data[1], data[2])
    path_data = BASE / "neural-interfaces-2026/data/proteus_windows.npz"
    with np.load(path_data, allow_pickle=False) as z:
        channels = z["ch_names"]
    return path, fit, names, weights, channels, {v: rows_and_zero(who, v, data, fit) for v in ("2", "3")}
