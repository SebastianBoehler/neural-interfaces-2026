"""Authenticate complete FBL2/stock inputs and their actual recognition paths."""

import json

import numpy as np

from eye_scope import BASE, DATA, DATA_SHA, ORDER_FILE, ORDER_SHA, pin, sha
from reference_eye import reference
from verify_fbl2 import BANK, deliveries, smokes, verify


RELEASE = BASE / "ni26_friday_c21_corrected_v2_20261119"
CAMP = BASE / "ni26_friday_causal_rebuild_20261118"
NATIVE_SOURCE = "cae74104dce81e0635fd915ef17dc09539ee7125621e06a8b0533c73a4db9ee6"
C21_SOURCE = "5b15295e48b605f87ab4855d1557393da09d0df56e67b92d1b033865de518747"
RETURN = BASE / "ni26_friday_return_fbl2_score_v2_20261119"
RETURN_SEAL = "53cfa9628a6bdc210b4fbaf1456c5309a1d51e373856e3d57b61232184c28cf0"
C21_SMOKE = "a23bffba1c60ab0c96e9e4ddb44104f8a0084f1e69f322a70a2229f351244ed8"

NEW = {
    "within0": (
        "within0_shared_bridge_v1", "within0_terminal_shared_v1", "within",
        "5874a02fa2cff86daa6d69c50bc7c0696c86f653e3430877c6e85f18d606109b",
        "9fdec9410ae0319813ec7a3443a6a6032efaf9cde6b6ef481dc42c9d71dcd9e6",
        "8dc7da81083bf117a38a17ac4f9a1333929f27b7c41c3cbd87193a4ca22390ae",
        "e7d1ec712824f33d8e8237844e9f87d6509e608170beb294316a9f9bb571d59d",
        "c41f6df763d86dee454db348560fce1943620f6ed3a3a1ef8824d4c54ac38472",
        "109a572de0f062a62ddf69d80f847d5a107e33236b45f34d7984af303df6df45"),
    "full0": (
        "full0_common_bridge_v1", "full0_common_terminal_v1", "full",
        "daa475511792886db119b03471d3aec6ed2b688ef5488b31f23f5a3daff9d277",
        "cbb39b25165ea4bfc0d7b10e9bf1c398e29f46009c4b56565dc3fed78cd6d100",
        "3e96c32dd86984df589e4414e7e039410b9df8e11521b92bce5670d27961510b",
        "d99fab293f89a9059af7ab35488a22aa3a88f151887339ffaa652efb415ce6a8",
        "442ba868279c332e550bc4cffbd29f372c1e1afdf7fb8d6e86fe1a548fd6f331",
        "5b9da9b5ba865c4b0b6bcdd2251536a472884d561bda85cfb4e624358f3f5650"),
}


def read(path, digest):
    pin(path, digest)
    return json.loads(path.read_text())


def returning():
    whole = RETURN / BANK
    p = read(whole.with_suffix(".json"), RETURN_SEAL)
    sequence = deliveries()
    if (p["source"] != NATIVE_SOURCE or p["smokes"] != smokes(sequence, NATIVE_SOURCE)
            or p["held_labels_opened"] or p["accuracy_computed"] or p["windows"] != 762):
        raise ValueError("incomplete original native F-C21r proof")
    delivered, stock, fbl2, ramps, receipts = [], [], [], [], []
    for i, rows in enumerate(sequence):
        z, proof = verify(i, False, rows, NATIVE_SOURCE)
        if proof != p["parts"][i]:
            raise ValueError("the native sealed constituent changed")
        archive, fingerprint = reference(i)
        with np.load(archive, allow_pickle=False) as old:
            np.testing.assert_array_equal(old["rows"], rows)
            live = (old["causal|contrast"][:, 2] > 0) | (old["causal|identity"][:, 2] > 0)
            latch = np.logical_or.accumulate(live)
            np.testing.assert_array_equal(latch, old["causal|latch"])
        if np.flatnonzero(latch)[0] != proof["activity"]["recognition"]:
            raise ValueError("true returning recognition differs from its own full solver path")
        delivered.append(rows)
        stock.append(z["stock_scores"])
        fbl2.append(z["fbl2_scores"])
        ramps.append(live)
        receipts.append(dict(proof=proof["receipt"], anchor=fingerprint))
    rows, stock, fbl2, ramps = [np.concatenate(x) for x in (delivered, stock, fbl2, ramps)]
    pin(whole, p["bank"])
    with np.load(whole, allow_pickle=False) as z:
        for key, a in (("rows", rows), ("stock_scores", stock), ("fbl2_scores", fbl2)):
            np.testing.assert_array_equal(z[key], a)
    return rows, stock, fbl2, ramps, dict(predictions=p["bank"], receipt=RETURN_SEAL, visits=receipts)


def held_new(order):
    bridge, terminal, scheme, route_proof, route_hash, parent_proof, parent_bank, proof, bank = NEW[order]
    route_dir, terminal_dir = CAMP / bridge, CAMP / terminal
    pin(RELEASE / "experiment.sha256", C21_SOURCE)
    smoke = read(RELEASE / "smoke.json", C21_SMOKE)
    parent = read(terminal_dir / f"matched_{order}.json", parent_proof)
    status = read(route_dir / "status.json", route_proof)
    candidate = read(RELEASE / f"{order}.json", proof)
    if (not smoke["passed"] or smoke["held_labels_opened"] or smoke["source"] != C21_SOURCE
            or parent["bridge"] != route_proof or parent["output"]["sha256"] != parent_bank
            or any(q["held_labels_opened"] for q in (parent, status, candidate))
            or not all(candidate[k] for k in ("passed","native_identity","stock_bitwise","shadow_fixed","strict_past"))
            or candidate["anchor"] != parent_proof or candidate["order"] != order or parent["order"] != order
            or candidate["rows"] != 2655 or candidate["sessions"] != 14 or candidate["source"] != C21_SOURCE
            or candidate["output"]["sha256"] != bank):
        raise ValueError("unqualified complete corrected held-new comparison")
    gate = status["recognition"]
    if (status["order"] != order or gate["routing"]["corrected"]["routed_sessions"] != 0
            or (gate["bio_finite_scores"], gate["bio_prefix_checks"], gate["bio_reset_checks"],
                gate["contrast_prefix_checks"]) != (2515, 14, 14, 28)):
        raise ValueError("actual independent native recognition proof differs")
    expected = candidate["work"]["stock"]
    current = candidate["work"]["FBL2"]
    if (expected["count_latch"] != current["count_latch"] or not any(v is not None for v in current["count_latch"].values())
            or any((q["csp_refits"], q["csp_fusions"]) != (126, 1395) or q["count_effect"] <= 0 for q in (expected,current))):
        raise ValueError("incomplete independently evolved count or CSP paths")
    paths = (RELEASE / f"{order}.npz", terminal_dir / f"matched_{order}.npz", route_dir / "route.npz")
    for path, digest in zip(paths, (bank, parent_bank, route_hash), strict=True):
        pin(path, digest)
    pin(ORDER_FILE, ORDER_SHA)
    with np.load(ORDER_FILE, allow_pickle=False) as p:
        delivery = p[f"{scheme}|0|all_loso"]
    with np.load(paths[0], allow_pickle=False) as z, np.load(paths[1], allow_pickle=False) as orig, np.load(paths[2], allow_pickle=False) as route:
        for source in (z, route):
            np.testing.assert_array_equal(source["index"], np.arange(2655))
            np.testing.assert_array_equal(source["delivery"], delivery)
        for key in ("raw", "filtered", "terminal"):
            np.testing.assert_array_equal(z[f"stock|{key}"], orig[f"corrected|cap250_{key}"])
            np.testing.assert_array_equal(z[f"identity|{key}"], z[f"stock|{key}"])
        ramps = (route["bio_ramp"] > 0) | (route["corrected|contrast_ramp"] > 0)
        if ramps.any():
            raise ValueError("held-new native recognition already fired")
        return delivery, z["stock|terminal"][delivery], z["FBL2|terminal"][delivery], ramps[delivery], dict(
            receipt=proof, predictions=bank, route_proof=route_proof, route_bank=route_hash,
            parent=parent_proof, parent_bank=parent_bank)


def metadata(rows):
    pin(DATA, DATA_SHA)
    with np.load(DATA, allow_pickle=False) as z:
        return tuple(z[k][rows] for k in ("subject", "session", "interface"))
