"""Pin the original production traces and independent returning decision tapes."""

import json
import zipfile

import numpy as np

from eye_scope import BASE, BUNDLE, INNER, ORIGINAL_TRACES, VISITS, pin, sha


ZIP = BASE / "ni26_s318_coral_fbl_20261117/runs/sealed/final_zips/v66_cp1_cap250.zip"
ZIP_SHA = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
INNER_SOURCE = "3f513cf5e0fd9f2db3cfdb15c9f09ae1d09e8bbbf9092b0f2e3176444fed8626"
VISIT_PINS = (
    ("f3d353ba309a7fc13e97b41a419ca11e900f351734cc68f73ef9c9e672f48aeb",
     "3f7937f9959cf85718902486c48a454b38bb084e2eb5d67096668b9674b72e1c"),
    ("4ff3a26e12d6e9f468634faab0bf1160bc664e4457afb55f7d33c1b9746e3897",
     "ff642b6bdd78f14e6208fcbf633482188f534984d3e241e45c6a2c003354f4a9"),
    ("82704d2bb84b13be8d26292f232213b1ce7c54eb477e2056d3817e7259a16cf6",
     "434c1c30520d598c01cb71d69213a91b58f9118f949765d67963662f63e30af7"),
    ("d482a6546310dbc7de246e74294fb8970a8ef93bfee100ebb30c2ac3edd16a85",
     "4aef450617748341b65d023fd3154dc0327a724b8f225d233bf51a634a48f89a"),
)
GALLERY = {
    "P04": "b6d0630745c6a71e40d34f88f84d7ec51abd1bfa6cee824daf5311aa1e5b298c",
    "P10": "f5c8804a876a6f49f9580dce4b941e12f39d20c278d9678ee3b133a8b2051873",
}


def original_traces(who, root):
    import hashlib

    if root != BUNDLE / f"{who}_s23/base":
        raise ValueError("expected original §321 returning base")
    with np.load(root / "weights.npz", allow_pickle=False) as z:
        names = list(map(str, z["v6_eye_files"]))
        if names != [f"eye_s{i}.pt" for i in range(3)] or int(z["v6_ft_labels"]) != 0:
            raise ValueError("native Eye slot or warm-label convention changed")
    pin(ZIP, ZIP_SHA)
    with zipfile.ZipFile(ZIP) as z:
        for name, digest in zip(names, ORIGINAL_TRACES, strict=True):
            pin(root / name, digest)
            items = [k for k in z.namelist() if k.rsplit("/", 1)[-1] == name]
            if len(items) != 1 or hashlib.sha256(z.read(items[0])).hexdigest() != digest:
                raise ValueError("native returning Eye file differs from production")
    return names


def reference(slot):
    root = INNER / f"visit{slot}"
    receipt, path = root / "native_inner.json", root / "native_inner.npz"
    expected, digest = VISIT_PINS[slot]
    pin(INNER / "source.sha256", INNER_SOURCE)
    pin(receipt, expected)
    pin(path, digest)
    rec = json.loads(receipt.read_text())
    who, day, n = VISITS[slot]
    if (rec["output"] != digest or rec["n"] != n or rec["visit"] != f"{who}|{day}"
            or rec["held_labels_opened"] or rec["accuracy_computed"]):
        raise ValueError("independent exact D-EArN reference scope differs")
    return path, expected


def gallery(model, who, source):
    root = BASE / "ni26_js321_20261117_returning_galleries2" / f"{who}_s23"
    path = root / "proof.json"
    pin(path, GALLERY[who])
    proof = json.loads(path.read_text())
    if (proof["source_basis_sha256"] != source or proof["query_labels_read"]
            or not proof["old_OOF_outer_exposure_RETAINED"]):
        raise ValueError("native calibrated gallery scope disagrees")
    bank = root / "galleries.npz"
    pin(bank, proof["arrays_sha256"])
    with np.load(bank, allow_pickle=False) as z:
        np.testing.assert_array_equal(model.route.contrast.Qb, z["route_Q"])
    return sha(path)
