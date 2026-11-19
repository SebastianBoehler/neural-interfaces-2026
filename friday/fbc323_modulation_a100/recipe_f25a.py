"""Authenticate the independently sealed A100 hard references and F25 operators."""
import json
from pathlib import Path

import numpy as np

from anchors import sha
from recipe import BD, CANARY_SHA, CPU_CUDA, GPU, ROOT, TORCH, canary, historical, tag


WORK = Path(__file__).resolve().parent
F25 = WORK.parent / "ni26_friday_s1c_modulation_20261119"
F33A = WORK.parent / "ni26_friday_s1c_covariance_a100_whole_20261119"
F25_ACTOR = "589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c"
F33A_ACTOR = "94598f4eb485b20ea40c0d7c13f4497ed07d04e9e9c062c72953e97460b782bc"
F33A_BLIND = "f996e3a55d376f6856bbd07425632033924a175636acc21b332d1a4d3f4f0525"
F33A_SCORER = "6c8e27573d2b91176ec6d2e98068e94d8e671d878c92e515f6c58899ce558dbe"
SMOKE = {
    "bare": "f933db59f90e6cc4a297bf3c1f923d8701d48281c576f0a532be4d59b04e8e72",
    "hard": "109127232db936701bdba5c49521327f3e3ba049d78277541118d460fb302028",
}


def anchors(held, seed, smoke):
    locks = ((F25 / "actor.sha256", F25_ACTOR), (F33A / "actor.sha256", F33A_ACTOR),
             (F33A / "before_labels.sha256", F33A_BLIND), (F33A / "score.sha256", F33A_SCORER))
    if any(sha(file) != expected for file, expected in locks):
        raise ValueError("the pinned native F25 or independent F33a actor or blind seal differs")
    if smoke:
        if (held, seed) != ("P01", 0):
            raise ValueError("registered smoke requires P01 seed 0")
        pins = SMOKE
        fold = None
    else:
        rollup = dict(line.split()[::-1] for line in (F33A / "before_labels.sha256").read_text().splitlines())
        name = f"folds/{held}_s{seed}.npz"
        fold = F33A / name
        if len(rollup) != 20 or sha(fold) != rollup[name]:
            raise ValueError("unchanged independently sealed A100 held reference required")
        with np.load(fold, allow_pickle=False) as z:
            signed = json.loads(str(z["metadata"]))
        if (signed["actor"] != F33A_ACTOR or signed["scorer"] != F33A_SCORER
                or not signed["exact_bare"] or signed["held_labels_opened"]
                or (signed["held"], signed["seed"]) != (held, seed)):
            raise ValueError("sealed A100 held reference provenance differs")
        pins = {kind: signed["evidence"][kind]["receipt"] for kind in ("bare", "hard")}
    result = {}
    for kind, digest in pins.items():
        path = F33A / "checkpoints" / f"{kind}_{tag(held, seed, smoke)}.pt"
        file = path.with_suffix(".json")
        if sha(file) != digest:
            raise ValueError(f"authenticated original A100 {kind} receipt differs")
        m = json.loads(file.read_text())
        if (m["kind"] != "hard" or m["role"] != ("bare" if kind == "bare" else "matched")
                or m["actor"] != F33A_ACTOR or m["device"] != GPU or m["checkpoint"] != sha(path)
                or m["held_labels_opened"] or m["h100_weights_loaded"]):
            raise ValueError("invalid independently fitted A100 hard reference")
        result[kind] = (path, m, digest)
    a, b = (result[k][1] for k in ("bare", "hard"))
    if any(a[k] != b[k] for k in ("checkpoint", "state", "initial", "scale", "source_rows", "batches")):
        raise ValueError("independent original bare and wrapped A100 hard anchors disagree")
    return result, fold
