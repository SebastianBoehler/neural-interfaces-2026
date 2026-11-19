"""Require actual P10 visit 2 prefix and warm evidence before successor tasks."""

import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check(root, mode="smoke"):
    proof = root / mode / "native_inner.json"
    bank = proof.with_suffix(".npz")
    data = json.loads(proof.read_text())
    if (data["source"] != sha(root / "source.sha256") or data["output"] != sha(bank)
            or any(p.stat().st_mode & 0o222 for p in (proof, bank))
            or not all(data[k] for k in ("native_zero_exact", "prefix_reset_exact", "matched_warm_qualified", "no_gold"))
            or max(data["native_weighted_delta_max_abs"].values()) > 1e-14
            or min(data["native_slot_delta"].values()) <= .01 or data["native_mix_delta"] <= .001):
        raise ValueError("native matched execution receipt does not qualify")
    if mode == "smoke":
        if ((data["visit"], data["fold"], data["n"]) != ("P10|2", "P10_s23", 31)
                or data["strict_target_delta"] <= 1e-4 or max(data["warm_family_delta"].values()) <= 1e-5
                or any(x["refit_ages"] != [20] or x["steps"] != 20 for x in data["arms"].values())):
            raise ValueError("P10 visit 2 real matched warm scope missing")
    return data


if __name__ == "__main__":
    print("authenticated P10 visit 2 31-row warm smoke", check(Path(__file__).resolve().parent)["output"], flush=True)
