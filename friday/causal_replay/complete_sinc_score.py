"""Score the fresh matched full0 systems after authenticating the sealed decisions."""
import argparse
import json
from pathlib import Path

import numpy as np

from bridge_owner import HERE, ROOT, pin, sha
from complete_score import KEYS, score_columns
from complete_sinc_pair import PREFIX_SHA, ZERO_SHA
from complete_zero import FAMILIES
from member_inputs import PERMUTATION_SHA256


def authenticate(proof_sha):
    proof = HERE / "matched_sinc_full0.json"
    pin(proof, proof_sha)
    receipt = json.loads(proof.read_text())
    if (receipt["order"] != "full0" or receipt["source_sha256"] != sha(HERE / "source.sha256")
            or not receipt["complete"] or receipt["held_labels_opened"] or receipt["scored"]
            or receipt["fresh_sinc_seed"] != 0 or receipt["exact_zero"] != ZERO_SHA
            or receipt["active_archive_prefix"] != PREFIX_SHA or not receipt["corrected_shadows_exact"]
            or set(receipt["refits"]) != set(FAMILIES)):
        raise ValueError("complete matched fresh Sinc proof is missing")
    if any(v["exact_interleaved_prefix_checks"] != 3 or min(v["count_affected_prefix_rows"]) <= 0
           for v in receipt["terminal"].values()):
        raise ValueError("terminal stage activity or active prefixes are missing")
    bank = Path(receipt["output"]["path"])
    if bank != HERE / "matched_sinc_full0.npz":
        raise ValueError("unexpected full-system decision bank path")
    pin(bank, receipt["output"]["sha256"])
    perm = ROOT / "runs/shuffle_perms.npz"
    pin(perm, PERMUTATION_SHA256)
    with np.load(bank, allow_pickle=False) as z, np.load(perm, allow_pickle=False) as order:
        if (not np.array_equal(z["index"], np.arange(2655))
                or not np.array_equal(z["delivery"], order["full|0|all_loso"])):
            raise ValueError("sealed decision layout differs")
        labels = np.column_stack([z[key] for key in KEYS])
        if labels.shape != (2655, 4) or not np.isin(labels, (0, 1, 2)).all():
            raise ValueError("invalid sealed decisions")
        for i, key in enumerate(KEYS):
            suffix = "_terminal" if key.endswith("cap250") else "_posterior"
            if not np.array_equal(z[key + suffix].argmax(1), labels[:, i]):
                raise ValueError("sealed decisions differ from authenticated stage output")
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proof_sha")
    args = parser.parse_args()
    score_columns(authenticate(args.proof_sha), args.proof_sha, HERE / "matched_sinc_full0_score.json",
                  "FULL-SYSTEM DEV, held-new full0, B2_pool, shared fresh Sinc seed 0")
