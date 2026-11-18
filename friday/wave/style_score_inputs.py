"""Authenticate the independently scored F7 unaugmented control for F8."""

import json

from score_motor import sha


STYLE_KINDS = ("raw_style_full", "raw_style_diagonal")
F7_ACTOR = "1c246675c86008fe9e1aaa09d1e41824460359145871a39a65d8da634b3348e1"
F7_SCORER = "b419a5523af761f056e6e211bb03e1d213c36de6dc086395cdac3e2fa700e17c"


def f7_bank(work, seed, people, committed):
    root = work / "ni26_friday_wave_raw_20261118"
    if sha(root / "actor.sha256") != F7_ACTOR or sha(root / "score.sha256") != F7_SCORER:
        raise ValueError("F7 unaugmented release changed")
    stem = f"raw_supervised_{seed}"
    proof_path = root / f"{stem}_before_labels.json"
    proof_hash = sha(proof_path)
    proof = json.loads(proof_path.read_text())
    if proof["source"] != F7_SCORER or set(proof["actors"]) != set(people):
        raise ValueError("F7 score source or held fold roster changed")
    for held in people:
        checkpoint = root / "checkpoints" / f"{stem}_{held}.pt"
        receipt = checkpoint.with_suffix(".json")
        actor = json.loads(receipt.read_text())
        expected = dict(actor=F7_ACTOR, kind="raw_supervised", seed=seed, held=held, steps=600,
                        checkpoint=sha(checkpoint), held_labels_opened=False)
        if any(actor[key] != value for key, value in expected.items()):
            raise ValueError(f"F7 unaugmented checkpoint mismatch for {held}")
        if sorted(actor["train_people"]) != sorted(set(people) - {held}):
            raise ValueError(f"F7 source roster mismatch for {held}")
        if proof["actors"][held] != dict(checkpoint=expected["checkpoint"], receipt=sha(receipt)):
            raise ValueError(f"F7 locked decisions use a different actor for {held}")
    return committed(stem, root, proof_hash)
