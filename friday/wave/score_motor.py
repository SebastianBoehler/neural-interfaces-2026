"""Authenticate the matched PhysioNet encoder lineage for held PROTEUS fits."""
import hashlib
import json
from pathlib import Path
import re


MOTOR_KINDS = ("motor_transfer", "motor_null")
EXTERNAL_PEOPLE = {f"PhysionetMI/{idx}" for idx in range(1, 31)}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pretraining(kind, seed):
    pretrain = Path("pretrain")
    source = sha(pretrain / "pretrain.sha256")
    inputs_manifest = pretrain / "physionet.sha256"
    entries = [line.split(maxsplit=1) for line in inputs_manifest.read_text().splitlines()]
    if (len(entries) != 3 or any(len(e) != 2 or not re.fullmatch(r"[a-f0-9]{64}", e[0]) for e in entries)
            or len({e[1] for e in entries}) != 3):
        raise ValueError("pretraining must reference the three distinct authorized PhysioNet shards")
    inputs = sha(inputs_manifest)
    checkpoint = pretrain / f"{kind}_{seed}_encoder.pt"
    receipt = checkpoint.with_suffix(".json")
    record = json.loads(receipt.read_text())
    expected = dict(kind=kind, seed=seed, steps=1200, checkpoint=sha(checkpoint),
                    actor=source, inputs=inputs)
    if any(record.get(field) != value for field, value in expected.items()):
        raise ValueError(f"pretraining lineage differs: {receipt}")
    if set(record["people"]) != EXTERNAL_PEOPLE or len(record["people"]) != len(EXTERNAL_PEOPLE):
        raise ValueError("unexpected PhysioNet pretraining cohort")
    return dict(pretrain=expected["checkpoint"], pretrain_source=source,
                pretrain_inputs=inputs, pretrain_receipt=sha(receipt))
