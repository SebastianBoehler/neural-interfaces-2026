"""Authenticate the two within0 target and historical warm prerequisites."""

import json
from pathlib import Path


BASE = Path("/mnt/lustre/work/gehler/gfh527")
CAMP = BASE / "ni26_friday_causal_rebuild_20261118"
BRIDGE = CAMP / "within0_sinc_bridge_v1/status.json"
BRIDGE_SHA = "73786ae9962346680109c1b8d8c6aabe9ed194e7ab1a5a8971b88b9917a8b055"
WARM = CAMP / "within0_warm_v2/qualification.json"
FAMILIES = ("fbraw", "fb", "sinc", "fbl")
ORDER = "within0"
ACTOR_SHA = "a3d227069dd9989bf77b1c1feddb3ce221e75687d354ef17caab0df12b335ca6"


def authenticate(pin, warm_sha):
    pin(WARM, warm_sha)
    old = json.loads(WARM.read_text())
    if (old["order"] != ORDER or old["held_labels_opened"] or old["legacy_sinc_qualified"]
            or (old["rows"], old["visits"]) != (2655, 14) or set(old["families"]) != {"fbraw", "fb", "fbl"}
            or any(not f["archive_parity"] or f["warm_changed"] or f["warm_max_abs"] > 2e-5
                   for f in old["families"].values())):
        raise ValueError("legacy within0 warm controls are incomplete")
    pin(BRIDGE, BRIDGE_SHA)
    proof = json.loads(BRIDGE.read_text())
    if (proof["order"] != ORDER or proof["held_labels_opened"] or not proof["targets_sealed"]
            or not proof["exact_historical_zero"] or proof["recognition"]["bio_finite_scores"] != 2515
            or any(v["routed_sessions"] or v["sessions"] != 14 for v in proof["recognition"]["routing"].values())):
        raise ValueError("within0 fresh shared upstream set differs")
    pin(Path(proof["bridge"]["path"]), proof["bridge"]["sha256"])
    for info in proof["inputs"].values():
        pin(Path(info["path"]), info["sha256"])
    return proof
