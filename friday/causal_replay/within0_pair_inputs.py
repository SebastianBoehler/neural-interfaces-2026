"""Authenticate the independent legacy controls and the new shared-base within0 bridge."""

import json
from pathlib import Path


BASE = Path("/mnt/lustre/work/gehler/gfh527")
CAMP = BASE / "ni26_friday_causal_rebuild_20261118"
BRIDGE = CAMP / "within0_shared_bridge_v1/status.json"
BRIDGE_SOURCE = "5cc3a1ed8e727319d8e92ec2671f7bd2de5ef08f28c7631d1773e91ee2d5f6de"
LEGACY = CAMP / "within0_warm_v1"
LEGACY_SOURCE = "ca5607d8fd894c5dca2d3f9271d3a1f53404b9dcc794d95653adaded99c908aa"
LEGACY_RECEIPTS = {
    "members/fb.json": "cc74a8321075bb53063dbc87bc3009da18f05e5097240ed6894cd55bf1088feb",
    "members/fbraw.json": "b287db47218201df609270321085f179b772b93820c6d75de3b2be7b287c94eb",
    "smoke/fb.json": "fbee5d4af4a113d3f1bc203582026a28307f53447bd6352143df3a7e3002188f",
    "smoke/fbraw.json": "c8ac62cb5796866eca3eb923a50a20328f369249cab65452cd799e5eb9645daf",
}
FAMILIES = ("fbraw", "fb", "sinc", "fbl")
ORDER = "within0"
ACTOR_SHA = "a3d227069dd9989bf77b1c1feddb3ce221e75687d354ef17caab0df12b335ca6"


def authenticate(pin, bridge_sha):
    pin(LEGACY / "source.sha256", LEGACY_SOURCE)
    for name in ("fbraw", "fb"):
        first = f"smoke/{name}.json"
        rest = f"members/{name}.json"
        seen = set()
        for key in (first, rest):
            path = LEGACY / key
            pin(path, LEGACY_RECEIPTS[key])
            proof = json.loads(path.read_text())
            if (proof["family"] != name or proof["held_labels_opened"] or not proof["archive_parity"]
                    or proof["actor_sha256"] != ACTOR_SHA or proof["source"] != LEGACY_SOURCE
                    or proof["order"] != ORDER or seen.intersection(proof["sessions"])):
                raise ValueError("qualified historical FB warm reference differs")
            if key == rest and proof["smoke_sha256"] != LEGACY_RECEIPTS[first]:
                raise ValueError("remaining historical warm control lost its adopted smoke")
            if any(r["warm"]["max_abs"] or r["warm"]["changed"] for r in proof["sessions"].values()):
                raise ValueError("legacy FB reference is not bitwise exact")
            seen.update(proof["sessions"])
        if len(seen) != 14:
            raise ValueError("historical FB control misses a real visit")
    pin(BRIDGE, bridge_sha)
    pin(BRIDGE.parent / "source.sha256", BRIDGE_SOURCE)
    proof = json.loads(BRIDGE.read_text())
    if (proof["order"] != ORDER or proof["held_labels_opened"] or not proof["targets_sealed"]
            or proof["source_sha256"] != BRIDGE_SOURCE or proof["status"] != "shared_fbl_sinc_targets_sealed"
            or not proof["exact_historical_zero"] or proof["recognition"]["bio_finite_scores"] != 2515
            or proof["fbl_bank"]["legacy_warm_parity"]
            or any(v["routed_sessions"] or v["sessions"] != 14 for v in proof["recognition"]["routing"].values())):
        raise ValueError("within0 fresh shared upstream set differs")
    pin(Path(proof["bridge"]["path"]), proof["bridge"]["sha256"])
    for bank in (proof["bank"], proof["fbl_bank"]):
        pin(Path(bank["path"]), bank["bank_sha256"])
    for info in proof["inputs"].values():
        pin(Path(info["path"]), info["sha256"])
    return dict(proof, legacy_fb_controls=LEGACY_RECEIPTS)
