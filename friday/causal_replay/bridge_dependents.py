"""Seal verified full0 live recognition and strict-prefix refit state."""
import json

import numpy as np

from bridge_owner import HERE, MEANS, ORDER, PINS, pin, guarded_load, owner, sha
from bridge_recognition import REFERENCE, recognise
from bridge_refit import relabel

BRIDGE_DIR = MEANS / "bridge_full0_v1"
BRIDGE = BRIDGE_DIR / "bridge_full0.npz"
BRIDGE_SHA = "418a04b0648a48add856533ca59b23d26902d4df7dad3ad748bb92c051c15eaf"
BRIDGE_RECEIPT_SHA = "c61b3c47ac051452eed49a2afdf5ae994fc26432366ac77d68c421879b7f7f89"
ARCHIVED_ROUTE_SHA = "befe52d9c2491c15fbc613eb62ed45186969de69ee5b1cffe63906d834a600d7"


def upload_free_write(path, value):
    with path.open("xb") as handle:
        np.savez_compressed(handle, **value)
    return sha(path)


def main():
    pin(BRIDGE, BRIDGE_SHA)
    bridge_receipt = BRIDGE.with_suffix(".json")
    pin(bridge_receipt, BRIDGE_RECEIPT_SHA)
    receipt = json.loads(bridge_receipt.read_text())
    if receipt["held_labels_opened"] or not receipt["zero"] or receipt["output_sha256"] != BRIDGE_SHA:
        raise ValueError("upstream bridge proof is invalid")
    pin(REFERENCE, ARCHIVED_ROUTE_SHA)
    archive = json.loads(REFERENCE.read_text())
    if archive["order"] != ORDER or archive["archive_sha256"] not in PINS.values():
        raise ValueError("archived route does not authenticate the zero arm")
    routed_path = HERE / "route_full0.npz"
    refit_path = HERE / "relab_corrected_v51b_full0.npz"
    with guarded_load() as opened:
        ctx = owner()
        with np.load(BRIDGE, allow_pickle=False) as sealed:
            bridge = dict(sealed)
        routes, trace = recognise(ctx, bridge, archive)
        route_sha = upload_free_write(routed_path, dict(index=bridge["index"], delivery=bridge["delivery"], **routes))
        manifest = dict(**trace, order=ORDER, bridge=BRIDGE_SHA, bridge_receipt=BRIDGE_RECEIPT_SHA,
                        recognition_output=route_sha, historical_route=ARCHIVED_ROUTE_SHA, held_labels_opened=False,
                        scored=False, code=sha(HERE / "source.sha256"))
        if trace["routing"]["corrected"]["routed_sessions"] == 0:
            bank, stats = relabel(ctx, bridge)
            bank["relab"] = bank.pop("corrected|relab")
            bank["all_loso"] = bank.pop("corrected|all_loso")
            manifest.update(refit_output=upload_free_write(refit_path, bank), refits=stats)
        manifest["opened"] = {str(path): digest for path, digest in sorted(opened.items())}
    with (HERE / "dependents_full0.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if trace["routing"]["corrected"]["routed_sessions"]:
        raise ValueError("corrected arm requires a live routed-tail implementation before relabelling")
    print(f"PASS live recognition and strict refit prefixes {manifest['refit_output']}", flush=True)


if __name__ == "__main__":
    main()
