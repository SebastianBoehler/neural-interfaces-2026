"""Replay the pinned held-new EA bridge and seal paired refit inputs per order."""
import argparse
import json
from pathlib import Path

import numpy as np

from bridge_aliases import compare
from bridge_mix import replay
from bridge_owner import BASE, HERE, ROOT, guarded_load, owner, pin, sha, stamp
from bridge_recognition import recognise
from bridge_refit import relabel
from member_inputs import ORDERS

EXTENSION = tuple(o for o in ORDERS if o != "full0")
CONTROL_MANIFEST = ROOT / "runs/coral304_inputs/control-original.sha256"
CONTROL_MANIFEST_SHA = "42782d6c2b7de4a8adddfcc5282eadbc0294dd84185a31a2706a90856eb35660"


def pinned_control(order):
    pin(CONTROL_MANIFEST, CONTROL_MANIFEST_SHA)
    rel = f"s281_v66p_control_{order}"
    raw, gate, provenance = (ROOT / path for path in (f"runs/chainout/{rel}.npz",
        f"runs/newmember/{rel}_gate.npz", f"runs/newmember/{rel}_provenance.json"))
    hashes = {ROOT / rel: digest for digest, rel in (line.split() for line in CONTROL_MANIFEST.read_text().splitlines())}
    return raw, gate, provenance, {path: hashes[path] for path in (raw, gate, provenance)}


def write_npz(path, bank):
    with path.open("xb") as handle:
        np.savez_compressed(handle, **bank)
    path.chmod(0o444)
    return dict(path=str(path), sha256=sha(path))


def write_receipt(path, **data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o444)


def legacy_comparison(order, bank, path):
    with np.load(path, allow_pickle=False) as archive:
        receipt = dict(sha256=sha(path), layout_matches=True)
        for key in ("pos", "tgrid"):
            if not np.array_equal(archive[key], bank[key]):
                receipt["layout_matches"] = False
                return receipt
        for key in ("all_loso", "relab"):
            a, b = archive[key], bank[f"historical|{key}"]
            if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
                receipt["layout_matches"] = False
                return receipt
            receipt[key] = compare(a, b)
    return receipt


def seal_prefix_inputs(order, work, labels):
    common = {key: labels[key] for key in ("index", "delivery", "pos", "tgrid")}
    out = {}
    for arm in ("historical", "corrected"):
        path = work / arm / f"chain_v51b_{order}.npz"
        path.parent.mkdir(exist_ok=False)
        bank = dict(common, **{key: labels[f"{arm}|{key}"] for key in ("all_loso", "relab")})
        out[arm] = write_npz(path, bank)
    return out


def run(order, work):
    raw, gate, source, pins = pinned_control(order)
    archived_route = ROOT / f"runs/s281_v66_route/{order}.json"
    legacy_relab = BASE / f"neural-interfaces-2026/runs/labels/relab_v51b_{order}.npz"
    route = json.loads(archived_route.read_text())
    if route["order"] != order or route["archive_sha256"] != pins[raw]:
        raise ValueError("historical recognition and incumbent shadow archives differ")
    receipt = dict(order=order, status="running", held_labels_opened=False, scored=False,
                   terminal_score_allowed=False, source=sha(HERE / "source.sha256"),
                   original_control={str(path): digest for path, digest in pins.items()},
                   original_route=sha(archived_route))
    try:
        with guarded_load(pins, source) as opened:
            ctx = owner(order)
            bridge, identity = replay(ctx, order, raw, gate)
            receipt.update(bridge=write_npz(work / "bridge.npz", bridge),
                           delivery_sha256=stamp(bridge["delivery"].astype("<i8")),
                           exact_historical_zero=True, **identity)
            outputs, routing = recognise(ctx, bridge, route, archived_route)
            if routing["bio_finite_scores"] != 2515:
                raise ValueError("expected true Bio activity for all delivered sessions")
            receipt.update(**routing, recognition=write_npz(work / "route.npz",
                           dict(index=bridge["index"], delivery=bridge["delivery"], **outputs)))
            if routing["routing"]["corrected"]["routed_sessions"]:
                receipt.update(status="corrected_route_requires_replay", matched_refits_allowed=False)
            else:
                labels, refits = relabel(ctx, bridge)
                if any((v["refits"], v["past_rows"], v["tsem_active"]) != (240, 24180, 240)
                       for v in refits["arms"].values()):
                    raise ValueError("the registered per-order prefix grid or TSEM activity differs")
                receipt.update(refits=refits, legacy_comparison=legacy_comparison(order, labels, legacy_relab),
                               inputs=seal_prefix_inputs(order, work, labels),
                               status="ready_for_matched_refits", matched_refits_allowed=True)
            receipt["opened"] = {str(path): digest for path, digest in sorted(opened.items())}
    except (ValueError, RuntimeError, OSError, AssertionError) as error:
        receipt.update(status="guard_failed", matched_refits_allowed=False, error=str(error))
        write_receipt(work / "status.json", **receipt)
        raise
    write_receipt(work / "status.json", **receipt)
    print(order, receipt["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order", choices=EXTENSION)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    args.work.mkdir(exist_ok=False)
    run(args.order, args.work)
