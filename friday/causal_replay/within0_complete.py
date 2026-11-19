"""Qualify the archived within0 zero and seal both matched complete systems."""

import argparse
import json
from pathlib import Path

import numpy as np

from bridge_mix import archived_zero, install
from bridge_orders import pinned_control, write_npz, write_receipt
from bridge_owner import HERE, PROTO, ROOT, STACK, corrected, guarded_load, owner, pin, sha
from complete_mix import candidate
from complete_score import KEYS, score_columns
from complete_sinc_pair import pair_refits, verify_incumbent
from complete_tail import live_tail, load_count, load_personalization
from complete_zero import ARCHIVE, FEAT, STAGES, read
from member_inputs import ORDERS, pinned_delivery
from within0_pair_inputs import BRIDGE, CAMP, FAMILIES, ORDER, authenticate


CONTROL = ROOT / "runs/s281_v66_cp/control_within0_B2_pool.npz"
CONTROL_ZERO = ROOT / "runs/s281_v66_cp/zero318_within0_B2_pool.npz"
REPLAY = STAGES / "replay.txt"
STOCK = STAGES / "within0.npz"
TAPES = STAGES / "stages_within0.npz"
TAPE_PROOF = STAGES / "stages_within0.json"
ROUTE = ROOT / "runs/s281_v66_route/within0.json"
ZERO = CAMP / "within0_terminal_v1/zero.json"
ZERO_SOURCE = "510d4a029b8d87780a32a487e8cf6708b7558b41dc727d028a283f717e478207"
PINS = {
    CONTROL: "65f0c1c60205bbea67df37701754fa6e3691e0997509dc063a79d19847163556",
    CONTROL_ZERO: "da072fae3f18a3ef104396813fc463a72e9945ed52675e5ef2f9af5847dda931",
    REPLAY: "5faba4e1c72f60850a7e557fea817212b48db99497111cd263639d29bbed3bdf",
    STOCK: "89535bc8786a6b77dfaa32d8071bd31d84f08db75bb2ac695a37866d3d9c6c22",
    TAPES: "a8cc81b662c2e0f2a8bd594a909cb18ec8588a71274057753834564bc723f153",
    TAPE_PROOF: "0d65e20fc8d2df8957425bd6b7373b3638b984c37f3ca83a70c998b0eca286fd",
    ROUTE: "3a16eb9ce67d8451164f7038946d9753f317f49db8e96a686d5f16d36603925b",
    FEAT: "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327",
}


def input_delivery(order=ORDER):
    return pinned_delivery(ROOT / "runs/shuffle_perms.npz")[ORDERS.index(order)]


def three_aux(ctx):
    for attr, files, prefix in (("OCU", ctx.OCS, "oc_nomix"), ("RESTR", ctx.RSS, "rest4_sh_r1")):
        if [Path(path).name for path in files] != [f"{prefix}-all_loso-s{s}.npz" for s in range(9)]:
            raise ValueError(f"unexpected within0 §281 {attr} inventory")
        mean = np.mean([ctx.ld(path) for path in files[:3]], 0)
        value = np.log(np.clip(mean, 1e-3, 1)) if attr == "OCU" else mean
        if np.array_equal(getattr(ctx, attr), value):
            raise ValueError(f"the registered three-seed {attr} switch is inactive")
        setattr(ctx, attr, value)


def archive_zero(order=ORDER, pins=PINS):
    for path, digest in pins.items():
        pin(path, digest)
    if REPLAY.read_text().splitlines()[-1] != "OK":
        raise ValueError("recording-order replay reference is missing")
    raw, gate, provenance, expected = pinned_control(order)
    route = json.loads((ROOT / f"runs/s281_v66_route/{order}.json").read_text())
    if route["order"] != order or route["archive_sha256"] != expected[raw] or route["bases"][STACK]["routed_sessions"]:
        raise ValueError("archived recognition differs")
    final = load_personalization()
    refs = {(f, s): ARCHIVE / f"ttt18{order}LPC_{f}_r15c250-all_loso-s{s}.npz"
            for f in FAMILIES for s in range(6)}
    with guarded_load(expected, provenance, archived_refits=refs.values()) as opened:
        ctx = owner(order)
        p = ctx.Pperm
        np.testing.assert_array_equal(p, input_delivery(order))
        original = ctx.Z[f"{STACK}|Ps|{PROTO}"].copy()
        ctx.CU17 = {key: ctx.ld(path) for key, path in refs.items()}
        incoming = install(ctx, {})
        archived_zero(ctx, incoming, raw, gate)
        original_cap, a9 = candidate(ctx, original, {}, incoming)
        np.testing.assert_array_equal(original_cap, read(raw, f"{STACK}|V6Fr"))
        three_aux(ctx)
        auxiliary_cap, a3 = candidate(ctx, original, {}, incoming)
        if a3 != a9 or np.array_equal(original_cap, auxiliary_cap):
            raise ValueError("auxiliary switch is inactive or changed archived refit deltas")
        count = load_count(ctx)
        x = read(FEAT, "X").astype("float64")
        nine, _ = live_tail(count, final, original_cap, x, p, verify_prefix=False, order=order)
        for arm in ("control", "zero318"):
            path = ROOT / f"runs/s281_v66_cp/{arm}_{order}_B2_pool.npz"
            np.testing.assert_array_equal(nine["terminal"], read(path, "scores"))
        three, checks = live_tail(count, final, auxiliary_cap, x, p, order=order)
        for key, val in three.items():
            np.testing.assert_array_equal(val, read(STAGES / f"stages_{order}.npz", key))
        np.testing.assert_array_equal(three["terminal"], read(STAGES / f"{order}.npz", "scores"))
    write_receipt(HERE / "zero.json", order=order, exact_archive_zero=True, three_aux_exact=True,
                  held_labels_opened=False, scored=False, prefix=checks, deltas=a3,
                  source=sha(HERE / "source.sha256"), inputs={str(path): v for path, v in sorted(opened.items())})
    print(f"PASS {order} archived zero", sha(HERE / "zero.json"), flush=True)


def qualified_zero(zero_sha):
    pin(ZERO, zero_sha)
    pin(ZERO.parent / "source.sha256", ZERO_SOURCE)
    zero = json.loads(ZERO.read_text())
    if (zero["source"] != ZERO_SOURCE or zero["order"] != ORDER or not zero["exact_archive_zero"]
            or not zero["three_aux_exact"] or zero["held_labels_opened"] or zero["scored"]
            or zero["prefix"]["exact_interleaved_prefix_checks"] != 3
            or min(zero["prefix"]["count_affected_prefix_rows"]) <= 0):
        raise ValueError("within0 terminal zero proof differs")


def predict(zero_sha, matrix, matrix_sha, bridge_sha):
    qualified_zero(zero_sha)
    evidence = authenticate(pin, bridge_sha)
    raw, gate, provenance, expected = pinned_control(ORDER)
    for path, digest in PINS.items():
        pin(path, digest)
    final = load_personalization()
    with guarded_load(expected, provenance) as opened:
        ctx = owner(ORDER)
        p = ctx.Pperm
        np.testing.assert_array_equal(p, input_delivery())
        original = ctx.Z[f"{STACK}|Ps|{PROTO}"].copy()
        archived_zero(ctx, install(ctx, {}), raw, gate)
        bank = evidence["bank"]
        path = Path(bank["path"])
        pin(path, bank["bank_sha256"])
        with np.load(path, allow_pickle=False) as z:
            j = ORDERS.index(ORDER)
            np.testing.assert_array_equal(z["delivery"][j], p)
            np.testing.assert_array_equal(z["index"], np.arange(len(p)))
            ctx.FR17["sinc", 0] = z["proba"][j]
        with np.load(evidence["fbl_bank"]["path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(z["delivery"], p)
            np.testing.assert_array_equal(z["index"], np.arange(len(p)))
            ctx.FR17["fbl", 0] = z["proba"]
        with np.load(evidence["bridge"]["path"], allow_pickle=False) as z:
            bridge = dict(z)
        np.testing.assert_array_equal(bridge["index"], np.arange(len(p)))
        np.testing.assert_array_equal(bridge["delivery"], p)
        sessions = bridge["session"]
        initial = np.concatenate([p[sessions[p] == s][:20] for s in dict.fromkeys(sessions[p])])
        if len(initial) != 280:
            raise ValueError("invalid matched warmup coverage")
        banks, proofs = pair_refits(matrix, matrix_sha, ctx, evidence, bridge_sha, initial, ORDER)
        for family in FAMILIES:
            m = json.loads((matrix / "members" / f"{family}_s0.json").read_text())
            smoke = matrix / "smoke" / f"{family}_s0.json"
            if (m["legacy_fb_controls"] != evidence["legacy_fb_controls"] or m["source"] != matrix_sha
                    or m["smoke_sha256"] != sha(smoke) or len(m["visits"]) != 14):
                raise ValueError(f"unverified warm actor smoke gate for {family}")
            if family == "fbl" and any(m["fit_states"][person]["sha256"] != item["checkpoint_sha256"]
                                       for person, item in evidence["fbl_bank"]["folds"].items()):
                raise ValueError("the matched FBLight banks differ from the true shared source")
        ctx.FT_BASE = banks["base"]
        three_aux(ctx)
        incumbents, mixes, activity = {}, {}, {}
        for arm, patches in (("historical", {}), ("corrected", corrected(p, ORDER))):
            tape = install(ctx, patches)
            verify_incumbent(ctx, bridge, arm, tape)
            incumbents[arm] = tape["shadow"][np.argsort(p)]
            ctx.CU17 = banks["historical" if arm == "historical" else "causal"]
            mixes[arm], activity[arm] = candidate(ctx, original, patches, tape)
        if any(np.array_equal(x["historical"], x["corrected"]) for x in (incumbents, mixes)):
            raise ValueError("corrected EA failed to reach both live systems")
        count = load_count(ctx)
        x = read(FEAT, "X").astype("float64")
        result, proofs_tail = dict(index=np.arange(len(p)), delivery=p), {}
        for arm in incumbents:
            tape, proofs_tail[arm] = live_tail(count, final, mixes[arm], x, p, order=ORDER)
            result[f"{arm}|P0r"] = incumbents[arm].argmax(1)
            result[f"{arm}|P0r_posterior"] = incumbents[arm]
            result[f"{arm}|cap250"] = tape["terminal"].argmax(1)
            result.update({f"{arm}|cap250_{key}": val for key, val in tape.items()})
        if np.array_equal(result["historical|cap250_terminal"], result["corrected|cap250_terminal"]):
            raise ValueError("terminal stage had no matched causal effect")
    bank = write_npz(HERE / "matched_within0.npz", result)
    write_receipt(HERE / "matched_within0.json", order=ORDER, complete=True, held_labels_opened=False, scored=False,
                  output=bank, matches_upstream=True, bridge=bridge_sha, legacy_fb_controls=evidence["legacy_fb_controls"], archive_zero=zero_sha,
                  matrix_source=matrix_sha, source=sha(HERE / "source.sha256"), refits=proofs,
                  terminal=proofs_tail, active_seed0_deltas=activity)


def score():
    path = HERE / "matched_within0.json"
    receipt = json.loads(path.read_text())
    if (not receipt["complete"] or receipt["order"] != ORDER or receipt["held_labels_opened"] or receipt["scored"]
            or receipt["bridge"] != sha(BRIDGE) or receipt["source"] != sha(HERE / "source.sha256")
            or not receipt["matches_upstream"] or set(receipt["refits"]) != set(FAMILIES)
            or set(receipt["terminal"]) != {"historical", "corrected"}
            or any(t["exact_interleaved_prefix_checks"] != 3 or min(t["count_affected_prefix_rows"]) <= 0
                   for t in receipt["terminal"].values())):
        raise ValueError("unqualified sealed decisions")
    qualified_zero(receipt["archive_zero"])
    evidence = authenticate(pin, receipt["bridge"])
    if receipt["legacy_fb_controls"] != evidence["legacy_fb_controls"]:
        raise ValueError("sealed legacy FB control proofs differ")
    bank = HERE / "matched_within0.npz"
    if bank != Path(receipt["output"]["path"]) or any(p.stat().st_mode & 0o222 for p in (bank, path)):
        raise ValueError("unsealed prediction bank")
    pin(bank, receipt["output"]["sha256"])
    with np.load(bank, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["index"], np.arange(2655))
        np.testing.assert_array_equal(z["delivery"], input_delivery())
        columns = np.column_stack([z[k] for k in KEYS])
        if columns.shape != (2655, 4) or not np.isin(columns, (0, 1, 2)).all():
            raise ValueError("invalid sealed held-new decisions")
        for j, key in enumerate(KEYS):
            suffix = "_terminal" if key.endswith("cap250") else "_posterior"
            np.testing.assert_array_equal(z[key + suffix].argmax(1), columns[:, j])
    score_columns(columns, sha(path), HERE / "matched_within0_score.json",
                  "FULL-SYSTEM DEV, held-new within0, shared fresh Sinc and FBLight seed 0", order=ORDER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("zero")
    modes.add_parser("score")
    live = modes.add_parser("predict")
    live.add_argument("matrix", type=Path)
    live.add_argument("matrix_sha")
    args = parser.parse_args()
    if args.mode == "predict":
        predict(sha(ZERO), args.matrix, args.matrix_sha, sha(BRIDGE))
    elif args.mode == "score":
        score()
    else:
        archive_zero()
