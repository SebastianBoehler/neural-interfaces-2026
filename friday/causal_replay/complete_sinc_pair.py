"""Seal the two complete full0 replays on one shared fresh Sinc frozen input."""
import argparse
import json
from pathlib import Path

import numpy as np

from bridge_mix import install
from bridge_orders import write_npz, write_receipt
from bridge_owner import HERE, MEANS, ORDER, PROTO, STACK, corrected, guarded_load, owner, pin, sha, stamp
from complete_mix import candidate
from complete_tail import live_tail, load_count, load_personalization
from complete_zero import FEAT, FAMILIES, HASHES, read, set_aux_three
from sinc_seed0_bridge import SEED0

ZERO = MEANS / "terminal_full0_v2/zero.json"
PREFIX = MEANS / "terminal_full0_v3/active_prefix.json"
ZERO_SHA = "9b9dc4070b234a6df2c2d806c7ad5f2ddf2dbd231370fae688369a78b046663c"
PREFIX_SHA = "d9b53c4f4289de0f784c6754d21b9018e805045668ad9a8e52fbc142d85fb490"
FIT_TOL = 2e-5
REFIT_ACTOR_SHA = "a3d227069dd9989bf77b1c1feddb3ce221e75687d354ef17caab0df12b335ca6"


def authenticate_upstream(path, digest):
    pin(ZERO, ZERO_SHA)
    pin(PREFIX, PREFIX_SHA)
    pin(path, digest)
    evidence = json.loads(path.read_text())
    routes = evidence["recognition"]
    if (evidence["order"] != ORDER or evidence["status"] != "ready_for_matched_refits"
            or evidence["held_labels_opened"] or evidence["scored"] or not evidence["historical_zero"]
            or evidence["terminal_zero_sha256"] != ZERO_SHA or routes["bio_finite_scores"] != 2515
            or any(v["routed_sessions"] or v["sessions"] != 14 for v in routes["routing"].values())):
        raise ValueError("fresh matched incumbent bridge or unrouted recognition proof differs")
    bridge = evidence["bridge"]
    pin(Path(bridge["path"]), bridge["sha256"])
    if set(evidence["inputs"]) != {"historical", "corrected"}:
        raise ValueError("both regenerated fitted targets are required")
    for inputs in evidence["inputs"].values():
        pin(Path(inputs["path"]), inputs["sha256"])
    return evidence


def restore_sinc(ctx, path, bank):
    pin(path, bank["bank_sha256"])
    with np.load(path, allow_pickle=False) as z:
        if (not np.array_equal(z["index"], np.arange(len(ctx.Pperm)))
                or not np.array_equal(z["delivery"], ctx.Pperm)):
            raise ValueError("fresh Sinc frozen bank does not match full0")
        ctx.FR17[SEED0] = z["full0"]


def pair_refits(root, source_sha, ctx, evidence, status_sha, initial):
    pin(root / "source.sha256", source_sha)
    target_pins = {"historical": evidence["inputs"]["historical"]["sha256"],
                   "causal": evidence["inputs"]["corrected"]["sha256"]}
    with np.load(FEAT, allow_pickle=False) as z:
        subjects = z["subject"]
    fit_hashes = {name.rsplit("/", 1)[1]: stamp(np.flatnonzero(subjects != name).astype("<i8"))
                  for name in np.unique(subjects)}
    proofs, banks = {}, {arm: {} for arm in ("base", "historical", "causal")}
    for family in FAMILIES:
        key = family, 0
        path = root / "members" / f"{family}_s0.npz"
        receipt = path.with_suffix(".json")
        meta = json.loads(receipt.read_text())
        expected = dict(member=family, seed=0, rows=len(ctx.Pperm), order=ORDER, smoke=False,
                        held_labels_opened=False, actor_sha256=REFIT_ACTOR_SHA, targets=target_pins,
                        bridge_sha256=evidence["bridge"]["sha256"], bridge_status_sha256=status_sha)
        got_fit = {name: item["fit_rows_sha"] for name, item in meta["fit_states"].items()}
        if any(meta[k] != value for k, value in expected.items()) or got_fit != fit_hashes:
            raise ValueError(f"{family}: the fitted input is not tied to the shared fresh Sinc bridge")
        pin(path, meta["output_sha256"])
        with np.load(path, allow_pickle=False) as z:
            if (not np.array_equal(z["index"], np.arange(len(ctx.Pperm)))
                    or not np.array_equal(z["delivery"], ctx.Pperm)):
                raise ValueError(f"{family}: refit delivery or canonical rows differ")
            values = {arm: z[arm] for arm in banks}
        for name, a in values.items():
            if (a.shape != (len(ctx.Pperm), 3) or not np.isfinite(a).all() or (a < 0).any()
                    or not np.allclose(a.sum(1), 1, rtol=0, atol=1e-10)):
                raise ValueError(f"{family}: refit output {name} is invalid")
        base = values["base"]
        anchor = ctx.FR17[key]
        frozen_delta = float(np.abs(base - anchor).max())
        if frozen_delta > FIT_TOL or not np.array_equal(base.argmax(1), anchor.argmax(1)):
            raise ValueError(f"{family}: fitted base differs from its anchored frozen member")
        if any(not np.array_equal(values[a][initial], base[initial]) for a in ("historical", "causal")):
            raise ValueError(f"{family}: refits altered a strict warmup row")
        if float(np.abs(values["historical"] - values["causal"]).max()) <= 1e-4:
            raise ValueError(f"{family}: matched targets did not activate the refits")
        for arm, output in banks.items():
            output[key] = values[arm]
        proofs[family] = dict(artifact=meta["output_sha256"], receipt=sha(receipt), base_max_abs=frozen_delta)
    return banks, proofs


def verify_incumbent(ctx, bridge, arm, tape):
    inverse = np.argsort(ctx.Pperm)
    for key in ("mix", "rebal", "gate", "em", "shadow"):
        if not np.array_equal(tape[key][inverse], bridge[f"{arm}|{key}"]):
            raise ValueError(f"live {arm} {key} differs from its sealed recognition input")


def run(upstream, upstream_sha, frozen, matrix, matrix_sha):
    evidence = authenticate_upstream(upstream, upstream_sha)
    for path, expected in HASHES.items():
        pin(path, expected)
    final = load_personalization()
    with guarded_load() as opened:
        ctx = owner()
        p = ctx.Pperm
        original = ctx.Z[f"{STACK}|Ps|{PROTO}"].copy()
        restore_sinc(ctx, frozen, evidence["bank"])
        with np.load(evidence["bridge"]["path"], allow_pickle=False) as z:
            bridge = dict(z)
        if not np.array_equal(bridge["index"], np.arange(len(p))) or not np.array_equal(bridge["delivery"], p):
            raise ValueError("seal held a different pinned order")
        sessions = bridge["session"]
        initial = np.concatenate([p[sessions[p] == s][:20] for s in dict.fromkeys(sessions[p])])
        if len(initial) != 280:
            raise ValueError("missing first twenty rows in a real delivered session")
        banks, proofs = pair_refits(matrix, matrix_sha, ctx, evidence, upstream_sha, initial)
        ctx.FT_BASE = banks["base"]
        set_aux_three(ctx)
        arms = {"historical": {}, "corrected": corrected(p)}
        incumbents, candidates, active = {}, {}, {}
        for arm, slots in arms.items():
            incumbent = install(ctx, slots)
            verify_incumbent(ctx, bridge, arm, incumbent)
            incumbents[arm] = incumbent["shadow"][np.argsort(p)]
            ctx.CU17 = banks["causal" if arm == "corrected" else arm]
            candidates[arm], active[arm] = candidate(ctx, original, slots, incumbent)
        if any(np.array_equal(a["historical"], a["corrected"]) for a in (incumbents, candidates)):
            raise ValueError("corrected EA did not reach both live systems")
        count = load_count(ctx)
        windows = read(FEAT, "X").astype(np.float64)
        result, checks = dict(index=np.arange(len(p)), delivery=p), {}
        for arm in arms:
            terminal, checks[arm] = live_tail(count, final, candidates[arm], windows, p)
            result[f"{arm}|P0r"] = incumbents[arm].argmax(1)
            result[f"{arm}|P0r_posterior"] = incumbents[arm]
            result[f"{arm}|cap250"] = terminal["terminal"].argmax(1)
            result.update({f"{arm}|cap250_{key}": array for key, array in terminal.items()})
        if np.array_equal(result["historical|cap250_terminal"], result["corrected|cap250_terminal"]):
            raise ValueError("corrected EA did not reach the terminal cap250 output")
    target = write_npz(HERE / "matched_sinc_full0.npz", result)
    write_receipt(HERE / "matched_sinc_full0.json", order=ORDER, complete=True, scored=False,
                  held_labels_opened=False, exact_zero=ZERO_SHA, active_archive_prefix=PREFIX_SHA,
                  bridge_receipt_sha256=upstream_sha, fresh_sinc_seed=0, fresh_sinc_bank=evidence["bank"]["bank_sha256"],
                  source_sha256=sha(HERE / "source.sha256"), refit_source_sha256=matrix_sha,
                  refits=proofs, active_seed0_deltas=active, terminal=checks, output=target,
                  corrected_shadows_exact=True, opened={str(path): digest for path, digest in sorted(opened.items())})
    print("PASS both matched full0 systems sealed before gold", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_status", type=Path)
    parser.add_argument("upstream_status_sha")
    parser.add_argument("frozen_sinc", type=Path)
    parser.add_argument("refit_matrix", type=Path)
    parser.add_argument("refit_source_sha")
    args = parser.parse_args()
    run(args.upstream_status, args.upstream_status_sha, args.frozen_sinc, args.refit_matrix, args.refit_source_sha)
