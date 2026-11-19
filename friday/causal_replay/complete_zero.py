"""Reprove the original live full0 raw, filtered and personalized stages without held gold."""
import json

import numpy as np

from bridge_mix import archived_zero, install, prove_activation
from bridge_owner import BASE, GATE, HERE, MEANS, ORDER, PROTO, RAW, ROOT, STACK, corrected, guarded_load, owner, pin, sha
from complete_mix import candidate
from complete_tail import live_tail, load_count, load_personalization

ARCHIVE = ROOT / "runs/diag/ttt/probs18"
STAGES = BASE / "ni26_match456_stages_20261118"
CONTROL = ROOT / "runs/s281_v66_cp/control_full0_B2_pool.npz"
CONTROL_ZERO = ROOT / "runs/s281_v66_cp/zero318_full0_B2_pool.npz"
UPSTREAM = MEANS / "bridge_full0_v1/bridge_full0.npz"
ROUTING = MEANS / "dependents_full0_v2/dependents_full0.json"
FEAT = ROOT / "data/proteus_windows.npz"
HASHES = {
    CONTROL: "087b14f54b65496de07e728549d71501d0f4fdfd711794e10b41bec5f044fa7f",
    CONTROL_ZERO: "d0a8fddef6e6f50c0b232c9760ba116e04a3a12ea60024964caafce07c83d4b0",
    STAGES / "replay.txt": "5faba4e1c72f60850a7e557fea817212b48db99497111cd263639d29bbed3bdf",
    STAGES / "stages_full0.npz": "09a21fb45aa7d7125a59137f792f55914ac65080af534f889360489a373aac91",
    STAGES / "full0.npz": "707d8a842fd4e924ffcb7f50ac5381e6a6c33abcd361abb30c89c185c8696f5a",
    UPSTREAM: "418a04b0648a48add856533ca59b23d26902d4df7dad3ad748bb92c051c15eaf",
    ROUTING: "bc8d1c4fa910747730f0e7710243263bf5c4478f468fd373035f36d4f61f20c5",
    FEAT: "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327",
}
FAMILIES = ("fbraw", "fb", "sinc", "fbl")


def read(path, key):
    with np.load(path, allow_pickle=False) as z:
        return z[key]


def no_route():
    r = json.loads(ROUTING.read_text())
    if (r["bridge"] != HASHES[UPSTREAM] or r["held_labels_opened"] or r["order"] != ORDER
            or r["bio_finite_scores"] != 2515 or
            any(v["routed_sessions"] or v["sessions"] != 14 for v in r["routing"].values())):
        raise ValueError("fixed full0 recognition proof no longer qualifies the unrouted path")


def set_aux_three(ctx):
    counts = (len(ctx.OCS), len(ctx.RSS))
    if counts != (9, 3):
        raise ValueError("full0 source eye/rest inventory differs from §430")
    eye = np.mean([ctx.ld(path) for path in ctx.OCS[:3]], 0)
    eye = np.log(np.clip(eye, 1e-3, 1))
    rest = np.mean([ctx.ld(path) for path in ctx.RSS[:3]], 0)
    if np.array_equal(eye, ctx.OCU) or not np.array_equal(rest, ctx.RESTR):
        raise ValueError("the three-auxiliary seed replay is inactive")
    ctx.OCU, ctx.RESTR = eye, rest


def verify_upstream(ctx, pairs):
    inverse = np.argsort(ctx.Pperm)
    with np.load(UPSTREAM, allow_pickle=False) as z:
        if not np.array_equal(ctx.Pperm, z["delivery"]):
            raise ValueError("upstream recognition is tied to a different order")
        for name, tape in pairs.items():
            for key in ("mix", "rebal", "gate", "em", "shadow"):
                if not np.array_equal(z[f"{name}|{key}"], tape[key][inverse]):
                    raise ValueError(f"freshly evaluated {name} {key} differs from its routing proof")


def main():
    for path, expected in HASHES.items():
        pin(path, expected)
    no_route()
    if (STAGES / "replay.txt").read_text().splitlines()[-1] != "OK":
        raise ValueError("recording-order replay reference is missing")
    final = load_personalization()
    paths = {(family, seed): ARCHIVE / f"ttt18full0LPC_{family}_r15c250-all_loso-s{seed}.npz"
             for family in FAMILIES for seed in range(6)}
    with guarded_load(archived_refits=paths.values()) as opened:
        ctx = owner()
        original = ctx.Z[f"{STACK}|Ps|{PROTO}"].copy()
        ctx.CU17 = {key: ctx.ld(path) for key, path in paths.items()}
        zero = install(ctx, {})
        archived_zero(ctx, zero, RAW, GATE)
        raw9, deltas9 = candidate(ctx, original, {}, zero)
        if not np.array_equal(raw9, read(RAW, f"{STACK}|V6Fr")):
            raise ValueError("archived nine-eye raw output differs")
        set_aux_three(ctx)
        raw3, deltas3 = candidate(ctx, original, {}, zero)
        if deltas9 != deltas3 or np.array_equal(raw9, raw3):
            raise ValueError("three-auxiliary control changed SessionBN refits or did not activate")
        patched = corrected(ctx.Pperm)
        causal = install(ctx, patched)
        activation = prove_activation(zero, causal, patched)
        verify_upstream(ctx, {"historical": zero, "corrected": causal})
        count = load_count(ctx)
        windows = read(FEAT, "X").astype(np.float64)
        terminal9, log9 = live_tail(count, final, raw9, windows, ctx.Pperm, verify_prefix=False)
        for path in (CONTROL, CONTROL_ZERO):
            if not np.array_equal(terminal9["terminal"], read(path, "scores")):
                raise ValueError("the historical personalized final control differs")
        print("PASS original §281 nine-eye archived final 2655 exact", flush=True)
        terminal3, log3 = live_tail(count, final, raw3, windows, ctx.Pperm)
        for key, value in terminal3.items():
            if not np.array_equal(value, read(STAGES / "stages_full0.npz", key)):
                raise ValueError(f"§430 exact stage comparison failed at {key}")
        if not np.array_equal(terminal3["terminal"], read(STAGES / "full0.npz", "scores")):
            raise ValueError("§430 archived three-eye final differs")
    proof = dict(order=ORDER, exact_archive_zero=True, three_aux_exact=True, held_labels_opened=False,
                 refit_matrix_opened=False, scored=False, stages=log3, original_stages=log9,
                 corrected_slots=activation, refit_seed0_deltas=deltas3,
                 source_sha256=sha(HERE / "source.sha256"),
                 archived_inputs={str(p): h for p, h in sorted(opened.items())})
    with (HERE / "zero.json").open("x") as fd:
        json.dump(proof, fd, indent=2, sort_keys=True)
        fd.write("\n")
    print("PASS full0 historical terminal and live corrected incumbent bridge", flush=True)


if __name__ == "__main__":
    main()
