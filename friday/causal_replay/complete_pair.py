"""Seal the two full0 systems only after authenticating every matched member."""
import argparse
import json
from pathlib import Path

import numpy as np

from bridge_mix import archived_zero, install
from bridge_owner import GATE, HERE, ORDER, PROTO, RAW, STACK, corrected, guarded_load, owner, pin, sha
from complete_mix import candidate
from complete_tail import live_tail, load_count, load_personalization
from complete_zero import FEAT, FAMILIES, HASHES, no_route, read, set_aux_three, verify_upstream

TARGETS = {
    "historical": "5290dd47d3d19523bfc1308db5882fe116780ff102fd757deaa78d2e051399cf",
    "causal": "ca9aef1a8781316e64cf875ccd9c1a7fe722cecb9e71de3a6d753d07312f5cc6",
}
FIT_TOL = 2e-5


def matrix(root, ctx, source_sha):
    pin(root / "source.sha256", source_sha)
    with np.load(FEAT, allow_pickle=False) as z:
        subject, visit = z["subject"], z["session"]
    sessions = np.char.add(np.char.add(subject, "|"), visit)
    p = ctx.Pperm
    prefix = np.concatenate([p[sessions[p] == s][:20] for s in dict.fromkeys(sessions[p])])
    if len(prefix) != 280:
        raise ValueError("unexpected session warmup count")
    proofs, banks = {}, {arm: {} for arm in ("base", *TARGETS)}
    for name in FAMILIES:
        for seed in range(6):
            key = name, seed
            path = root / "members" / f"{name}_s{seed}.npz"
            receipt = path.with_suffix(".json")
            m = json.loads(receipt.read_text())
            expected = dict(member=name, seed=seed, smoke=False, order=ORDER,
                            rows=len(p), targets=TARGETS, held_labels_opened=False)
            if any(m[k] != v for k, v in expected.items()) or len(m["fit_states"]) != 10:
                raise ValueError(f"missing matched refit proof {key}")
            if not 0 <= m["archived_frozen_max_abs"] <= FIT_TOL:
                raise ValueError(f"regenerated member does not qualify against the anchored frozen input {key}")
            pin(path, m["output_sha256"])
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z["index"], np.arange(len(p))) or not np.array_equal(z["delivery"], p):
                    raise ValueError(f"member order or rows differ {key}")
                for arm, out in banks.items():
                    values = z[arm]
                    if (values.shape != (len(p), 3) or not np.isfinite(values).all() or (values < 0).any()
                            or not np.allclose(values.sum(1), 1, rtol=0, atol=1e-10)):
                        raise ValueError(f"invalid member probabilities {key} {arm}")
                    out[key] = values
            frozen = banks["base"][key]
            archived = ctx.FR17[key]
            if (not np.array_equal(frozen.argmax(1), archived.argmax(1))
                    or float(np.abs(frozen - archived).max()) > FIT_TOL):
                raise ValueError(f"common regenerated base is too far from the live frozen input {key}")
            causal, historical = (banks[arm][key] for arm in ("causal", "historical"))
            if (not np.array_equal(causal[prefix], frozen[prefix])
                    or not np.array_equal(historical[prefix], frozen[prefix])
                    or float(np.abs(causal - historical).max()) <= 1e-4):
                raise ValueError(f"paired refit activation or warmup differs {key}")
            proofs[f"{name}_s{seed}"] = dict(npz=m["output_sha256"], receipt=sha(receipt), actor=m["actor_sha256"])
    return banks, proofs


def seal(zero_path, zero_sha, matrix_dir, matrix_sha):
    pin(zero_path, zero_sha)
    zero = json.loads(zero_path.read_text())
    if not zero["exact_archive_zero"] or not zero["three_aux_exact"] or zero["held_labels_opened"]:
        raise ValueError("terminal archive proof is missing")
    for path, expected in HASHES.items():
        pin(path, expected)
    no_route()
    final = load_personalization()
    with guarded_load() as opened:
        ctx = owner()
        p = ctx.Pperm
        original = ctx.Z[f"{STACK}|Ps|{PROTO}"].copy()
        banks, proofs = matrix(matrix_dir, ctx, matrix_sha)
        ctx.FT_BASE = banks["base"]
        set_aux_three(ctx)
        patches = {"historical": {}, "corrected": corrected(p)}
        incumbents, mixes, ft = {}, {}, {}
        for arm, slots in patches.items():
            incumbents[arm] = install(ctx, slots)
            if arm == "historical":
                archived_zero(ctx, incumbents[arm], RAW, GATE)
            ctx.CU17 = banks["causal" if arm == "corrected" else arm]
            mixes[arm], ft[arm] = candidate(ctx, original, slots, incumbents[arm])
        verify_upstream(ctx, incumbents)
        if np.array_equal(mixes["corrected"], mixes["historical"]):
            raise ValueError("matched causal changes did not propagate to the candidate shadow")
        count = load_count(ctx)
        windows = read(FEAT, "X").astype(np.float64)
        result = dict(index=np.arange(len(p)), delivery=p)
        checks = {}
        for arm, tape in incumbents.items():
            stage, checks[arm] = live_tail(count, final, mixes[arm], windows, p)
            incumbent = tape["shadow"][np.argsort(p)]
            result[f"{arm}|P0r"] = incumbent.argmax(1)
            result[f"{arm}|cap250"] = stage["terminal"].argmax(1)
            result[f"{arm}|P0r_posterior"] = incumbent
            for key, value in stage.items():
                result[f"{arm}|cap250_{key}"] = value
    target = HERE / "matched_full0.npz"
    with target.open("xb") as handle:
        np.savez_compressed(handle, **result)
    target.chmod(0o444)
    receipt = HERE / "matched_full0.json"
    with receipt.open("x") as handle:
        json.dump(dict(order=ORDER, exact_archive_zero=zero_sha, output_sha256=sha(target),
                       held_labels_opened=False, source_sha256=sha(HERE / "source.sha256"),
                       matrix_source_sha256=matrix_sha, matrix=proofs, terminal=checks, slot_refit_deltas=ft,
                       opened={str(path): digest for path, digest in sorted(opened.items())},
                       complete=True), handle, sort_keys=True, indent=2)
        handle.write("\n")
    receipt.chmod(0o444)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zero", type=Path)
    parser.add_argument("zero_sha")
    parser.add_argument("matrix", type=Path)
    parser.add_argument("matrix_sha")
    args = parser.parse_args()
    seal(args.zero, args.zero_sha, args.matrix, args.matrix_sha)
