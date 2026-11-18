"""Match the §257 within0 warm refits on their own archived targets."""

import argparse
import json
import os
import runpy

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from fresh_sinc import evaluate, state_hash
from member_inputs import ORDERS, pinned_delivery
from paired_refit_smoke import target_table
from refit_smoke import ARCHIVE, HERE, ROOT, TOL, changed, file_sha, waveform_data
from restore_refit import RELEASE, RELEASE_SHA, restore
from within0_frozen_probe import PEOPLE, rows_sha


ORDER = "within0"
ACTORS = {0: "fbraw", 1: "fbl", 3: "fb"}
FROZEN = HERE.parent / "within0_frozen_probe_v4/qualification.json"
FROZEN_SHA = "43fe6dc888a0cef5fd423976577785b671474be2cfa13668059bbb53839a44eb"
FTREL = ROOT / "studies/14nov/ftrelabel/ftrel.py"
CHAIN = ROOT / "runs/labels/chain_v51b_within0.npz"
RELAB = ROOT / "runs/labels/relab_v51b_within0.npz"
REFIT_NAME = "r15c250"
PREFIX = 80


def actor():
    gpu = torch.cuda.get_device_name()
    if "RTX 2080 Ti" not in gpu or torch.__version__ != "2.14.1+cu126":
        raise ValueError("requires the original §257 RTX inference geometry")
    if any(os.environ[key] != "8" for key in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS",
                                             "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("requires the original eight-thread geometry")
    torch.set_num_threads(8)
    os.environ.update(NI26_WS=str(ROOT), SCHEME="within", PSEED="0", KIND="v51b", ARMS="s15,r15",
                      CAPS="100,150,250", DRAW="0", CHAIN_NPZ=str(CHAIN), RELAB_NPZ=str(RELAB))
    scope = runpy.run_path(str(FTREL))
    api, owner, stream = (scope[k] for k in ("T5", "T7", "stream"))
    names = {"s15", "r15", "r15c100", "r15c150", REFIT_NAME}
    if ((scope["DRAW"], scope["ARMS"], scope["CAPS"], scope["CHECK"]) != (0, ["s15", "r15"], [100, 150, 250], False)
            or owner.TAG != "ttt18within0LPC" or owner.run.__globals__["stream"] is not stream
            or set(owner.VARS) != names or api.ORDER != ["fbraw", "fbl", "sinc", "fb"]
            or tuple(owner.VARS[REFIT_NAME]) != (5, 10, 1e-3, "all", 1., "hard", True, 1, True, 20)):
        raise ValueError("the loaded actor recipe differs from the §257 invocation")
    return api, stream, api.chain_labels("all_loso")


def reference(family):
    out = {}
    for key in ("base", REFIT_NAME):
        path = ARCHIVE / f"ttt18within0LPC_{family}_{key}-all_loso-s0.npz"
        with np.load(path, allow_pickle=False) as old:
            np.testing.assert_array_equal(old["index"], np.arange(2655))
            out[key] = old["proba"]
        if out[key].shape != (2655, 3) or not np.isfinite(out[key]).all():
            raise ValueError(f"invalid archive for {family}/{key}")
    return out


def run_stream(stream, model, x, target):
    actual = 0

    def counted(opt, _args, _kwargs):
        nonlocal actual
        if not isinstance(opt, torch.optim.AdamW) or any(
                g["lr"] != 1e-3 or g["weight_decay"] != 0 for g in opt.param_groups):
            raise ValueError("historical optimizer changed")
        actual += 1

    hook = register_optimizer_step_post_hook(counted)
    try:
        got = stream(model, x, target, 0)
    finally:
        hook.remove()
    if actual != 10 * len(range(20, len(x), 10)):
        raise ValueError(f"unexpected live optimizer step count {actual} for {len(x)} rows")
    np.testing.assert_array_equal(got[REFIT_NAME], got["r15"])
    np.testing.assert_array_equal(got[REFIT_NAME][:20], got["base"][:20])
    return got, actual


def session(stream, model, rows, source_target, relab_sha, archive, reset_rows):
    n = len(rows)
    if not PREFIX <= n <= 199:
        raise ValueError(f"unexpected §257 visit length {n}")
    independent = target_table(RELAB, relab_sha, rows, (n, 58))
    target = np.stack([source_target[int(row)] for row in rows])
    np.testing.assert_array_equal(target, independent)
    x = torch.from_numpy(model._aligned(rows, causal=True)).to(model.dev)
    xp = torch.from_numpy(model._aligned(rows[:PREFIX], causal=True)).to(model.dev)
    with torch.no_grad():
        torch.testing.assert_close(xp, x[:PREFIX], rtol=0, atol=0)
    prefix_target = target[:PREFIX].copy()
    prefix_target[:, 3:-1].reshape(PREFIX, 18, 3)[:, 20 + 10 * np.arange(18) >= PREFIX] = np.nan
    got, steps = run_stream(stream, model, x, target)
    p1, prefix_steps = run_stream(stream, model, xp, prefix_target)
    evaluate(model, reset_rows)
    p2, _ = run_stream(stream, model, xp, prefix_target)
    np.testing.assert_array_equal(p1[REFIT_NAME], p2[REFIT_NAME])
    finite = np.isfinite(got[REFIT_NAME]).all()
    active = float(np.abs(got[REFIT_NAME][20:] - got["base"][20:]).max())
    first_active = float(np.abs(got[REFIT_NAME][20] - got["base"][20]).max())
    prefix = changed(p1[REFIT_NAME], got[REFIT_NAME][:PREFIX])
    if not finite or active <= 1e-3 or first_active <= 1e-8 or prefix["changed"] or prefix["max_abs"] > TOL:
        raise ValueError(f"inactive, noncausal, or nonfinite warm refit: {active}, {first_active}, {prefix}")
    np.testing.assert_array_equal(archive[REFIT_NAME][rows[:20]], archive["base"][rows[:20]])
    return dict(rows=n, base=changed(got["base"], archive["base"][rows]),
                warm=changed(got[REFIT_NAME], archive[REFIT_NAME][rows]), steps=steps,
                prefix_steps=prefix_steps, prefix=prefix, first_active_abs=first_active, warm_movement_abs=active,
                first_20_exact=True, interleaved_reset_exact=True, strict_targets_exact=True)


def run(family, smoke):
    if file_sha(FROZEN) != FROZEN_SHA or file_sha(RELEASE / "source.sha256") != RELEASE_SHA:
        raise ValueError("qualified frozen provenance differs")
    frozen = json.loads(FROZEN.read_text())
    if (frozen["held_labels_opened"] or frozen["order"] != ORDER or not frozen["frozen_inputs_eligible"]
            or not frozen["families"][family]["eligible"] or frozen["rows"] != 2655):
        raise ValueError(f"unqualified frozen import for {family}")
    api, stream, targets = actor()
    archive = reference(family)
    chain, relab = file_sha(CHAIN), file_sha(RELAB)
    source = file_sha(HERE / "source.sha256")
    proof = json.loads((RELEASE / "members" / f"{family}_s0.json").read_text())
    if proof["held_labels_opened"] or proof["actor_sha256"] != file_sha(FTREL):
        raise ValueError("fit source or archived actor changes")
    data = waveform_data()
    p = pinned_delivery(ROOT / "runs/shuffle_perms.npz")[ORDERS.index(ORDER)]
    if not np.array_equal(np.sort(p), np.arange(2655)) or rows_sha(p) != frozen["delivery_sha256"]:
        raise ValueError("within0 permutation disagrees with the frozen qualification")
    smoke_sha = None if smoke else os.environ["NI26_WITHIN0_WARM_SMOKE_SHA"]
    all_sessions, fits = {}, {}
    for person in PEOPLE[:1] if smoke else PEOPLE[1:]:
        entry = proof["fit_states"][person]
        checkpoint = RELEASE / "checkpoints" / f"{family}_s0_{person}.pt"
        model, fit, held = restore(api, data, family, person, checkpoint, entry["sha256"])
        if rows_sha(fit) != entry["fit_rows_sha"] or len(fit) != entry["n_fit"]:
            raise ValueError(f"historical fit excludes the wrong person {person}")
        state = state_hash(model.net)
        delivered = p[held[p]]
        np.testing.assert_array_equal(np.sort(delivered), np.flatnonzero(held))
        reset_person = "P02" if person == "P01" else "P01"
        reset = p[data["stream"][p] == f"Dreyer2026Proteus/{reset_person}|1"][:PREFIX]
        if len(reset) != PREFIX or held[reset].any():
            raise ValueError("interleaving session is missing or held")
        for visit in dict.fromkeys(data["stream"][delivered]):
            rows = delivered[data["stream"][delivered] == visit]
            all_sessions[visit] = session(stream, model, rows, targets, relab, archive, reset)
        if state_hash(model.net) != state or data["y"][held].max() != -1:
            raise ValueError(f"refit mutated a frozen source or opened gold {person}")
        fits[person] = entry
    passed = all(r[k]["changed"] == 0 and r[k]["max_abs"] <= TOL for r in all_sessions.values()
                 for k in ("base", "warm"))
    path = HERE / ("smoke" if smoke else "members") / f"{family}.json"
    with path.open("x") as handle:
        evidence = dict(family=family, order=ORDER, held_labels_opened=False, smoke=smoke, source=source,
                        actor_sha256=file_sha(FTREL), frozen_sha256=FROZEN_SHA, chain_sha256=chain,
                        relab_sha256=relab, fits=fits, sessions=all_sessions, archive_parity=passed,
                        smoke_sha256=smoke_sha, gpu=torch.cuda.get_device_name(), node=os.environ["SLURMD_NODENAME"])
        json.dump(evidence, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    print(f"within0 {family}: archive parity {passed}, receipt {file_sha(path)}", flush=True)
    if not passed:
        raise SystemExit("historical warm archive parity failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=int, choices=ACTORS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(ACTORS[args.task], args.smoke)
