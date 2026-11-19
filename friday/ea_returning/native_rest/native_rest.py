"""Exercise a source-only Rest replacement in the actual conditional §321 worker."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import benchmark_utils.base_solver
from inputs import load, native_ft_proof
from record import load_model
from returning_pair_guard import zero
from worker import same_prefix, session


BASE = Path("/mnt/lustre/work/gehler/gfh527")
DATA = BASE / "neural-interfaces-2026/data/proteus_windows.npz"
INNER = BASE / "ni26_friday_returning_native_inner_v2_20261119"
PILOT = INNER / "smoke"
GALLERY = BASE / "ni26_js321_20261117_returning_galleries2/P10_s23"
FIT = BASE / "ni26_friday_return_rest_v1_20261119"
FIT_SOURCE = "4b520962c4910a179a5b98acfd710e4f41f31d8c9855a4fc79a20716144945f9"
FIT_SMOKE = "62a9be6b0b6bd721d3e5f00eac365b2bf5e6be7199faad3233b0fac883609e1c"
FIT_ROWS = "db255fd45af6d17fa4ba1829781bac5a92a08afa1bec4e5bf85266b56ea84790"
HELD_ROWS = "8aaa3c2c7d59a3851fb0c6cb0ece879771484a9a0151885c645df7b889e439b2"
PRECUE_ROWS = "84c8840fcefd6d09c0a95f72ff836040cd1653d8276096b38fcd90c8907ca5c0"
GALLERY_SHA = "f5c8804a876a6f49f9580dce4b941e12f39d20c278d9678ee3b133a8b2051873"
VIEW_SHA = "ab59b6e9b2bd1104db9bfec80acf4e075a3151819f9b1d33c15d26f14e46a833"
PRIOR_KEYS = ("align", "k", "G", "scale", "shrink")
MASS, N, N_PREFIX = .0625, 31, 17


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pin(path, expected):
    if sha(path) != expected:
        raise ValueError(f"changed artifact {path}")


def prepare_rest():
    pin(FIT / "fit_source.sha256", FIT_SOURCE)
    pin(FIT / "smoke/rest_s0.json", FIT_SMOKE)
    compare = dict(member="rest", smoke=False, epochs=40, steps=2160, source=FIT_SOURCE,
                   task_rows=2270, rest_rows=1152, fit_rows_sha256=FIT_ROWS, held_rows_sha256=HELD_ROWS,
                   rest_rows_sha256=PRECUE_ROWS, held_labels_opened=False, held_waveforms_used_in_fit=False)
    fits, files, priors = {}, [], []
    for seed in range(3):
        path = FIT / "fits" / f"rest_s{seed}"
        receipt = path.with_suffix(".json")
        meta = json.loads(receipt.read_text())
        if (meta["seed"] != seed or any(meta[k] != v for k, v in compare.items())
                or not meta["checks"]["exact_reset"] or meta["checks"]["source_visits"] != 12
                or any(not 0 <= meta["checks"][key] < 1e-4 for key in
                       ("input_relative_max_abs", "eager_max_abs", "causal_prefix_max_abs"))
                or any(path.with_suffix(s).stat().st_mode & 0o222 for s in (".json", ".pt", ".npz"))):
            raise ValueError(f"invalid authentic Rest source receipt seed {seed}")
        pin(path.with_suffix(".pt"), meta["trace"])
        pin(path.with_suffix(".npz"), meta["prior"])
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as bank:
            if set(bank.files) != set(PRIOR_KEYS):
                raise ValueError("nonmatching Rest InputStream prior schema")
            priors.append(dict(bank))
        files.append(path.with_suffix(".pt").name)
        fits[seed] = dict(receipt=sha(receipt), trace=meta["trace"], prior=meta["prior"], steps=meta["steps"])
    if len({item["trace"] for item in fits.values()}) != 3:
        raise ValueError("the three trained Rest traces must be distinct")
    for prior in priors[1:]:
        for key in PRIOR_KEYS:
            np.testing.assert_array_equal(prior[key], priors[0][key])
    common = priors[0]
    if (not bool(common["align"]) or float(common["k"]) != 5. or float(common["shrink"]) != .001
            or common["G"].shape != (41, 41) or np.linalg.eigvalsh(common["G"]).min() <= 0
            or not np.isfinite(common["scale"]) or float(common["scale"]) <= 0):
        raise ValueError("invalid common source prior")
    return {**{f"v6_rest_{k}": v for k, v in common.items()}, "v6_rest_files": files}, fits


class ObserveRest:
    def __init__(self, model):
        self.model = model
        if model.w_rest != MASS or len(model.rest.nets) != 3 or not model.rest.stream.align:
            raise ValueError("matched Rest path is inactive")
        self.owner = model.rest.step
        model.rest.step = self.step
        self.p, self.eye = [], []

    def step(self, x):
        p = self.owner(x)
        self.p.append(p.copy())
        return p

    def inspect(self, trace, t):
        m, v = self.model, trace["v6"]
        if len(self.p) != t + 1 or m.rest.stream.n != t + 1 or len(m.routeB.tail.em.A) != t + 1:
            raise ValueError("Rest and the outer chain did not update on the actual window")
        np.testing.assert_array_equal(v["rest"], self.p[-1])
        np.testing.assert_array_equal(m.rest_p, self.p[-1])
        np.testing.assert_array_equal(m.routeB.tail.em.A[-1][-3:], v["extra"])
        self.eye.append(v["extra"].copy())


def call(model, arm, case, n):
    state = []

    def observer(current):
        o = ObserveRest(current)
        state.append(o)
        return o.inspect

    result = session(model, arm, case, n, observer)
    return result, np.asarray(state[0].p), np.asarray(state[0].eye)


def check_reset(model, case, other, done):
    call(model, "causal", other, 9)
    short = call(model, "causal", case, N_PREFIX)
    same_prefix(done[0], short[0], N_PREFIX)
    for a, b in zip(done[1:], short[1:]):
        np.testing.assert_array_equal(a[:N_PREFIX], b)


def owner():
    root, fit, _, weights, channels, cases = load("P10")
    if fit["fit_sha"] != VIEW_SHA or cases["2"]["rows"][0] != 1238:
        raise ValueError("unexpected authorized fold or delivery")
    model = load_model(root, channels, "genuine_p10_rest", "cpu")
    if Path(type(model).step.__globals__["__file__"]).resolve() != (root / "submission.py").resolve():
        raise ValueError("wrong native §321 model")
    np.testing.assert_array_equal(model.w, weights)
    native_ft_proof(model, "P10")
    model.js_bank = None
    pin(GALLERY / "proof.json", GALLERY_SHA)
    gallery = json.loads((GALLERY / "proof.json").read_text())
    pin(GALLERY / "galleries.npz", gallery["arrays_sha256"])
    if (gallery["source_basis_sha256"] != fit["fit_sha"] or not gallery["old_OOF_outer_exposure_RETAINED"]
            or gallery["query_labels_read"]):
        raise ValueError("unqualified conditional calibration")
    with np.load(GALLERY / "galleries.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(model.route.contrast.Qb, z["route_Q"])
    with np.load(DATA, allow_pickle=False) as z:
        rows = np.flatnonzero((z["subject"] == "Dreyer2026Proteus/P10") & (z["session"] == "1"))[:17]
        for x in z["X"][rows]:
            model.predict(torch.from_numpy(x[None]))
    model.reset_state()
    zero(model)
    return model, cases


def main():
    if (torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized()
            or "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text()):
        raise ValueError("wrong native §321 container")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    z, fits = prepare_rest()
    model, cases = owner()
    case, other = cases["2"], cases["3"]
    stock, _, _ = call(model, "native_zero", case, N)
    before = call(model, "causal", case, N)
    check_reset(model, case, other, before)
    with np.load(PILOT / "native_inner.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["rows"], case["rows"][:N])
        for arm, tape in (("native_zero", stock[0]), ("causal", before[0][0])):
            for key in ("scores", "labels", "mix", "shadow", "contrast", "identity", "latch"):
                np.testing.assert_array_equal(tape[key], archive[f"{arm}|{key}"])
    model.rest = type(model.rest)(z, "cpu", FIT / "fits")
    model.reset_state()
    zero(model)
    after = call(model, "causal", case, N)
    check_reset(model, case, other, after)
    old, new = before[0][0], after[0][0]
    for key in ("mix", "shadow", "u", "core", "contrast", "identity", "latch", "out_v5", "unpatched_ea"):
        np.testing.assert_array_equal(new[key], old[key])
    np.testing.assert_array_equal(before[2], after[2])
    same_prefix(before[0], (old, after[0][1], after[0][2]), N)
    activity = before[0][2]
    if (activity != after[0][2] or activity["refit_ages"] != [20] or activity["steps"] != 20
            or activity["recognition"] != 28):
        raise ValueError("actual P10 warm path or recognition changed")
    rest = after[1] - before[1]
    outer = new["outer"] - old["outer"]
    effects = dict(rest=float(np.abs(rest).max()), outer=float(np.abs(outer).max()),
                   terminal=float(np.abs(new["scores"] - old["scores"]).max()),
                   formula=float(np.abs(outer - MASS * rest).max()))
    if (effects["rest"] <= .03 or effects["outer"] <= .001875 or effects["terminal"] <= 1e-4
            or effects["formula"] > 1e-14):
        raise ValueError(f"Rest failed the native contribution check: {effects}")
    out = Path("rest_native.npz")
    keep = ("scores", "labels", "mix", "shadow", "outer", "contrast", "identity", "latch")
    with out.open("xb") as handle:
        np.savez_compressed(handle, rows=case["rows"][:N], old_rest=before[1], new_rest=after[1],
                            **{f"{arm}|{k}": tape[k] for arm, tape in (("old", old), ("new", new)) for k in keep})
    out.chmod(0o444)
    proof = dict(scope="conditional P10 visit 2 native full0 with causal EA",
                 source=sha(Path(__file__).parent / "source.sha256"), fits=fits, fit_source=FIT_SOURCE,
                 weights=MASS, windows=N, exact_interleaved_prefix=N_PREFIX, fixed_inner_exact=True,
                 native_baselines_exact=True, fixed_gallery=sha(GALLERY / "proof.json"),
                 effects=effects, activity=activity,
                 changed=int(np.count_nonzero(new["labels"] != old["labels"])),
                 held_labels_opened=False, accuracy_computed=False, output=sha(out))
    path = out.with_suffix(".json")
    with path.open("x") as handle:
        json.dump(proof, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    print("PASS native source Rest", sha(path), proof, flush=True)


if __name__ == "__main__":
    main()
