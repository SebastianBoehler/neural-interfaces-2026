"""Use a fit-excluded Hilbert Core2 in the actual §321 returning worker."""

import json
from pathlib import Path

import numpy as np
import torch

import benchmark_utils.base_solver
from artifact import BASE, PILOT, PILOT_BANK, PILOT_SHA, REPO, pin, sha, source_params
from inputs import load, native_ft_proof
from record import load_model
from returning_pair_guard import zero
from worker import same_prefix, session


REF = BASE / "ni26_friday_returning_core2_reference_v1_20261119"
REF_SOURCE = "1feb8f8ab23d78822c8ac1b2fb7467f6fd77fb55f3bf5790c7a339efff5a9b0d"
GALLERY = BASE / "ni26_js321_20261117_returning_galleries2/P10_s23"
FULL_NATIVE = PILOT.parent / "visit2"
GALLERY_SHA = "f5c8804a876a6f49f9580dce4b941e12f39d20c278d9678ee3b133a8b2051873"
FULL_SHA = "82704d2bb84b13be8d26292f232213b1ce7c54eb477e2056d3817e7259a16cf6"
FULL_BANK = "434c1c30520d598c01cb71d69213a91b58f9118f949765d67963662f63e30af7"


class Audit:
    def __init__(self, model):
        self.model = model
        self.owner = model.core2.step
        model.core2.step = self.advance
        self.p, self.rest, self.eye, self.rest_effect = [], [], [], []

    def advance(self, x):
        p, z = self.owner(x)
        self.p.append(p.copy())
        return p, z

    def frame(self, trace, t):
        m, v = self.model, trace["v6"]
        if len(self.p) != t + 1 or m.core2.n != t + 1 or not m.core2.hilbert or abs(m.w[-1] - .125) > 1e-15:
            raise ValueError("the live Hilbert slot is missing or weighted incorrectly")
        if t == 0 and abs(m.w_rest - .0625) > 1e-15:
            raise ValueError("native Rest weight changed")
        eye, rest = v["extra"], v["rest"]
        if eye.shape != (3,) or rest.shape != (3,) or len(m.routeB.tail.em.A) != t + 1:
            raise ValueError("native ocular and Rest contributions are inactive")
        np.testing.assert_array_equal(m.routeB.tail.em.A[-1][-3:], eye)
        np.testing.assert_array_equal(m.rest_p, rest)
        before_rest = (v["mix6"] - m.w_rest * rest) / (1 - m.w_rest)
        self.rest.append(rest.copy())
        self.eye.append(eye.copy())
        self.rest_effect.append(float(np.abs(v["mix6"] - before_rest).max()))

    def arrays(self):
        return {k: np.asarray(getattr(self, k)) for k in ("p", "rest", "eye")}


def call(model, name, case, n):
    inspectors = []

    def instrument(m):
        audit = Audit(m)
        inspectors.append(audit)
        return audit.frame

    result = session(model, name, case, n, instrument)
    return result, inspectors[0]


def references(case, fit, artifact):
    p = REF / "reference.json"
    pin(REF / "ref_source.sha256", REF_SOURCE)
    rec = json.loads(p.read_text())
    expected = dict(fit=fit, core_bank=artifact, origin=PILOT_SHA, source=REF_SOURCE, held_labels_opened=False)
    if any(rec[k] != v for k, v in expected.items()) or rec["old_owning_path_max_abs"] > 1e-6:
        raise ValueError("independent Hilbert reference does not qualify")
    path = REF / "reference.npz"
    pin(path, rec["output"])
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["rows"], case["rows"][:31])
        return {k: z[k] for k in ("old", "new")}, sha(p)


def check_reference(actual, oracle):
    delta = float(np.abs(actual - oracle).max())
    if actual.shape != (31, 3) or not np.isfinite(actual).all() or delta > 1e-6:
        raise ValueError(f"native Hilbert differs from its independent causal oracle: {delta}")
    np.testing.assert_array_equal(actual.argmax(1), oracle.argmax(1))
    return delta


def main():
    if (torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized()
            or "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text()):
        raise ValueError("native Hilbert execution must be the clean worker CPU")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    params, fit_sha, artifact_sha, held = source_params()
    root, fit, names, weights, channels, cases = load("P10")
    case = cases["2"]
    np.testing.assert_array_equal(held, fit["held"])
    if case["rows"][0] != 1238:
        raise ValueError("unexpected P10 visit-2 prefix")
    refs, ref_sha = references(case, fit_sha, artifact_sha)
    pin(PILOT / "native_inner.json", PILOT_SHA)
    pin(PILOT / "native_inner.npz", PILOT_BANK)
    model = load_model(root, channels, "genuine_p10_core2", "cpu")
    if Path(type(model).step.__globals__["__file__"]).resolve() != (root / "submission.py").resolve():
        raise ValueError("wrong native §321 module")
    np.testing.assert_array_equal(model.w, weights)
    native_ft_proof(model, "P10")
    model.js_bank = None
    pin(GALLERY / "proof.json", GALLERY_SHA)
    gallery = json.loads((GALLERY / "proof.json").read_text())
    pin(GALLERY / "galleries.npz", gallery["arrays_sha256"])
    if (gallery["source_basis_sha256"] != fit["fit_sha"] or not gallery["old_OOF_outer_exposure_RETAINED"]
            or gallery["query_labels_read"]):
        raise ValueError("conditional gallery scope differs")
    with np.load(GALLERY / "galleries.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(model.route.contrast.Qb, z["route_Q"])
    pin(FULL_NATIVE / "native_inner.json", FULL_SHA)
    pin(FULL_NATIVE / "native_inner.npz", FULL_BANK)
    with np.load(FULL_NATIVE / "native_inner.npz", allow_pickle=False) as prior:
        np.testing.assert_array_equal(prior["rows"], case["rows"])
        contrast = np.flatnonzero(prior["causal|contrast"][:, 0] >= 0)
        if len(contrast) == 0 or contrast[0] < 31:
            raise ValueError("archived live contrast scope changed")
    with np.load(REPO / "data/proteus_windows.npz", allow_pickle=False) as z:
        rows = np.flatnonzero((z["subject"] == "Dreyer2026Proteus/P10") & (z["session"] == "1"))[:17]
        for x in z["X"][rows]:
            model.predict(torch.from_numpy(x[None]))
    model.reset_state()
    zero(model)
    zero_run, _ = call(model, "native_zero", case, 31)
    old, old_observer = call(model, "causal", case, 31)
    with np.load(PILOT / "native_inner.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["rows"], case["rows"][:31])
        for key in ("scores", "labels", "mix", "shadow", "contrast", "identity", "latch"):
            np.testing.assert_array_equal(z[f"native_zero|{key}"], zero_run[0][key])
            np.testing.assert_array_equal(z[f"causal|{key}"], old[0][key])
    model.core2 = type(model.core2)(params)
    model.reset_state()
    zero(model)
    new, new_observer = call(model, "causal", case, 31)
    call(model, "causal", cases["3"], 9)
    short, short_observer = call(model, "causal", case, 17)
    same_prefix(new, short, 17)
    measurements = {}
    for key, val in new_observer.arrays().items():
        np.testing.assert_array_equal(val[:17], short_observer.arrays()[key])
    before, after = old_observer.arrays(), new_observer.arrays()
    for key in ("rest", "eye"):
        np.testing.assert_array_equal(after[key], before[key])
    for key in ("u", "core", "unpatched_ea"):
        np.testing.assert_array_equal(new[0][key], old[0][key])
    measurements["harness_max_abs"] = {k: check_reference(p["p"], refs[k]) for k, p in (("old", before), ("new", after))}
    delta = after["p"] - before["p"]
    native = new[0]["mix"] - old[0]["mix"]
    mix_residue = float(np.abs(native - weights[-1] * delta).max())
    measurements.update(core2=float(np.abs(delta).max()), mix=float(np.abs(native).max()),
                        shadow=float(np.abs(new[0]["shadow"] - old[0]["shadow"]).max()),
                        terminal=float(np.abs(new[0]["scores"] - old[0]["scores"]).max()))
    if (mix_residue > 1e-14 or any(measurements[k] <= floor for k, floor in
            (("core2", .03), ("mix", .00375), ("shadow", .001), ("terminal", .001)))
            or max(old_observer.rest_effect) <= 1e-4 or not (np.ptp(before["eye"], axis=0) > 0).all()
            or not (old[0]["identity"][:, 0] >= 0).any()):
        raise ValueError(f"source Core2, retained auxiliaries or gallery were inactive: {measurements}, {mix_residue}")
    output = Path("hilbert_native.npz")
    with output.open("xb") as handle:
        keys = ("scores", "labels", "mix", "shadow", "contrast", "identity", "latch")
        np.savez_compressed(handle, rows=case["rows"][:31], old_core2=before["p"], new_core2=after["p"],
                            **{f"{arm}|{k}": value[0][k] for arm, value in (("old", old), ("new", new)) for k in keys})
    output.chmod(0o444)
    proof = dict(scope="conditional native P10|2 full0, causal EA, only fresh source Core2",
                 fit_sha256=fit_sha, fit_artifact=artifact_sha, ref_sha256=ref_sha, basis=fit["fit_sha"],
                 source=sha(Path(__file__).resolve().parent / "source.sha256"), windows=31, route_names=names,
                 native_baselines_exact=True, exact_interleaved_prefix=True, errors=measurements, mixture_residue=mix_residue,
                 retained_rest_effect=max(old_observer.rest_effect), retained_eye_exact=True, fixed_gallery=GALLERY_SHA,
                 contrast_finite=int((old[0]["contrast"][:, 0] >= 0).sum()),
                 archived_first_contrast_age=int(contrast[0]), archived_contrast_count=len(contrast),
                 bio_finite=int((old[0]["identity"][:, 0] >= 0).sum()),
                 old_activity=old[2], new_activity=new[2], held_labels_opened=False, accuracy_computed=False,
                 decision_changes=int(np.count_nonzero(old[0]["labels"] != new[0]["labels"])), output=sha(output))
    path = output.with_suffix(".json")
    with path.open("x") as handle:
        json.dump(proof, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    print("PASS fresh native Hilbert slot", sha(path), json.dumps(proof, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
