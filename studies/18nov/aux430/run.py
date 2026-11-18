"""Reconstruct the B2_pool historical control and change just its auxiliary means."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
RUN = ROOT / "runs"
OUT = Path(__file__).resolve().parent
ORDER = sys.argv[1]
assert ORDER in ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, key):
    with np.load(path) as data:
        return data[key]


def main():
    assert (OUT / "replay.txt").read_text().splitlines()[-1] == "OK"
    manifest = RUN / "coral304_inputs/control-original.sha256"
    assert digest(manifest) == "42782d6c2b7de4a8adddfcc5282eadbc0294dd84185a31a2706a90856eb35660"
    locks = {ROOT / path: sha for sha, path in (line.split() for line in manifest.read_text().splitlines())}
    tag = f"s281_v66p_control_{ORDER}"
    provenance = RUN / f"newmember/{tag}_provenance.json"
    gates = RUN / f"newmember/{tag}_gate.npz"
    raw = RUN / f"chainout/{tag}.npz"
    cp = RUN / f"s281_v66_cp/control_{ORDER}_B2_pool.npz"
    for path in (raw, cp, gates, provenance):
        assert digest(path) == locks[path], path
    pins = json.loads(provenance.read_text())["inputs_sha256"]

    def verify_aux(files, directory, prefix):
        expected = [RUN / directory / ORDER / f"{prefix}-all_loso-s{s}.npz" for s in range(9)]
        assert list(map(Path, files)) == expected
        for path in expected:
            match = [v for k, v in pins.items() if k.endswith("/" + str(path.relative_to(ROOT)))]
            assert match == [digest(path)], path

    os.environ.update(SCHEME=ORDER[:-1], PSEED=ORDER[-1], STRICT="1", GATE_P0="1",
                      OCU_DIR=f"runs/ocushuf/{ORDER}", FT_V="r15c250", REC_PFX="ttt7",
                      T6_DIR=str(RUN / "ttt/probs7"), T17_DIR=str(RUN / "diag/ttt/probs18"),
                      FT_TAG=f"ttt18{ORDER}LPC", SAVE_OUT="in_memory_only")
    source = ROOT / "studies/13nov/shufcombo.py"
    assert digest(source) == "800ac13c8a1e1d080681da3d580076d46575b114565a0821bebb36f9c7fbd5c0"
    context = {"__file__": str(source), "__name__": "aux430"}
    # Execute the owning setup and pinned-order guards, then call its unchanged chain.
    preamble, sep, _ = source.read_text().partition("\nfor bn in S.BASES:")
    assert sep
    exec(compile(preamble, str(source), "exec"), context)
    verify_aux(context["OCS"], "ocushuf", "oc_nomix")
    verify_aux(context["RSS"], "restshuf", "rest4_sh_r1")
    base = "B2_pool"
    gate = context["gm"].gate

    def checked_gate(x, proto):
        result = gate(x, proto)
        assert np.array_equal(x, reference_input) and np.array_equal(result, reference_gate)
        return result

    reference_input = read(gates, base + "|input")
    reference_gate = read(gates, base + "|gate")
    context["gm"].gate = checked_gate
    aux_max = {}
    outputs = {}
    for count in (9, 3):
        if count == 3:
            for key, files in (("OCU", "OCS"), ("RESTR", "RSS")):
                mean = np.mean([context["ld"](f) for f in context[files][:3]], axis=0)
                value = np.log(np.clip(mean, 1e-3, 1)) if key == "OCU" else mean
                aux_max[key] = float(np.max(np.abs(value - context[key])))
                assert aux_max[key] > 0
                context[key] = value
        for arm in ("P0r", "V6Fr"):
            context["chain"](base, arm, context["Pperm"])
            value = context["OUTS"][base + "|" + arm].copy()
            if count == 9 or arm == "P0r":
                assert np.array_equal(value, read(raw, base + "|" + arm)), (count, arm)
            if arm == "V6Fr":
                outputs[count] = value
    assert not np.array_equal(outputs[3], outputs[9])
    context["gm"].gate = gate
    S = context["S"]
    S.apply({p: np.arange(S.N[p]) for p in S.PR})
    sys.path.insert(0, str(ROOT / "studies/14nov/newmember"))
    import s281_cp as final
    import score_s_v66p as metric
    sys.path.insert(0, str(ROOT / "studies/14nov/rundetect"))
    import package as K
    assert K.PROTO == "all_loso"
    ref = RUN / f"s281_v66_reference/s242c250c_{ORDER}.npz"
    metric.verify_route(json.loads((RUN / f"s281_v66_route/{ORDER}.json").read_text()),
                        ORDER, digest(ref), metric.BASES, 14)
    for arm in ("P0r", "V6Fr"):
        assert np.array_equal(read(raw, base + "|" + arm), read(ref, base + "|" + arm))
    final.verify_package_config()
    lat = K.latch(ORDER)
    hc = K.J.hyp_cache(ORDER, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
    sessions = [K.C.order_rows(ORDER, s, K.PROTO) for s in K.SESS]
    windows = read(ROOT / "data/proteus_windows.npz", "X").astype(np.float64)
    for count, value in outputs.items():
        value = value.astype(np.float64)
        filtered = K.gate(value, K.J.filt_joint(value, hc, "urn"), ORDER, lat)
        outputs[count] = final.apply_sessions(windows, value, filtered, sessions)
    assert np.array_equal(outputs[9], read(cp, "scores"))
    assert np.array_equal(outputs[9], read(RUN / f"s281_v66_cp/zero318_{ORDER}_B2_pool.npz", "scores"))
    np.savez_compressed(OUT / f"{ORDER}.npz", scores=outputs[3], raw_sha=locks[raw], cp_sha=locks[cp],
                        provenance_sha=locks[provenance], gate_sha=locks[gates],
                        max_eye_log_change=aux_max["OCU"], max_rest_change=aux_max["RESTR"])
    print("PASS", ORDER, aux_max, flush=True)


if __name__ == "__main__":
    main()
