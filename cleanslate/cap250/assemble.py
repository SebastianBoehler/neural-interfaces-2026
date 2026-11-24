"""Build a fresh stock or variant package from fold-scoped family artifacts."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from components import ORDINARY, SBN, Variant, assets, expected_files, part_path
from source import ROOT, add_scope, file_digest, row_digest, selection
from receipts import read_receipt

ALIGN = {"align", "k", "G", "scale", "shrink"}
CORE = {"bands", "log_glob", "k", "shrink", "mu", "sd", "W", "b", "slow_cutoff", "slow_step", "car", "nsub"}
FIXED = {
    "geometry": {"em_ev_g", "tsem_lam", "tsem_rho", "tsem_shrink", "tsem_m0", "tsem_n0", "tsem_nmin",
                 "tsem_m", "tsem_V", "route_V_hi", "route_gal", "route_pop", "route_b", "route_Q",
                 "route_names", "route_clip", "route_rho", "route_t_start", "route_tau",
                 "route_ps1_W", "route_ps1_c", "route_ramp", "route_slot_g", "route_slot_a", "route_v3"}
                | {f"ts_{key}" for key in CORE},
    "hilbert": {f"ts2_{key}" for key in CORE | {"hilbert"}},
    "identity": {f"bio_{key}" for key in ("mu", "sd", "V", "B", "W", "S", "gal", "m", "tau", "ramp", "t0")}
                | {f"own_{key}" for key in ("prior", "csp", "t_mu", "t_sd", "t_W", "t_c", "c_W", "c_c",
                                           "s_W", "s_c", "bands", "beta", "kref")},
}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as npz:
        arrays = dict(npz)
    for key, value in arrays.items():
        if value.dtype.kind not in "biufcSU" or (value.dtype.kind in "fc" and not np.isfinite(value).all()):
            raise ValueError(f"{path}: invalid array {key}")
    return arrays


def check_net(arrays, index, trace=None):
    prefix = f"net{index}_"
    fitted = {key[len(prefix):] for key in arrays if key.startswith(prefix)}
    if not ALIGN <= fitted:
        raise ValueError(f"net {index}: missing {ALIGN - fitted}")
    if trace:
        kind = "jit_file" if index == 34 else "sjit_file"
        if fitted != ALIGN | {kind} or str(arrays[prefix + kind]) != trace:
            raise ValueError(f"net {index}: expected {kind} {trace}")
    elif fitted & {"jit_file", "sjit_file"} or fitted == ALIGN:
        raise ValueError(f"net {index}: expected fitted ordinary decoder arrays")
    return {prefix + key for key in fitted}


def check_keys(arrays, family, seed):
    if family in ORDINARY:
        keys = check_net(arrays, ORDINARY[family] + seed)
    elif family == "teacher":
        keys = check_net(arrays, 34, "teacher.pt")
    elif family in SBN:
        keys = set().union(*(check_net(arrays, index + seed, f"{name}_s{seed}.pt")
                             for index, name in SBN[family]))
    else:
        keys = FIXED[family]
    if arrays.keys() != keys:
        raise ValueError(f"{family} s{seed}: missing {keys - arrays.keys()}, extra {arrays.keys() - keys}")


def base_arrays(channels):
    sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
    import session_em as em

    def indices(names):
        return np.array([channels.index(name) for name in names], np.int64)

    # Effective exported values after the normalized R19e build and subsequent slot changes.
    shallow, learned_fir, teacher, session_bn = (
        .024999999999999994, .031249999999999993, .6249999999999999, .02083333333333333
    )
    weights = ([.125] + [shallow] * 10 + [0.] * 10 + [learned_fir] * 4 + [0.] * 10 +
               [teacher] + [session_bn] * 30 + [.125])
    return dict(n_nets=np.int64(65), member_weights=np.array(weights),
                rb_alpha=np.float64(1.15), rb_k=np.float64(5), rb_w=np.int64(30),
                em_hl=indices(em.FRONT_L), em_hr=indices(em.FRONT_R), em_vt=indices(em.VERT_TOP),
                em_vb=indices(em.VERT_BOTTOM), em_bins=np.int64(16), em_lam=np.float64(.5),
                em_n0=np.float64(60), em_m0=np.float64(20), em_shrink=np.float64(.7),
                em_iters=np.float64(2), em_nmin=np.float64(20), em_ev_vt=indices(em.EV_TOP),
                em_ev_vb=indices(em.EV_BOTTOM), em_ev_k=np.float64(5), em_ev_q=np.array([4., 8.]),
                em_ev_bins=np.int64(4), em_boost=np.float64(2), tsem_boost=np.float64(4),
                eb_T=np.float64(30), eb_gate_lo=np.float64(.2), eb_gate_hi=np.float64(.35))


def build6(base, extras, stage, tpri, count, variant):
    command = [
        sys.executable, "-B", str(ROOT / "lean/tools/build6.py"), str(base), str(extras), str(stage),
        "--v66", "--t0", "20", "--every", "10", "--steps", "5", "--lr", ".001", "--labels", "v51b",
        "--routed", "1", "--cap", "250", "--session-s", "1000000000", "--tpri-on", "1",
        "--tpri", str(tpri), "--tpri-kappa", "20", "--cf-on", "1", "--cf", str(count),
        "--relabel", "1", "--cp-on", "1",
    ]
    if variant.ttt == "off":
        command += ["--ft-off"]
    subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def install_variants(stage, optional, variant):
    code = stage / "submission.py"
    blob = code.read_bytes()
    if optional:
        sys.path.insert(0, str(ROOT / "submissions/final_stack"))
        from build_final import revised

        hard, cb = [], []
        for family, path in optional:
            digest = file_digest(path)
            shutil.copyfile(path, stage / path.name)
            (hard if family == "hard" else cb).append((path.name, digest))
        blob = revised(blob, "hardcb" if hard else "cb", cb, hard_files=hard)
    if variant.ttt.startswith("ttt"):
        sys.path.insert(0, str(ROOT / "experiments/lanes/session-adaptation/test-time-training"))
        from build_ttt import patch

        blob = patch(blob, variant.ttt.removeprefix("ttt"))
    if blob != code.read_bytes():
        code.write_bytes(blob)


def assemble(people, first, data, parts, out, variant, *, require_receipts=True):
    if out.exists():
        raise FileExistsError(out)
    with np.load(data, allow_pickle=False) as archive:
        rows, person, sessions = selection(archive, people, first)
        channels = archive["ch_names"].tolist()
    scope = row_digest(rows)
    data_hash = file_digest(data)
    visits = [len(set(sessions[person == p])) for p in people]
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cap250-", dir=out.parent) as tmp:
        base, extras, stage = (Path(tmp) / name for name in ("base", "extras", "package"))
        base.mkdir()
        extras.mkdir()
        z, optional = base_arrays(channels), []
        for family, seed in variant.fits():
            path = part_path(parts, family, seed)
            if require_receipts:
                read_receipt(path, family, seed, scope, data_hash)
            files = expected_files(family, seed)
            if set(p.name for p in path.iterdir() if p.name not in ("receipt.json", "diagnostics")) != set(files):
                raise ValueError(f"unexpected artifact inventory for {path}: expected {files}")
            if "weights.npz" in files:
                values = load_arrays(path / "weights.npz")
                check_keys(values, family, seed)
                if z.keys() & values.keys():
                    raise ValueError(f"duplicate array in {path}: {z.keys() & values.keys()}")
                z.update(values)
            for name in assets(family, seed):
                if family in ("hard", "cbramod"):
                    optional.append((family, path / name))
                elif family not in ("geometry", "count"):
                    shutil.copyfile(path / name, (extras if family in ("eye", "rest") else base) / name)
        count = part_path(parts, "count", 0) / "cf_params.npz"
        tpri = part_path(parts, "geometry", 0) / "tpri.npz"
        prior = load_arrays(tpri)
        if (tuple(z["route_names"]) != people or tuple(prior["names"]) != people
                or tuple(prior["sessions"]) != tuple(visits) or tuple(z["bio_m"]) != tuple(visits)):
            raise ValueError("route, spectral and TSEM enrollment must match source visits")
        np.savez(base / "weights.npz", **z)
        build6(base, extras, stage, tpri, count, variant)
        install_variants(stage, optional, variant)
        if require_receipts:
            sys.path[:0] = [str(ROOT / "cleanslate/harness"),
                            str(ROOT / "2026-competition/tracks/bci_decoding"), str(ROOT / "2026-competition")]
            from stream import load_solver
            load_solver(stage, channels, "cpu")
        manifest = "".join(f"{file_digest(p)}  {p.name}\n" for p in sorted(stage.iterdir()))
        stage.rename(out)
        out.with_suffix(out.suffix + ".sha256").write_text(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_scope(parser)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.people, args.first_session_only, args.data, args.parts, args.out, Variant.from_env())
