"""Rebuild a sealed ensemble package from scratch from one config file.

Usage: python scripts/build_sealed.py CONFIG OUT_DIR [--ref REF_DIR]
       [--device cuda|mps|cpu] [--cache DIR]
Example: python scripts/build_sealed.py scripts/sealed_r6nf.json \
             runs/sealed/ensemble_all_r6nf_rebuild

Replaces the hand-run chain train_ensemble.py -> add_rebalance.py ->
add_fbcnet.py (-> add_rebalance.py -> add_fbcnet.py --raw) with a single run
(Rule 7 audit): fits v2 and every net of the config through the harness
(``experiments/ensemble.Ensemble``, nets ``deep.AlignedNet``, seeds as
listed) on the configured windows, exports ``weights.npz`` next to the
current ``submissions/ensemble/submission.py`` and checks every official-test
window, streamed per session, against the harness (agreement must be 1.0).
v2 band covariances are recomputed in a fresh cache (``--cache`` reuses one).
With a reference package (``--ref`` or the config's ``reference``) both
packages stream every PROTEUS window per session and the report gives
window-level agreement, probability differences and scores. Writes
OUT_DIR_check.json. Nothing is uploaded.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))

import features  # noqa: E402
import protocol  # noqa: E402
from add_fbcnet import fbc_params  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402
from ensemble import Ensemble, rebalance  # noqa: E402
from train_ensemble import net_params, tangent_params  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


PACKAGES = ("torch", "braindecode", "numpy", "scipy", "scikit-learn", "mne",
            "exca", "neuralset", "neuralfetch", "neuraltrain", "neuralbench")


def environment(dev):
    from importlib.metadata import version
    import torch
    rev = os.environ.get("NI26_SOURCE_REV") or subprocess.run(  # rsynced copies
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
        text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                            "--untracked-files=normal"], capture_output=True,
                           text=True).stdout.strip()
    off = subprocess.run(["git", "-C", str(ROOT / "2026-competition"), "rev-parse",
                          "HEAD"], capture_output=True, text=True).stdout.strip()
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else dev
    return {"git_rev": rev, "git_dirty": bool(dirty), "official_rev": off,
            "host": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "train_device": gpu,
            "cuda": torch.version.cuda, "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "versions": {p: version(p) for p in PACKAGES},
            "data_sha256": sha256(protocol.DATA)}


def _align(m):
    return dict(align=np.bool_(m.align), k=np.float64(m.k), G=m.G,
                scale=np.float64(m.scale), shrink=np.float64(1e-3))


def cbm_params(m):
    """Fine-tuned CBraMod (``deep.Resampled``) for ``submission.CBra``."""
    sd = {f"cbm_{k}": v.detach().cpu().float().numpy()
          for k, v in m.net.net.state_dict().items() if "mask_encoding" not in k}
    sd.update(cbm_n_times=np.int64(m.net.n_times), cbm_factor=np.float64(m.net.factor),
              **_align(m))
    return sd


def fbl_params(m):
    """FBLightConvNet for ``submission.FBL`` (FBCNet keys + window length)."""
    net = m.net.to("cpu").eval()
    assert net.attn_conv.kernel_size == 1 and net.attn_conv.weight_softmax
    sd = {k: v.detach().double().numpy() for k, v in net.state_dict().items()}
    out = {f"fbc_fir{i}": f.detach().double().numpy()
           for i, f in enumerate(net.spectral_filtering.fir_list)}
    out.update(fbc_n_bands=np.int64(len(net.spectral_filtering.fir_list)),
               fbc_W=sd["spatial_conv.0.weight"][..., 0], fbc_bw=sd["spatial_conv.0.bias"],
               fbc_rm=sd["spatial_conv.1.running_mean"], fbc_rv=sd["spatial_conv.1.running_var"],
               fbc_g=sd["spatial_conv.1.weight"], fbc_be=sd["spatial_conv.1.bias"],
               fbc_eps=np.float64(net.spatial_conv[1].eps), fbc_L=sd["final_layer.weight"],
               fbc_lb=sd["final_layer.bias"], fbl_win=np.int64(net.n_times_truncated), **_align(m))
    return out


def sinc_params(m):
    """SincShallowNet for ``submission.Sinc``: sinc FIRs built as in training."""
    net = m.net.to("cpu").eval()
    sd = {k: v.detach().double().numpy() for k, v in net.state_dict().items()}
    with torch.no_grad():
        f = net.sinc_filter_layer.build_sinc_filters().permute(3, 2, 0, 1)
    pool = net.temporal_aggregation[2]
    out = dict(sinc_f=f.double().numpy(), sinc_dw=sd["depthwiseconv.1.weight"],
               sinc_L=sd["final_layer.weight"], sinc_lb=sd["final_layer.bias"],
               sinc_eps=np.float64(net.depthwiseconv[0].eps),
               sinc_pool=np.array([pool.kernel_size[1], pool.stride[1]], np.int64), **_align(m))
    assert net.depthwiseconv[0].eps == net.temporal_aggregation[0].eps
    for i, pre in ((1, "depthwiseconv.0"), (2, "temporal_aggregation.0")):
        for a, b in (("rm", "running_mean"), ("rv", "running_var"), ("g", "weight"), ("b", "bias")):
            out[f"sinc_{a}{i}"] = sd[f"{pre}.{b}"]
    return out


def build(cfg, d, out):
    from deep import AlignedNet
    from models import TangentSpace
    from run_cv import V2
    split = {"official": d["split"] == "train",          # warm-up, honest
             "official_tv": d["split"] != "test",         # + validation split
             "all": np.ones(len(d["y"]), bool)}           # sealed
    assert cfg["train"] in split, cfg["train"]
    rows = np.flatnonzero(split[cfg["train"]])
    nets, w = [], [cfg["v2_weight"]]
    for g in cfg["members"]:
        for s in g["seeds"]:
            kw = {**cfg["net_defaults"], **g.get("kw", {})}  # per-member overrides
            nets.append(AlignedNet(d, arch=g["arch"], align=g["align"], seed=s, **kw))
            w.append(g["weight"])
    assert abs(sum(w) - 1) < 1e-9, f"member weights sum to {sum(w)}"
    rb = cfg["rebalance"]
    model = Ensemble([TangentSpace(d, **V2)] + nets, weights=np.array(w),
                     alpha=rb["alpha"], k=rb["k"], W=rb["w"])
    times = []
    for m in model.members:
        t0 = time.time()
        m.fit(rows)
        times.append(round(time.time() - t0, 1))
        print(f"fitted {getattr(m, 'arch', 'v2')} align={getattr(m, 'align', '-')} "
              f"seed={getattr(m, 'seed', '-')} in {times[-1]} s", flush=True)
    z = {f"ts_{k}": v for k, v in tangent_params(model.members[0]).items()}
    for i, m in enumerate(nets):  # export and harness both on CPU
        m.net.to("cpu")
        params = {"fbcnet": fbc_params, "cbramod": cbm_params, "fblight": fbl_params,
                  "sincshallow": sinc_params}.get(m.arch, net_params)(m)
        m.dev = "cpu"
        z.update({f"net{i}_{k}": v for k, v in params.items()})
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "weights.npz", n_nets=np.int64(len(nets)), member_weights=model.w,
             rb_alpha=np.float64(rb["alpha"]), rb_k=np.float64(rb["k"]),
             rb_w=np.int64(rb["w"]), **z)
    shutil.copyfile(SRC, out / "submission.py")
    return model, times, len(rows)


def package_stream(folder, tag, d, mask, rb):
    """Pre-rebalancing member mix streamed per session, then rebalanced."""
    P = stream(load_solver(folder, tag), d, mask, lambda m, x: m.mixed_proba(x))
    streams = d["stream"][mask]
    Q = np.concatenate([rebalance(P[streams == s], rb["alpha"], rb["k"], rb["w"])
                        for s in dict.fromkeys(streams)])
    return P, Q


def score(d, mask, pred):
    r = protocol.score(d["y"][mask], pred, d, mask)
    r.pop("cell_scores")
    return r


def compare(out, ref, d, rb):
    with np.load(out / "weights.npz") as a, np.load(ref / "weights.npz") as b:
        same = sorted(a.files) == sorted(b.files)
        diff = {k: float(np.abs(a[k].astype(np.float64) - b[k]).max())
                for k in a.files if same and a[k].dtype.kind in "fiub"}
        n_nets = int(a["n_nets"])
    res = {"reference": str(ref), "reference_weights_sha256": sha256(ref / "weights.npz"),
           "same_weight_keys": same,
           "submission_py_identical": sha256(out / "submission.py")
           == sha256(ref / "submission.py")}
    if same:
        res["max_abs_weight_diff_v2"] = max(v for k, v in diff.items() if k.startswith("ts_"))
        res["max_abs_weight_diff_by_net"] = [
            max(v for k, v in diff.items() if k.startswith(f"net{i}_"))
            for i in range(n_nets)]
    for name, mask in (("official_test", d["split"] == "test"),
                       ("all_windows", np.ones(len(d["y"]), bool))):
        Pa, Qa = package_stream(out, "cmp_new", d, mask, rb)
        Pb, Qb = package_stream(ref, "cmp_ref", d, mask, rb)
        a, b = Qa.argmax(1), Qb.argmax(1)
        res[name] = {"n_windows": int(mask.sum()),
                     "label_agreement": float(np.mean(a == b)),
                     "n_disagree": int(np.sum(a != b)),
                     "mean_abs_prob_diff": float(np.abs(Qa - Qb).mean()),
                     "max_abs_prob_diff": float(np.abs(Qa - Qb).max()),
                     "max_abs_mix_diff": float(np.abs(Pa - Pb).max()),
                     "rebuild": score(d, mask, a), "reference": score(d, mask, b)}
        for key in ("balanced_accuracy_cell_mean", "balanced_accuracy"):
            res[name][f"delta_{key}"] = (res[name]["rebuild"][key]
                                         - res[name]["reference"][key])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ref", type=Path, help="reference package (default: config)")
    ap.add_argument("--device", help="training device (sets NI26_DEVICE)")
    ap.add_argument("--cache", type=Path, help="reuse this v2 covariance cache")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.device:
        os.environ["NI26_DEVICE"] = args.device
    from deep import device
    features.CACHE = args.cache or Path(tempfile.mkdtemp(prefix="ni26_cov_"))
    d = protocol.load()
    env = environment(device())
    print(json.dumps(env, indent=1), flush=True)

    t0 = time.time()
    model, times, n_rows = build(cfg, d, args.out)
    fit_s = time.time() - t0
    te = d["split"] == "test"
    ref = protocol.predict_streams(model, d, te, proba=True).argmax(1)
    t1 = time.time()
    got = stream(load_solver(args.out, "built_pkg"), d, te, lambda m, x: m.predict(x))
    per_window = (time.time() - t1) / len(got)
    res = {"config": str(args.config), "config_sha256": sha256(args.config),
           "name": cfg["name"], "environment": env, "n_train_windows": n_rows,
           "member_weights": [round(float(x), 6) for x in model.w],
           "rebalance": cfg["rebalance"], "fit_seconds": round(fit_s, 1),
           "fit_seconds_by_member": times,
           "agreement_with_harness": float(np.mean(got == ref)),
           "submission_seconds_per_window_cpu": round(per_window, 5),
           "harness": score(d, te, ref), "submission": score(d, te, got),
           "weights_sha256": sha256(args.out / "weights.npz"),
           "submission_sha256": sha256(args.out / "submission.py")}
    reference = args.ref or (ROOT / cfg["reference"] if cfg.get("reference") else None)
    if reference:
        res["comparison"] = compare(args.out, reference, d, cfg["rebalance"])
    check = args.out.parent / f"{args.out.name}_check.json"
    check.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert res["agreement_with_harness"] == 1.0, "package diverges from harness"


if __name__ == "__main__":
    main()
