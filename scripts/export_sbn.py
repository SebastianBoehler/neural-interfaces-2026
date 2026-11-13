"""Build cand_sbn / cand_rec (5-6 Nov): a sealed package plus per-session latent-normalisation nets as SJit members.

Usage: python scripts/export_sbn.py SRC_DIR OUT_DIR [--cfg fb_sbn_rot05] [--seeds 0-5] [--ks 2,5]
       [--drop 10-14] [--extra CFG:DROP:KS ...] [--ref-labels NAME=HARNESS.npz ...] [--ckpt-dir runs/sbn]
Arm D of the 5 Nov tournament (docs/AUDIT_SBN.md): the nets ``--drop`` (SRC's fbcnet_ea_rot05 members) get
weight 0; every seed of CFG (experiments/session_bn.py, trained on all public windows, GPU) is exported once
per test prior k as a stateful TorchScript member (submission.py SJit, key ``sjit_file``, files
sbn_k<k>_s<seed>.pt) on the package's own input stream (align keys copied from the first dropped net). Each
k group carries the dropped nets' total weight, split equally over seeds. Each ``--extra`` (REC, 6 Nov) does
the same for another slot, e.g. fbcnet_raw_sbn:15-19:20 (session_bn.make NAME_sbn, files
sbn_<NAME>_k<k>_s<seed>.pt). Every other key and file of SRC is unchanged; submission.py is the current
code. Trained nets are cached in CKPT_DIR/<cfg>-s<seed>.pt and reused when present.
Checks on all public windows, streamed per session: train-time input (EA: G, scale, k; raw: scale) and the
causal input stream equal the package's; traced members (batch, window by window, CUDA if present) equal the
eager batch-causal SessionBN probabilities; the package's labels equal the harness (rest of SRC's mix + member
probabilities -> rebalancing -> session EM, scripts/set_params.harness). Writes OUT_DIR_check.json.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
import session_em  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402
from deep import build  # noqa: E402
from export_external_teacher import ALIGN, device_free_trace  # noqa: E402
from session_bn import STATE, make, reset_stream, to_stream  # noqa: E402
from set_params import harness  # noqa: E402

sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()


def span(s):
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def trained(cfg, d, seed, ckpt_dir, train=False):
    """The cached net CKPT_DIR/<cfg>-s<seed>.pt; a missing checkpoint is trained anew only with train=True (review 13 Nov:
    a silent retrain gave the personal nets of s96 a twin base that their guard could not see)."""
    f = ckpt_dir / f"{cfg}-s{seed}.pt"
    if not f.exists() and not train:
        raise FileNotFoundError(f"{f}: no checkpoint; pass train=True (export_sbn.py --train-missing) to train a new net")
    m = make(cfg, d, seed)
    if f.exists():
        ck = torch.load(f, map_location="cpu", weights_only=False)
        m.net = build(m.arch, d["X"].shape[1], d["X"].shape[2], m.pretrained, **m.net_kw)
        m._prepare()
        m.net.load_state_dict(ck["state"])
        m.G, m.scale = ck["G"], ck["scale"]
        info = {k: ck[k] for k in ("fit_seconds", "train_device", "slurm_job")}
    else:
        t0 = time.time()
        m.fit(np.arange(len(d["y"])))
        info = {"fit_seconds": round(time.time() - t0, 1), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "train_device": torch.cuda.get_device_name(0) if str(m.dev).startswith("cuda") else str(m.dev)}
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state": {k: v.cpu() for k, v in m.net.state_dict().items()}, "G": m.G, "scale": m.scale,
                    "cfg": cfg, "seed": seed, **info}, f)
    m.net.cpu().eval()
    m.dev = "cpu"
    info["ckpt_sha256"] = sha(f)
    return m, info


# package keys that identify a replaced member's architecture
ARCH_KEY = {"fbcnet": lambda q: "fbc_n_bands" in q and "fbl_win" not in q, "fblight": lambda q: "fbl_win" in q,
            "sincshallow": lambda q: "sinc_f" in q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cfg", default="fb_sbn_rot05")
    ap.add_argument("--seeds", default="0-5")
    ap.add_argument("--ks", default="2,5")
    ap.add_argument("--drop", default="10-14")
    ap.add_argument("--extra", action="append", default=[], help="CFG:DROP:KS, e.g. fbcnet_raw_sbn:15-19:20")
    ap.add_argument("--ref-labels", action="append", default=[], help="NAME=PATH of a *_harness.npz (key new)")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "runs/sbn")
    ap.add_argument("--train-missing", action="store_true", help="train nets whose checkpoint is missing (otherwise fail)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NI26_TORCH_THREADS", "4")))
    seeds = span(a.seeds)
    groups = [("sbn", a.cfg, span(a.drop), [float(k) for k in a.ks.split(",")])]
    for e in a.extra:  # file prefix sbn_<R15 member>
        cfg, drop, ks = e.split(":")
        groups.append(("sbn_" + cfg.removesuffix("_sbn"), cfg, span(drop), [float(k) for k in ks.split(",")]))
    with np.load(a.src / "weights.npz") as f:
        z = {k: f[k] for k in f.files}
    n, w = int(z["n_nets"]), z["member_weights"].astype(np.float64)
    all_drop = [i for g in groups for i in g[2]]
    assert len(set(all_drop)) == len(all_drop) and max(all_drop) < n, all_drop

    d = protocol.load()
    st, rows = d["stream"], np.ones(len(d["y"]), bool)
    res = {"source": str(a.src), "seeds": seeds, "source_weights_sha256": sha(a.src / "weights.npz"),
           "git_rev": os.environ.get("NI26_SOURCE_REV"), "host": platform.node(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__, "groups": {}}
    InputStream = type(next(iter(load_solver(a.src, "src_streams").groups.values()))[0])
    nets, G = {}, {}
    for pre, cfg, drop, ks in groups:
        arch = make(cfg, d, 0).arch
        for i in drop:  # the replaced members: same architecture, alignment and input stream
            q = [k[len(f"net{i}_"):] for k in z if k.startswith(f"net{i}_")]
            assert ARCH_KEY[arch](q) and "jit_file" not in q and "sjit_file" not in q, (cfg, i)
            assert all(np.array_equal(z[f"net{i}_{k}"], z[f"net{drop[0]}_{k}"]) for k in ALIGN), (cfg, i)
        p = {k: z[f"net{drop[0]}_{k}"] for k in ALIGN}
        gr = res["groups"][pre] = {"cfg": cfg, "arch": arch, "dropped_nets": drop, "ks": ks,
                                   "group_weight": float(w[[1 + i for i in drop]].sum()), "members": {}}
        for s in seeds:
            nets[pre, s], gr["members"][f"s{s}"] = trained(cfg, d, s, a.ckpt_dir, train=a.train_missing)
            print(pre, "net", s, gr["members"][f"s{s}"], flush=True)

        # train-time input (EA or raw scale) == the package's input stream for these nets
        ins, m0 = InputStream(p), nets[pre, seeds[0]]
        Hs, ea = {}, 0.0
        for s in dict.fromkeys(st):
            idx = np.flatnonzero(st == s)
            ins.reset_state()
            got = np.stack([ins(d["X"][i].astype(np.float32)) for i in idx])
            Hs[s] = m0._aligned(idx, causal=True)
            ea = max(ea, float(np.abs(got - Hs[s]).max()))
        ms = [nets[pre, s] for s in seeds]
        gr["ea"] = {"G_max_rel_diff": max(float(np.abs(m.G - p["G"]).max() / np.abs(p["G"]).max()) for m in ms),
                    "scale_max_abs_diff": max(abs(m.scale - float(p["scale"])) for m in ms),
                    "k_train_net": m0.k, "k_package": float(p["k"]), "shrink_package": float(p["shrink"]),
                    "align_train_net": bool(m0.align), "align_package": bool(p["align"]),
                    "causal_stream_max_abs_diff": ea}
        assert m0.k == float(p["k"]) and bool(m0.align) == bool(p["align"]) and m0.align in (True, False), gr["ea"]
        assert gr["ea"]["G_max_rel_diff"] < 1e-9 and gr["ea"]["scale_max_abs_diff"] < 1e-9 and ea < 1e-4, gr["ea"]
        G[pre] = (p, Hs)

    # harness member probabilities (eager batch-causal SessionBN), traced export, trace checks
    a.out.mkdir(parents=True, exist_ok=True)
    P, tr_res = {}, {}
    for pre, cfg, drop, ks in groups:
        Hs = G[pre][1]
        first = next(iter(Hs.values()))
        for k in ks:
            for s in seeds:
                m = nets[pre, s]
                m.set_k(k)
                P[pre, k, s] = Pk = protocol.predict_streams(m, d, rows, proba=True)
                name = f"{pre}_k{k:g}_s{s}.pt"
                sm = to_stream(m.net, k)
                traced = device_free_trace(sm, torch.from_numpy(first[:4]))
                reset_stream(traced)
                traced.save(str(a.out / name))
                mod = torch.jit.load(str(a.out / name))
                bufs = [b for nm, b in mod.named_buffers() if nm.rsplit(".", 1)[-1] in STATE]
                r = {"state_buffers": len(bufs), "n_sbn": len(m.sbns()),
                     "saved_state_zero": all(float(b.abs().sum()) == 0 for b in bufs)}
                sm_ = lambda mm, X: torch.softmax(mm(torch.from_numpy(np.ascontiguousarray(X))).double(), 1).numpy()
                batch, one = 0.0, 0.0
                with torch.no_grad():
                    for j, (sname, H) in enumerate(Hs.items()):
                        reset_stream(mod)
                        batch = max(batch, float(np.abs(sm_(mod, H) - Pk[st == sname]).max()))
                        if j < 3:  # window by window, on a deepcopy (the harness's per-session copy)
                            cp = copy.deepcopy(mod)
                            reset_stream(cp)
                            got = np.concatenate([sm_(cp, H[i:i + 1]) for i in range(len(H))])
                            one = max(one, float(np.abs(got - Pk[st == sname]).max()))
                    r["max_abs_prob_batch_vs_eager"], r["max_abs_prob_stream_copy_vs_eager"] = batch, one
                    if torch.cuda.is_available():
                        g = torch.jit.load(str(a.out / name), map_location="cuda")
                        reset_stream(g)
                        out = torch.cat([g(torch.from_numpy(first[i:i + 1]).cuda()) for i in range(len(first))])
                        r["max_abs_prob_cuda_vs_eager"] = float(np.abs(torch.softmax(out.double(), 1).cpu().numpy()
                                                                       - Pk[st == next(iter(Hs))]).max())
                tr_res[name] = r
                print(name, r, flush=True)
                assert r["n_sbn"] > 0 and r["state_buffers"] == 3 * r["n_sbn"] and r["saved_state_zero"]
                assert batch < 1e-4 and one < 1e-4 and r.get("max_abs_prob_cuda_vs_eager", 0) < 1e-3, r
    res["trace"] = tr_res

    # package: SRC with the dropped nets at weight 0 and one SJit member per (group, k, seed), each k group of a
    # slot carrying the slot's total weight split equally over seeds
    zn = dict(z)
    w_new = w.copy()
    w_new[[1 + i for i in all_drop]] = 0.0
    adds, add_w = [], []
    for pre, cfg, drop, ks in groups:
        unit = res["groups"][pre]["group_weight"]
        for k in ks:
            for s in seeds:
                i = n + len(adds)
                zn.update({f"net{i}_{key}": v for key, v in G[pre][0].items()})
                zn[f"net{i}_sjit_file"] = np.array(f"{pre}_k{k:g}_s{s}.pt")
                adds.append((pre, k, s))
                add_w.append(unit / len(seeds))
    zn["n_nets"] = np.array(n + len(adds), dtype=z["n_nets"].dtype)
    zn["member_weights"] = np.r_[w_new, add_w]
    np.savez(a.out / "weights.npz", **zn)
    for f in a.src.iterdir():
        if f.is_file() and f.name not in ("weights.npz", "submission.py"):
            shutil.copyfile(f, a.out / f.name)
    shutil.copyfile(SRC, a.out / "submission.py")

    # harness: rest of SRC's mix (dropped nets at weight 0) + eager member probabilities -> rebalance -> EM
    with tempfile.TemporaryDirectory() as tmp:
        rest = Path(tmp)
        for f in a.src.iterdir():
            if f.is_file() and f.name != "weights.npz":
                shutil.copyfile(f, rest / f.name)
        np.savez(rest / "weights.npz", **dict(z, member_weights=w_new))
        mix_rest = stream(load_solver(rest, "rest_mix"), d, rows, lambda mm, x: mm.mixed_proba(x))
    w_rest = float(w_new.sum())
    mix = (mix_rest * w_rest + sum(wt * P[key] for key, wt in zip(adds, add_w))) / (w_rest + sum(add_w))
    X64 = d["X"].astype(np.float64)
    Fe = np.c_[session_em.gaze_features(X64, list(d["ch_names"])),
               session_em.event_features(X64, list(d["ch_names"]), st, z["em_ev_g"])]
    ref = harness(zn, mix, st, Fe)
    t0 = time.time()
    new = stream(load_solver(a.out, "new_pkg"), d, rows, lambda mm, x: mm.predict(x))
    res["package_cpu_seconds_per_window"] = round((time.time() - t0) / len(new), 4)
    old = stream(load_solver(a.src, "src_pkg"), d, rows, lambda mm, x: mm.predict(x))
    score = lambda lab: protocol.score(d["y"], lab, d, rows)["balanced_accuracy_cell_mean"]
    refs = {nm: np.load(pth)["new"] for nm, pth in (r.split("=", 1) for r in a.ref_labels)}
    res.update({"n_windows": int(len(new)), "member_weights_added": add_w,
                "member_weights_sum": float(zn["member_weights"].sum()),
                "package_matches_harness": float(np.mean(new == ref)),
                "n_windows_differ_from_harness": int(np.sum(new != ref)),
                "labels_changed_vs_source": int(np.sum(new != old)),
                "labels_changed_vs_ref": {nm: int(np.sum(new != lab)) for nm, lab in refs.items()},
                "in_sample_cell_mean": {"source": score(old), "package": score(new),
                                        **{nm: score(lab) for nm, lab in refs.items()}},
                "member_in_sample_cell_mean": {f"{pre}_k{k:g}": score(np.mean([P[pre, k, s] for s in seeds], 0).argmax(1))
                                               for pre, _, _, ks in groups for k in ks},
                "files_sha256": {f.name: sha(f) for f in sorted(a.out.iterdir()) if f.is_file()}})
    np.savez(a.out.parent / f"{a.out.name}_harness.npz", mix_rest=mix_rest, mix=mix, ref=ref, new=new, old=old,
             **{f"P_{pre}_k{k:g}_s{s}": P[pre, k, s] for pre, k, s in adds})
    (a.out.parent / f"{a.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert res["package_matches_harness"] == 1.0


if __name__ == "__main__":
    main()
