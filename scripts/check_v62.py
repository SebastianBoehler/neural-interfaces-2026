"""v6.2 check (b): the routed path with the additions against references built outside lean/src62, public sessions.

Usage: python scripts/check_v62.py PKG_V62 [--device cuda] [--sessions N] [--ref-shapes package] [--order SCHEME:SEED:PERMS.npz] [--out JSON]
Gates as shipped (the public sessions are recognised). Per window, from the package's recorded member posteriors:
  mix_B   (15/16)(v5.1b mix with the FT change of slots 5-8) + (1/16) rest, pkg.py's form;
  shadow  lean/src's (v5 tree) Tail on the v5.1b mix, no eye columns: v5.1b's own shadow, which its gates read (the
          gate-unchanged v6.2 has no shadow of chain B; before recognition RouteB's tail on mix_B is the output);
  routed  lean/src's Route at core1's share w0 (15/16) with the eye columns in its tail, fed mix_B and v5.1b's shadow
          (its gates then decide as v5.1b's; personal head, routed tail, own-space stage) -> the package's output;
  eye, rest, FT as scripts/check_v6.py (FT labels: the inner v5.1b's routed outputs of every window, the package relabels
          at recognition; asserted to happen before the first refit).
Also reported: chain B's first routed window against the inner v5.1b's, and the difference to v5.1b.
"""
import copy, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import check_v6 as C  # noqa: E402  (DEVICE, --ref-shapes and --order are read from the same argv)
from lean.src.routing import Route as RouteV5  # noqa: E402
from lean.src.stages import Tail as TailV5  # noqa: E402


def main():
    pkg = Path(sys.argv[1]); d = C.protocol.load(); torch.set_num_threads(1); z = np.load(pkg / "weights.npz")
    full = {k: z[k] for k in z.files}; routed = int(z["v6_routed"]); assert routed in (1, 2), routed
    if "--order" in sys.argv:
        sc, sd, f = C.arg("--order", "").split(":"); g = np.load(f)[f"{sc}|{sd}|all_loso"]
        C.ORDER.update({s: g[d["stream"][g] == s] for s in dict.fromkeys(d["stream"])})
    m = C.load_model(pkg, None, "v62", C.DEV); w = m.w; wr = float(z["v6_rest_w"]); t0 = int(z["v6_ft_t0"])
    O = importlib.util.spec_from_file_location("ocunet", ROOT / "studies/12nov/ocunet.py"); O_ = importlib.util.module_from_spec(O); O.loader.exec_module(O_)
    eyes = [torch.jit.load(str(pkg / str(f)), map_location=C.DEV).eval() for f in z["v6_eye_files"]]
    mem = {sl: int(k) for sl, k in zip(z["v6_slots"], z["v6_ft_member"])}; worst = dict(eye=0.0, mix=0.0, shadow=0.0, routed=0.0, ft=0.0)
    res = {}
    SS = list(dict.fromkeys(d["stream"]))
    for s in ([SS[C.arg("--only", 0)]] if "--only" in sys.argv else SS[:C.arg("--sessions", 3)]):  # --only K: one session per job
        rows = C.session_rows(d, s); o, tr, log, c = C.stream(m, d, rows); V = [t["v6"] for t in tr]
        rec = log["rec"]
        assert log["err"] == 0, log
        X = np.nan_to_num(np.asarray(d["X"][rows], np.float32))
        with torch.no_grad():  # eye, in delivery order, as the package (package shapes: batch 1)
            T = torch.from_numpy(O_.traces(dict(X=d["X"][rows], ch_names=d["ch_names"], stream=np.zeros(len(rows), int)))).to(C.DEV)
            pe = torch.stack([torch.softmax((torch.cat([e(T[k:k + 1]) for k in range(len(T))]) if C.PKG_SHAPES else e(T)).double(), 1) for e in eyes]).mean(0).cpu().numpy()
        worst["eye"] = max(worst["eye"], float(np.abs(np.log(np.clip(pe, 1e-3, 1)) - np.array([v["extra"] for v in V])).max()))
        for t, v in zip(tr, V):  # mix_B
            mm = t["mix"].copy()
            for sl, pf in v["p_ft"].items():
                ids = z[f"v6_slot_{sl}"].astype(int); W = w[1 + ids].sum(); fam = sum(w[1 + i] * v["members"][i] for i in ids) / W
                xx = np.clip(fam + pf - v["members"][mem[sl]], 1e-6, None); mm = mm - W * fam + W * xx / xx.sum()
            worst["mix"] = max(worst["mix"], float(np.abs((1 - wr) * mm + wr * v["rest"] - v["mix6"]).max()))
        sh, rt = TailV5(full), RouteV5(full, w[0] * (1 - wr)); f_rt = rt.tail.em.features
        n_tsem = z["tsem_V"].shape[1]
        for t, v, x, oo in zip(tr, V, X, o):  # v5.1b's shadow, then the routed step (v5 tree, eye columns in its tail)
            rt.tail.em.features = lambda xx, f0=f_rt, e=v["extra"]: np.concatenate([f0(xx), e])
            q = sh.step(x, t["u"][:n_tsem], t["mix"]); worst["shadow"] = max(worst["shadow"], float(np.abs(q - t["shadow"]).max()))
            r = rt.step(x, t["u"], v["mix6"], t["core1"], q); worst["routed"] = max(worst["routed"], float(np.abs(r - oo).max()))
        ftw = [k for k, v in enumerate(V) if v["p_ft"]]
        if routed == 2:  # FT on the whole session from t0, labels = the inner v5.1b's outputs (relabelled at recognition)
            assert 0 <= rec < t0, f"recognised at {rec}: refits before recognition used other labels; this reference does not cover it"
            outs = np.array([t["out_v5"] for t in tr]); sched = dict(steps=int(z["v6_ft_steps"]), every=int(z["v6_ft_every"]), t0=t0, lr=float(z["v6_ft_lr"]))
            Xf = {sl: np.stack([v["ft_inputs"][k] for v in V]) for sl, k in mem.items()}
            ref = C.ft_reference({sl: pkg / str(z[f"net{k}_sjit_file"]) for sl, k in mem.items()}, Xf, outs.argmax(1), outs.max(1).astype(np.float32), **sched)
            for sl in mem:
                got = np.array([v["p_ft"].get(sl, [np.nan] * 3) for v in V]); k_ = ~np.isnan(ref[sl][:, 0])
                assert np.array_equal(k_, ~np.isnan(got[:, 0])); worst["ft"] = max(worst["ft"], float(np.abs(got[k_] - ref[sl][k_]).max()))
        else:  # v6.2-safe: no FT after recognition
            assert all(k < rec for k in ftw), "v6.2-safe fine-tuned after recognition"
        o5 = np.array([t["out_v5"] for t in tr])
        res[s.split("/")[-1]] = dict(windows=len(rows), recognised_at=rec, ft_windows=len(ftw), refits=c.ft.refits, max_diff_v51b=float(np.abs(o - o5).max()),
                                     labels_changed_v51b=int((o.argmax(1) != o5.argmax(1)).sum()))
        print(s, json.dumps(res[s.split("/")[-1]]), flush=True)
    tol = dict(eye=1e-5, mix=1e-12, shadow=1e-10, routed=1e-10, ft=1e-4)
    ok = all(worst[k] < tol[k] for k in tol)
    out = dict(package=str(pkg), routed=routed, order=C.arg("--order", "recording"), ref_shapes="package" if C.PKG_SHAPES else "dev", max_diff=worst, tolerances=tol, sessions=res, ok=ok)
    print(json.dumps(out, indent=1))
    if "--out" in sys.argv:
        Path(C.arg("--out", "")).write_text(json.dumps(out, indent=1))
    assert ok, "v6.2 check FAILED"
    print("V62 CHECK OK")


if __name__ == "__main__":
    main()
