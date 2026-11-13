"""Checks of the v6 package (lean/src6; docs/V6_DESIGN.md) against v5.1b and against the dev formulas, public sessions.

Usage: python scripts/check_v6.py PKG_V6 PKG_V51B [--device cuda] [--sessions N] [--out JSON]
A. Recognised path (gates as shipped, all 14 sessions): v6's internal v5.1b output equals PKG_V51B's on every window,
   v6's output equals v5.1b's from each session's first routed window on (bitwise), the v6 chain ran before it, and
   no session logged a v6 error.
B. Unrecognised path (both gates forced off; N sessions, default 3): per window the v6 chain against references built
   here from the dev code or from first principles, not from lean/src6:
   eye   studies/12nov/ocunet.traces on the session (recording order), the package's eye traces, mean, log clip;
   rest  the input stream written out here, the rest traces run over the whole session from zero state, first three
         outputs, mean of seeds;
   ft    the ttt7 loop: from window 30 every 10, warm start, 5 Adam steps on windows 0..t-1 with the package's own
         labels and a device generator seeded 0, then rows t..t+9 from zero-state forwards (dev form);
   mix6  pkg.py's form from the recorded member posteriors;
   tail  lean/src's (v5 tree) Rebalance -> SessionEM (features + eye columns) -> TSEM on the recorded mix6.
   Also: refits happened and parameters changed (no silent no-op).
"""
import copy, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts"), str(ROOT / "lean/tools"), str(ROOT)]
import protocol  # noqa: E402
from record import load_model  # noqa: E402
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731
DEV = arg("--device", "cpu")


def stream(m, d, rows, gates_off=False):
    c = copy.deepcopy(m)
    if gates_off:
        c.route.contrast.tau = c.route.identity.tau = np.inf
    c.reset_state(); out, tr = [], []
    for i in rows:
        out.append(np.asarray(c.step(np.nan_to_num(np.asarray(d["X"][i], np.float32)))))
        tr.append(getattr(c, "trace", {}))
    log = dict(getattr(c, "log6", {}))
    return np.array(out), tr, log, c


def check_recognised(v6, v5, d):
    res, ok = {}, True
    for s in dict.fromkeys(d["stream"]):
        rows = np.flatnonzero(d["stream"] == s); o6, t6, log, _ = stream(v6, d, rows); o5, _, _, _ = stream(v5, d, rows)
        inner = np.array([t["out_v5"] for t in t6]); rec = log["rec"]
        tail6 = np.array([t["out"] for t in t6])  # what the package emitted
        pre = slice(0, rec if rec >= 0 else len(rows))
        emits_v6 = all(bool(t["v6"]) for t in t6[pre])  # every pre-recognition window went through the v6 chain
        pre_diff = float(np.abs(o6[pre] - o5[pre]).max()) if rec != 0 else 0.0
        ok &= emits_v6 and pre_diff > 0 and bool(np.array_equal(tail6, o6))
        same_inner = bool(np.array_equal(inner, o5)); same_after = rec >= 0 and bool(np.array_equal(o6[rec:], o5[rec:]))
        ok &= same_inner and same_after and log["err"] == 0
        res[s.split("/")[-1]] = dict(windows=len(rows), recognised_at=rec, v6_windows=log["v6"], inner_equal=same_inner,
                                    pre_recognition_max_diff=pre_diff, pre_recognition_emits_v6=emits_v6,
                                    pre_recognition_labels_changed=int((o6[pre].argmax(1) != o5[pre].argmax(1)).sum()),
                                    equal_from_recognition=same_after, errors=log["err"], labels_changed=int((o6.argmax(1) != o5.argmax(1)).sum()))
    return ok, res


def ft_reference(files, X, lab, wt, steps=5, every=10, t0=30, lr=5e-4, seed=0):
    out = {}
    for s, f in files.items():
        m = torch.jit.load(str(f), map_location=DEV); g = torch.Generator(device=DEV).manual_seed(seed)
        st = [b for n, b in m.named_buffers() if n.rsplit(".", 1)[-1] in ("s1", "s2", "n")]
        ps = [p for p in m.parameters() if p.requires_grad] + [b for n, b in m.named_buffers() if n.rsplit(".", 1)[-1] in ("w", "b")]
        for p in ps:
            p.requires_grad_(True)
        x = torch.from_numpy(X[s]).to(DEV); n = len(x); P = np.full((n, 3), np.nan); L = torch.as_tensor(lab, device=DEV); W = torch.as_tensor(wt, device=DEV)

        def zero():
            for b in st:
                b.detach_(); b.zero_()
        for t in range(t0, n, every):
            opt = torch.optim.Adam(ps, lr=lr)
            for _ in range(steps):
                opt.zero_grad(); zero()
                with torch.no_grad():
                    zc = m(x[:t])[:, :3]
                zero(); sh = int(torch.randint(-15, 16, (1,), generator=g, device=DEV)); xa = torch.roll(x[:t], sh, -1)
                xa = xa + 0.2 * xa.std(-1, keepdim=True) * torch.randn(xa.shape, generator=g, device=DEV)
                xa = xa * (torch.rand(xa.shape[0], xa.shape[1], 1, generator=g, device=DEV) > 0.1).float(); za = m(xa)[:, :3]
                loss = (W[:t] * F.cross_entropy(za, L[:t], reduction="none")).sum() / W[:t].sum() + F.kl_div(F.log_softmax(za, 1), F.softmax(zc, 1), reduction="batchmean")
                loss.backward(); opt.step()
            hi = min(n, t + every); zero()
            with torch.no_grad():
                P[t:hi] = torch.softmax(m(x[:hi])[:, :3].double(), 1).cpu().numpy()[t:hi]
        out[s] = P
    return out


def check_unrecognised(v6, pkg, d, sessions):
    z = np.load(pkg / "weights.npz"); O = importlib.util.spec_from_file_location("ocunet", ROOT / "studies/12nov/ocunet.py")
    O_ = importlib.util.module_from_spec(O); O.loader.exec_module(O_); T = O_.traces(d)
    eyes = [torch.jit.load(str(pkg / str(f)), map_location=DEV).eval() for f in z["v6_eye_files"]]
    rests = [torch.jit.load(str(pkg / str(f)), map_location=DEV).eval() for f in z["v6_rest_files"]]
    from lean.src.stages import Tail as TailV5  # the v5 tree, not lean/src6
    worst = dict(eye=0.0, rest=0.0, ft=0.0, mix6=0.0, tail=0.0); logs = {}
    for s in sessions:
        rows = np.flatnonzero(d["stream"] == s); o6, tr, log, c = stream(v6, d, rows, gates_off=True); logs[s.split("/")[-1]] = log
        assert log["v6"] == len(rows) and log["err"] == 0, (s, log)
        ft_on = bool(int(z["v6_ft_on"]))
        assert (c.ft.refits > 0 and c.ft.dtheta > 0) if ft_on else (c.ft.refits == 0 and not any(t["v6"]["p_ft"] for t in tr)), \
            ("fine-tuning flag not honoured", ft_on, c.ft.refits, c.ft.dtheta)
        V = [t["v6"] for t in tr]
        o5 = np.array([t["out_v5"] for t in tr]); logs[s.split("/")[-1]].update(
            differ_share=float(np.mean(np.abs(o6 - o5).max(1) > 1e-12)), max_diff=float(np.abs(o6 - o5).max()),
            labels_changed=int((o6.argmax(1) != o5.argmax(1)).sum()), em_view_cols=int(c.tail6.em.A[0].shape[0]),
            shadow_view_cols=int(c.shadow.em.A[0].shape[0]), rest_w=float(c.w_rest))
        assert logs[s.split("/")[-1]]["em_view_cols"] == logs[s.split("/")[-1]]["shadow_view_cols"] + 3 and abs(c.w_rest - 1 / 16) < 1e-15
        with torch.no_grad():  # eye
            Tt = torch.from_numpy(T[rows]).to(DEV); pe = torch.stack([torch.softmax(e(Tt).double(), 1) for e in eyes]).mean(0).cpu().numpy()
        worst["eye"] = max(worst["eye"], float(np.abs(np.log(np.clip(pe, 1e-3, 1)) - np.array([v["extra"] for v in V])).max()))
        p = {k: z[f"v6_rest_{k}"] for k in ("k", "G", "shrink")}; cs, H = 0.0, []  # rest
        for n_, i in enumerate(rows, 1):
            x = np.nan_to_num(np.asarray(d["X"][i], np.float32)).astype(np.float64); xc = x - x.mean(-1, keepdims=True); C = xc @ xc.T / x.shape[-1]
            C = (1 - p["shrink"]) * C + p["shrink"] * np.trace(C) / len(C) * np.eye(len(C)); cs = cs + C
            w, Vv = np.linalg.eigh((p["k"] * p["G"] + cs) / (p["k"] + n_)); H.append(((Vv / np.sqrt(np.maximum(w, 1e-12))) @ Vv.T @ x).astype(np.float32))
        with torch.no_grad():
            Hx = torch.from_numpy(np.stack(H)).to(DEV); pr = []
            for r in rests:
                for nm, b in r.named_buffers():
                    if nm.rsplit(".", 1)[-1] in ("s1", "s2", "n"):
                        b.zero_()
                pr.append(torch.softmax(r(Hx)[:, :3].double(), 1))
            pr = torch.stack(pr).mean(0).cpu().numpy()
        worst["rest"] = max(worst["rest"], float(np.abs(pr - np.array([v["rest"] for v in V])).max()))
        mem = {sl: int(k) for sl, k in zip(z["v6_slots"], z["v6_ft_member"])}  # ft
        X = {sl: np.stack([v["ft_inputs"][k] for v in V]) for sl, k in mem.items()}
        outs = np.array([t["out_v5"] for t in tr]); ref = {} if not ft_on else ft_reference({sl: pkg / str(z[f"net{k}_sjit_file"]) for sl, k in mem.items()}, X,
                                                                       outs.argmax(1), outs.max(1).astype(np.float32))
        for sl in (mem if ft_on else ()):
            got = np.array([v["p_ft"].get(sl, [np.nan] * 3) for v in V]); m_ = ~np.isnan(ref[sl][:, 0])
            assert np.array_equal(m_, ~np.isnan(got[:, 0])), "fine-tuned windows differ from the schedule"
            worst["ft"] = max(worst["ft"], float(np.abs(got[m_] - ref[sl][m_]).max()))
        w = c.w; wr = float(z["v6_rest_w"])  # mix6 in pkg.py's form
        for t, v in zip(tr, V):
            mm = t["mix"].copy()
            for sl, pf in v["p_ft"].items():
                ids = z[f"v6_slot_{sl}"].astype(int); fam = sum(w[1 + i] * v["members"][i] for i in ids) / w[1 + ids].sum()
                xx = np.clip(fam + pf - v["members"][mem[sl]], 1e-6, None); mm = mm - w[1 + ids].sum() * fam + w[1 + ids].sum() * xx / xx.sum()
            worst["mix6"] = max(worst["mix6"], float(np.abs(((1 - wr) * mm + wr * v["rest"]) - v["mix6"]).max()))
        full = {k: z[k] for k in z.files}; tail = TailV5(full); feat0 = tail.em.features  # tail
        for (t, v), i, o in zip(zip(tr, V), rows, o6):
            tail.em.features = lambda x, f0=feat0, e=v["extra"]: np.concatenate([f0(x), e])
            q = tail.step(np.nan_to_num(np.asarray(d["X"][i], np.float32)), t["u"][:z["tsem_V"].shape[1]], v["mix6"])
            worst["tail"] = max(worst["tail"], float(np.abs(q - o).max()))
    return worst, logs


def main():
    pkg6, pkg5 = Path(sys.argv[1]), Path(sys.argv[2]); d = protocol.load(); torch.set_num_threads(1)
    v6, v5 = load_model(pkg6, None, "v6", DEV), load_model(pkg5, None, "v51b", DEV)
    okA, resA = check_recognised(v6, v5, d)
    ss = list(dict.fromkeys(d["stream"]))[:arg("--sessions", 3)]
    worst, logs = check_unrecognised(v6, pkg6, d, ss)
    tol = dict(eye=1e-5, rest=1e-4, ft=1e-4, mix6=1e-12, tail=1e-10)
    okB = all(worst[k] < tol[k] for k in tol)
    res = dict(recognised_ok=okA, recognised=resA, unrecognised_max_diff=worst, tolerances=tol, unrecognised_logs=logs, ok=okA and okB)
    print(json.dumps(res, indent=1, default=str))
    if "--out" in sys.argv:
        Path(arg("--out", "")).write_text(json.dumps(res, indent=1, default=str))
    assert okA and okB, "v6 check FAILED"
    print("V6 CHECK OK")


if __name__ == "__main__":
    main()
