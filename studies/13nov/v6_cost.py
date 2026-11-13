"""v6 candidate path cost on an unrecognised session (ledger §120 addendum 3; parent session, 13 Nov), A100, warm, paired:
v5.1b streaming the session + one more routed tail (rebalancing -> SessionEM -> TSEM, timed as v5.1b's own routed-tail
stages inside Route3.step) + the eye net (studies/12nov/ocunet.py Net, three seeds, per window: 7 derived traces with
running session standardisation, then batch-1 forwards) + the rest member (a shallow SessionBN net as a StreamBN
TorchScript trace, batch 1) + FT (gC5 / cum10, delta1: one net per family, studies/13nov/ft_cost_session.finetune).
Weights of the eye net and the rest member are untrained (timing only). Sessions: 190 (P01|1) and 378 (P04|1+2).
  python studies/13nov/v6_cost.py ref PKG    (.venv312: the package)   python studies/13nov/v6_cost.py comp PKG   (.venv)"""
import copy, json, os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts"), str(Path(__file__).resolve().parent)]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
import protocol  # noqa: E402
import ft_cost_session as FC  # noqa: E402


def timed(obj, name, acc):
    f = getattr(obj, name)

    def g(*a):
        FC.sync(DEV); t0 = time.perf_counter(); o = f(*a); FC.sync(DEV); acc[0] += time.perf_counter() - t0; return o
    setattr(obj, name, g)


DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Eye:
    """per-window eye-net path: derived traces, running standardisation, three seeds' forwards"""

    def __init__(self, ch_names, nseeds=3):
        spec = __import__("importlib.util").util.spec_from_file_location("ocunet", WS / "studies/12nov/ocunet.py")
        O = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(O)
        self.nets = []
        for s in range(nseeds):
            torch.manual_seed(s); self.nets.append(O.Net().to(DEV).eval())
        ix = {str(c): i for i, c in enumerate(ch_names)}; self.sel = lambda X, cs: X[[ix[c] for c in cs]].mean(0)  # noqa: E731
        self.n, self.s1, self.s2 = 0, 0.0, 0.0

    @torch.no_grad()
    def step(self, x):
        X = x.astype(np.float64); sg = lambda a: savgol_filter(a, 25, 3)  # noqa: E731
        h1 = sg(self.sel(X, ("F5", "AF3", "F3")) - self.sel(X, ("F6", "AF4", "F4"))); h2 = sg(self.sel(X, ("FC5", "C5")) - self.sel(X, ("FC6", "C6")))
        v1 = sg(self.sel(X, ("AF3", "AF4")) - self.sel(X, ("P1", "P2", "Pz"))); v2 = sg(self.sel(X, ("AF3", "AF4")) - self.sel(X, ("C1", "C2", "Cz")))
        T = np.stack([h1, h2, v1, v2, np.sqrt(h1 ** 2 + v1 ** 2), np.gradient(h1), np.gradient(v1)])
        self.n += 1; self.s1 = self.s1 + T.mean(-1); self.s2 = self.s2 + (T ** 2).mean(-1); m1 = self.s1 / self.n
        T = np.clip((T - m1[:, None]) / np.sqrt(np.maximum(self.s2 / self.n - m1 ** 2, 1e-12))[:, None], -8, 8).astype(np.float32)
        xt = torch.from_numpy(T[None]).to(DEV)
        p = torch.stack([torch.softmax(n(xt), 1) for n in self.nets]).mean(0)
        return np.log(np.clip(p.cpu().numpy()[0], 1e-3, 1))


def rest_trace(d):
    import session_bn as SB
    from deep import build
    m = SB.make("sh_sbn16", d, 0); m.net = build(m.arch, d["X"].shape[1], d["X"].shape[2], m.pretrained, **m.net_kw); m._prepare()  # shallow SessionBN, timing only
    net = SB.to_stream(m.net, 20.0); x = torch.randn(4, 41, 480)
    tr = torch.jit.trace(net, x, check_trace=False); SB.reset_stream(tr); return tr.to(DEV)


def main():
    mode, pkg = sys.argv[1], Path(sys.argv[2]); d = protocol.load(); S = FC.sessions(d); torch.set_num_threads(1)
    if mode == "ref":
        from add_rebalance import load_solver
        m = load_solver(pkg, pkg.name, DEV)
        for tag, rows in S.items():
            c = copy.deepcopy(m); c.reset_state(); acc = [0.0]
            for st in (c.route.rb, c.route.em, c.route.tsem):
                timed(st, "step", acc)
            FC.sync(DEV); t0 = time.perf_counter()
            for i in rows:
                c.predict(torch.from_numpy(np.ascontiguousarray(d["X"][i:i + 1], np.float32)).to(DEV))
            FC.sync(DEV)
            if tag != "warm":
                print(json.dumps(dict(mode="ref", session=tag, windows=len(rows), v51b_s=round(time.perf_counter() - t0, 1), routed_tail_s=round(acc[0], 2))), flush=True)
        return
    eye_proto, rest = Eye(d["ch_names"]), rest_trace(d)
    from sbn_trace_base import base_from_package
    z = np.load(pkg / "weights.npz"); idx = {str(z[f"net{i}_sjit_file"]): i for i in range(int(z["n_nets"])) if f"net{i}_sjit_file" in z.files}
    base = {f: base_from_package(pkg, idx[FC.FAM[f].format(0)], d)[0] for f in FC.FAM}
    for tag, rows in S.items():
        eye = copy.deepcopy(eye_proto); FC.sync(DEV); t0 = time.perf_counter()
        for i in rows:
            eye.step(d["X"][i])
        FC.sync(DEV); eye_s = time.perf_counter() - t0
        r = copy.deepcopy(rest); SB_state = [b for n, b in r.named_buffers() if n.rsplit(".", 1)[-1] in ("s1", "s2", "n")]
        for b in SB_state:
            b.zero_()
        FC.sync(DEV); t0 = time.perf_counter()
        with torch.no_grad():
            for i in rows:
                torch.softmax(r(torch.from_numpy(np.ascontiguousarray(d["X"][i:i + 1], np.float32)).to(DEV)).double(), 1).cpu()
        FC.sync(DEV); rest_s = time.perf_counter() - t0
        X = {f: torch.from_numpy(np.ascontiguousarray(m._aligned(rows, True))).to(DEV) for f, m in base.items()}
        with torch.no_grad():
            p = torch.softmax(base["fb"].net.to(DEV)(X["fb"])[:, :3], 1)
        lab, w = p.argmax(1), p.max(1).values; ft = {}
        for sched, steps in FC.SCHED.items():
            ft[sched] = round(FC.finetune([base[f].net for f in FC.FAM], [X[f] for f in FC.FAM], lab, w, steps, DEV)[0], 1)
        if tag != "warm":
            print(json.dumps(dict(mode="comp", session=tag, windows=len(rows), eye_s=round(eye_s, 2), rest_s=round(rest_s, 2),
                                  ft_delta1_s=ft)), flush=True)


if __name__ == "__main__":
    main()
