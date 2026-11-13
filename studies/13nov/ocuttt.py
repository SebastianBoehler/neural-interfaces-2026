"""OCUTTT (ledger s188 form B): the eye net (studies/12nov/ocunet.py, arm nomix, ocushuf's delivery-order trace
standardisation) fine-tuned per test session on the windows delivered so far, like gC5 for the SessionBN nets.
Every STEP windows from T0 on, a copy of the fold's net is fine-tuned (all parameters, BN in eval mode, AdamW, STEPS
steps) on the delivered windows and predicts the next STEP windows. Two label sources:
  pl    pseudo-labels: the chain's outputs as emitted in that order (CHAIN_NPZ, e.g. chain_v6r_<order>.npz), weighted
        by confidence (the shippable form)
  true  the true labels (ceiling only, never shippable)
New people (all_loso). Saved as runs/diag/ocuttt/<SCHEME><PSEED>/{base,pl,true}/oc_nomix-all_loso-s<seed>.npz so that
shufcombo's OCU_DIR can read each variant directly.
Usage: SCHEME=within PSEED=0 CHAIN_NPZ=studies/13nov/chain_v6r_within0.npz python studies/13nov/ocuttt.py SEED | smoke"""
import copy, importlib.util, os, sys, time
from pathlib import Path
import numpy as np
import torch
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); SCHEME = os.environ["SCHEME"]; PSEED = int(os.environ["PSEED"])
T0, STEP, STEPS, LR = 30, 10, 10, 5e-4
_s = importlib.util.spec_from_file_location("ocushuf", WS / "studies/13nov/ocushuf.py"); OS_ = importlib.util.module_from_spec(_s); _s.loader.exec_module(OS_)
OC = OS_.OC; DEV = OC.DEV; LB = os.environ["CHAIN_NPZ"]
assert os.path.basename(LB) in (f"chain_v6r_{SCHEME}{PSEED}.npz", f"chain_v51b_{SCHEME}{PSEED}.npz"), f"label file {LB} is not this order's"
CH = np.load(WS / LB)["all_loso"]


def fit(T, y, tr, seed, epochs):
    """ocunet.fit_predict's training loop (nomix), returning the net."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); net = OC.Net().to(DEV); Xt, yt = torch.from_numpy(T[tr]), torch.from_numpy(y[tr])
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); nb = int(np.ceil(len(tr) / 64)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * nb)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(tr), generator=g)
        for b in range(nb):
            j = perm[b * 64:(b + 1) * 64]
            if len(j) < 8:
                sched.step(); continue
            xb, yb = Xt[j].to(DEV), torch.nn.functional.one_hot(yt[j], 3).float().to(DEV)
            xb = xb * (0.8 + 0.4 * torch.rand(len(j), 1, 1, generator=g).to(DEV)) + 0.05 * torch.randn(xb.shape, generator=g).to(DEV)
            opt.zero_grad(); loss = -(yb * torch.log_softmax(net(xb), 1)).sum(1).mean(); loss.backward(); opt.step(); sched.step()
    return net.eval()


def stream(net, X, lab, w, seed):
    """one session in delivery order -> (n, 3): base before T0, then the fine-tuned copy for each block of STEP windows."""
    with torch.no_grad():
        P = torch.softmax(net(X).double(), 1).cpu().numpy()
    torch.manual_seed(seed)
    for t in range(T0, len(X), STEP):
        n2 = copy.deepcopy(net).eval(); opt = torch.optim.AdamW(n2.parameters(), lr=LR, weight_decay=0.0)
        for _ in range(STEPS):
            opt.zero_grad(); loss = (torch.nn.functional.cross_entropy(n2(X[:t]), lab[:t], reduction="none") * w[:t]).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            P[t:t + STEP] = torch.softmax(n2(X[t:t + STEP]).double(), 1).cpu().numpy()
    return P


def main():
    smoke = sys.argv[1] == "smoke"; seed = 0 if smoke else int(sys.argv[1]); d = OC.protocol.load(); T = OS_.traces(d); y = d["y"]; st = d["stream"]
    out = WS / "runs/diag/ocuttt" / f"{SCHEME}{PSEED}"; res = {v: {} for v in ("base", "pl", "true")}; t0 = time.time()
    for fold, tr, te in OC.protocol.folds(d, "all_loso"):
        net = fit(T, y, np.flatnonzero(tr), seed, 2 if smoke else 60)
        for s in dict.fromkeys(st[te]):
            r = np.flatnonzero(te & (st == s)); r = r[np.argsort(OS_.RANK[r], kind="stable")]  # the session in delivery order
            X = torch.from_numpy(T[r]).to(DEV)
            with torch.no_grad():
                res["base"].update(zip(r, torch.softmax(net(X).double(), 1).cpu().numpy()))
            for v, lab, w in (("pl", CH[r].argmax(1), CH[r].max(1)), ("true", y[r], np.ones(len(r)))):
                P = stream(net, X, torch.from_numpy(lab.astype(np.int64)).to(DEV), torch.from_numpy(w.astype(np.float32)).to(DEV), seed); res[v].update(zip(r, P))
            if smoke:
                break
        acc = {v: np.mean([res[v][i].argmax() == y[i] for i in res[v]]) for v in res}
        dpt = max(np.abs(res["pl"][i] - res["true"][i]).max() for i in res["pl"]); dpb = max(np.abs(res["pl"][i] - res["base"][i]).max() for i in res["pl"])
        print(f"  max |pl - true| {dpt:.4f}, max |pl - base| {dpb:.4f} (both must be > 0: the label source and the fine-tuning must act)", flush=True)
        assert dpt > 0 and dpb > 0, "fine-tuning is a no-op or ignores the label source"
        print(f"s{seed} {fold}: window acc so far " + " ".join(f"{v} {a:.3f}" for v, a in acc.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        if smoke:
            return print("SMOKE PASS", flush=True)
    for v, m in res.items():
        idx = np.array(sorted(m)); (out / v).mkdir(parents=True, exist_ok=True)
        np.savez(out / v / f"oc_nomix-all_loso-s{seed}.npz", proba=np.stack([m[i] for i in idx]), index=idx)
    print("TOTAL " + " ".join(f"{v} acc {np.mean([m[i].argmax() == y[i] for i in m]):.4f}" for v, m in res.items()) + f" {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
