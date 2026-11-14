"""FTREL (ledger s221): s15 fine-tuning (ttt18 FORMS=s15, labels R = the v6 chain = v6.1's labels K for new people) with
  r15  prefix relabelling: at the refit at t, every window 0..t-1 carries the label chain's output under its state at t
       (runs/labels/relab_v6r_<order>.npz, relabel.py), not the output emitted when it arrived;
  e15  even visits: the loss weights of the prefix are water-filled so that every window has the same cumulative weight
       after each refit (budget t per refit = a batch of today's size t drawn with those probabilities, in expectation);
       the first refit is uniform, then about half the mass goes to the windows that arrived since the last refit;
  b15  both (the panel's proposal, P3);
  q15  even visits with a fixed budget of 20 per refit (a fixed batch of 20 in expectation): converges to the newest windows;
  s15  in-run control, the stock loss (bit-identical code path to ttt7.stream for this form).
All arms take the as-emitted labels from relab_v6r_<order>.npz (today's v6 chain, s214 fix in), not the 13 Nov chain_v6r.
Weights instead of drawn batches: the same expected gradient, the prefix forward (SessionBN statistics, KL anchor) and the
cost unchanged. New people only (all_loso); shufcombo scores only those. Tag ttt18<order>RP (probs18), same file format.
Usage (Galvani): [KIND=v51b ARMS=s15,r15 DRAW=1 CAPS=100,150,250] SCHEME=within PSEED=0 python studies/14nov/ftrelabel/ftrel.py TASK | smoke"""
import copy, importlib.util, os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3])); ORD = f"{os.environ['SCHEME']}{os.environ['PSEED']}"
for k in ("FORMS", "SWEEP", "EXT", "CTRL"):
    assert not os.environ.get(k), f"{k} is not supported here"
KIND = os.environ.get("KIND", "v6r"); SFX = {"v6r": "R", "v51b": "L"}[KIND]  # s229: KIND=v51b, v5.1b's labels (tag ttt18<order>LP)
os.environ.setdefault("CHAIN_NPZ", str(WS / f"runs/labels/chain_{KIND}_{ORD}.npz"))
RELAB = os.environ.get("RELAB_NPZ", str(WS / f"runs/labels/relab_{KIND}_{ORD}.npz"))
_s = importlib.util.spec_from_file_location("ttt18", WS / "studies/13nov/ttt18.py"); T18 = importlib.util.module_from_spec(_s); _s.loader.exec_module(T18)
T7 = T18.T7; T5 = T7.T5; STOCK = T7.stream
DRAW = int(os.environ.get("DRAW", 0))  # s243: fresh fine-tuning draws (seed + 1000 DRAW, as ttt7), tag ttt18<order>d<DRAW><SFX>P
assert T7.TAG == f"ttt18{ORD}{f'd{DRAW}' if DRAW else ''}{SFX}" and T7.DRAW == DRAW and not T7.EMA and not T7.CAP, (T7.TAG, T7.EMA, T7.CAP)
S15 = T18.SWEEP["s15"]; assert S15 == (5, 10, 1e-3, "all", 1.0, "hard", True, 1, True, 20)
MODE = {"s15": (False, None), "r15": (True, None), "e15": (False, "t"), "b15": (True, "t"), "q15": (False, 20)}  # (relabel, budget)
ARMS = os.environ.get("ARMS", ",".join(MODE)).split(","); assert "s15" in ARMS and set(ARMS) <= set(MODE), ARMS  # s229: ARMS=s15,r15
CAPS = [int(c) for c in os.environ.get("CAPS", "").split(",") if c]  # s257: r15 with refits stopped at window N (last net kept), from snapshots of the same run
assert not CAPS or "r15" in ARMS, "CAPS derive from the r15 arm"
T7.VARS = {**{v: S15 for v in ARMS}, **{f"r15c{c}": S15 for c in CAPS}}; T7.TAG += "P" + ("C" if CAPS else ""); T7.SMOKE_OUT = f"smoke_ftrel{ORD}"
RZ = np.load(RELAB); TG = RZ["tgrid"]; REL, POS = RZ["relab"], RZ["pos"]
assert (TG[0], TG[1] - TG[0]) == (S15[9], S15[1]), f"label file grid {TG[:2]} is not s15's refit grid"
EMIT = RZ["all_loso"]  # the label chain as emitted, from the same run as REL (all arms share one chain; see relabel.py)
assert np.mean(EMIT.argmax(1) == T5.CH["all_loso"].argmax(1)) > 0.94, "label file and CHAIN_NPZ are not the same chain"  # 0.948 min: relabel.py MIN_AGREE
_cl = T5.chain_labels


def chain_labels(proto):  # rows: [emitted (3) | relab (K*3) | delivery position]; seen_session has no per-refit labels
    assert proto == "all_loso", "ftrel runs new people only"
    return {g: np.concatenate([EMIT[g], REL[g].ravel(), [POS[g]]]) for g in _cl(proto)}


T5.chain_labels = chain_labels


def level(c, budget):
    """water-filling: w >= 0, sum w = budget, c + w as level as possible (the least-visited windows first)."""
    cs = np.sort(c); cum = np.cumsum(cs)
    for i in range(1, len(c) + 1):
        L = (budget + cum[i - 1]) / i
        if i == len(c) or L <= cs[i]:
            return np.maximum(L - c, 0.0)


def stream(m, X, Pc, seed):
    net = m.net.eval(); n = len(X); K = (Pc.shape[1] - 4) // 3; dev = X.device
    em, R = Pc[:, :3], Pc[:, 3:3 + 3 * K].reshape(n, K, 3)
    assert (Pc[:, -1] == np.arange(n)).all(), "the label file's delivery positions are not this session's order"
    with torch.no_grad():
        base = torch.softmax(net(X)[:, :3].double(), 1).cpu().numpy()
    out = {"base": base, **{v: base.copy() for v in T7.VARS}}
    for v, (steps, every, lr, scope, lam, form, warm, views, gpu, t0) in T7.VARS.items():
        if v not in MODE:  # s257: capped variants are filled inside the r15 loop
            continue
        rel, budget = MODE[v]; capd = {c: False for c in CAPS} if v == "r15" else {}; ds = seed + 1000 * DRAW; torch.manual_seed(ds); net2 = None; vis = np.zeros(n)
        gen = torch.Generator(device=dev).manual_seed(ds)
        for t in range(t0, n, every):
            for c in [c for c, d in capd.items() if not d and t >= c]:  # first refit at or after N: keep the last net from here on
                out[f"r15c{c}"][:t] = out[v][:t]; assert net2 is not None, "cap before the first refit"
                with torch.no_grad():  # frozen net, predicted block by block from the prefix (SessionBN causal, as below)
                    for tt in range(t, n, every):
                        hh = min(n, tt + every); out[f"r15c{c}"][tt:hh] = torch.softmax(net2(X[:hh])[:, :3].double(), 1).cpu().numpy()[tt:hh]
                capd[c] = True
            P = R[:t, (t - TG[0]) // (TG[1] - TG[0])] if rel else em[:t]; assert np.isfinite(P).all(), (v, t)
            lab = torch.from_numpy(P.argmax(1)).to(dev); wpl = torch.from_numpy(P.max(1)).float().to(dev)
            if budget is not None:
                w = level(vis[:t], t if budget == "t" else budget); vis[:t] += w; wt = torch.from_numpy(w).float().to(dev)
            if net2 is None:
                net2 = copy.deepcopy(net).eval()
            opt = torch.optim.AdamW(T7.params_of(net2, scope), lr=lr, weight_decay=0.0)
            for _ in range(steps):
                opt.zero_grad(); z = net2(X[:t])[:, :3]; zs = net2(T7.strong_dev(X[:t], gen))[:, :3]
                if budget is None:  # ttt7.stream's expressions (views 1)
                    ce = (wpl * F.cross_entropy(zs, lab, reduction="none")).sum() / wpl.sum().clamp_min(1e-6)
                    kl = F.kl_div(F.log_softmax(zs, 1), F.softmax(z.detach(), 1), reduction="batchmean")
                    loss = 0.0 + (ce + lam * kl) / 1
                else:
                    ce = (wt * wpl * F.cross_entropy(zs, lab, reduction="none")).sum() / (wt * wpl).sum().clamp_min(1e-6)
                    kl = (wt * F.kl_div(F.log_softmax(zs, 1), F.softmax(z.detach(), 1), reduction="none").sum(1)).sum() / wt.sum()
                    loss = ce + lam * kl
                loss.backward(); opt.step()
            hi = min(n, t + every)
            with torch.no_grad():
                out[v][t:hi] = torch.softmax(net2(X[:hi])[:, :3].double(), 1).cpu().numpy()[t:hi]
        for c in [c for c, d in capd.items() if not d]:  # s257: session ended before window N: the capped output is r15's
            out[f"r15c{c}"] = out[v].copy()
    if CHECK:  # smoke: the control must equal ttt7.stream's s15 on the same session
        vv = T7.VARS; T7.VARS = {"s15": S15}; ref = STOCK(m, X, em, seed); T7.VARS = vv
        dd = np.abs(ref["s15"] - out["s15"]).max(); print(f"control s15 vs ttt7.stream: max diff {dd:.2e}", flush=True); assert dd < 1e-4, dd
        for v in [a for a in ARMS if a != "s15"]:
            print(f"{v}: max change vs control {np.abs(out[v] - out['s15']).max():.3f}", flush=True)
        for c in CAPS:  # s257: identical to r15 before the cap, frozen after it
            k = min(c + (-(c - 20)) % 10, len(X)); dd = np.abs(out[f"r15c{c}"][:k] - out["r15"][:k]).max()
            print(f"r15c{c}: max diff to r15 before window {k} {dd:.1e}; after {np.abs(out[f'r15c{c}'][k:] - out['r15'][k:]).max() if k < len(X) else 0:.3f}", flush=True); assert dd < 1e-9, dd
    return out


CHECK = False; T7.stream = stream
assert T7.run.__globals__["stream"] is stream and T7.run.__globals__["T5"].chain_labels is chain_labels


def main(argv):
    global CHECK
    d = T7.protocol.load()
    if argv[1] == "smoke":
        CHECK = True; out = WS / "runs/diag/ttt" / T7.SMOKE_OUT; out.mkdir(parents=True, exist_ok=True)
        T7.run("fbl", d, "all_loso", 0, out, epochs=1); return print("SMOKE OK", flush=True)
    task = int(argv[1]); T7.OUT.mkdir(parents=True, exist_ok=True)
    T7.run(T5.ORDER[task % len(T5.ORDER)], d, "all_loso", task // len(T5.ORDER), T7.OUT)


if __name__ == "__main__":
    import sys
    main(sys.argv)
