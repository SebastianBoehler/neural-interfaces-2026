"""s247 protocol A nets: s29's PFT with two public sessions. Per person q (P04, P10) the base net is trained on every
public window except q's session 3, then fine-tuned on q's sessions 1 + 2 (true labels; pft.personalise, each session
one SessionBN chunk) and streamed over q's session 3 with causal SessionBN statistics / alignment (as s29).
Model code is s29's (studies/11nov/ttt/galv/pft.py: make, personalise, ARCHS, VARS), imported, not edited.
Task t: arch = ARCHS[t // 3], seed = t % 3 (15 tasks). Output runs/strongpersonal/pft12/pft12_<arch>_<variant>-s<seed>.npz.
Usage: python pft12.py TASK | python pft12.py smoke"""
import importlib.util, os, sys, time
from pathlib import Path
import numpy as np
import torch

WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3])); os.environ["NI26_WS"] = str(WS)
spec = importlib.util.spec_from_file_location("pft29", WS / "studies/11nov/ttt/galv/pft.py"); P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P); protocol = P.protocol
OUT = WS / "runs/strongpersonal/pft12"


def run(arch, d, seed, out, epochs=0):
    names = ["base"] + list(P.VARS); per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str)
    if all((out / f"pft12_{arch}_{v}-s{seed}.npz").exists() for v in names):
        return print("exists", arch, seed, flush=True)
    got, idx, t0 = {v: [] for v in names}, [], time.time()
    for q in ("P04", "P10"):
        te = (per == q) & (ses == "3"); tr = ~te & (d["split"] != "external"); pr = np.flatnonzero((per == q) & np.isin(ses, ["1", "2"]))
        assert te.sum() > 150 and len(pr) > 350, (q, te.sum(), len(pr))
        m = P.make(arch, d, seed)
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr)); m.set_k(P.ARCHS[arch][1]); rows = np.flatnonzero(te)
        nets = {"base": m.net.eval()}
        for v, (steps, lr, scope) in P.VARS.items():
            nets[v], l0, l1 = P.personalise(m, pr, d["y"], steps, lr, scope, seed)
            print(f"pft12 {arch} s{seed} {q} {v}: {len(pr)} own rows, loss {l0:.3f} -> {l1:.3f}", flush=True)
        X = torch.from_numpy(m._aligned(rows, causal=True)).to(m.dev)
        with torch.no_grad():
            for v in names:
                got[v].append(torch.softmax(nets[v](X)[:, :3].float().cpu().double(), 1).numpy())
        idx.append(rows)
        print(f"pft12 {arch} s{seed} {q}: base acc {np.mean(got['base'][-1].argmax(1) == d['y'][rows]):.3f}, "
              f"pf60 {np.mean(got['pf60'][-1].argmax(1) == d['y'][rows]):.3f}, {time.time() - t0:.0f}s", flush=True)
    idx = np.concatenate(idx)
    for v in names:
        Pv = np.concatenate(got[v]); assert np.isfinite(Pv).all() and len(Pv) == len(idx)
        np.savez(out / f"pft12_{arch}_{v}-s{seed}.npz", proba=Pv, index=idx)
    print(f"DONE {arch} s{seed} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load(); archs = list(P.ARCHS)
    if sys.argv[1] == "smoke":
        out = WS / "runs/strongpersonal/pft12_smoke"; out.mkdir(parents=True, exist_ok=True)
        run("sh", d, 0, out, epochs=1); return print("SMOKE OK", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); t = int(sys.argv[1]); run(archs[t // 3], d, t % 3, OUT)


if __name__ == "__main__":
    main()
