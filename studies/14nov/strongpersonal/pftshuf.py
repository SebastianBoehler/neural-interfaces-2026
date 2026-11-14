"""s247b: s29's PFT nets re-streamed in each delivery order. Folds: B = seen_session (base net on every public window
except q's sessions 2 and 3, fine-tuned on q's session 1, streamed over sessions 2 and 3); A = base net without q's
session 3, fine-tuned on sessions 1 + 2, streamed over session 3. Each test session is streamed in the order the
seen_session permutation of runs/shuffle_perms.npz delivers it (causal SessionBN statistics and alignment in that order)
and in recording order ('rec', = pft.py's own streaming). Model code is s29's pft.py (make, personalise), imported.
Task t: arch = ARCHS[t // 3], seed = t % 3. Output runs/strongpersonal/pftshuf/<B|A>_<arch>_<variant>-s<seed>.npz with
keys '<order>' -> proba (n, 3) in recording layout of 'index'.
Usage: python pftshuf.py TASK | python pftshuf.py smoke"""
import importlib.util, os, sys, time
from pathlib import Path
import numpy as np
import torch

WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3])); os.environ["NI26_WS"] = str(WS)
spec = importlib.util.spec_from_file_location("pft29", WS / "studies/11nov/ttt/galv/pft.py"); P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P); protocol = P.protocol
OUT = WS / "runs/strongpersonal/pftshuf"; VARS = ("pf60", "pf200")
PF = np.load(WS / "runs/shuffle_perms.npz")
ORDERS = ["rec"] + [f"within{k}" for k in range(8)] + [f"full{k}" for k in range(4)] + [f"runs{k}" for k in range(4)]


def delivery(order, rows):
    """rows (global, recording order) of one session -> the same rows in the order's delivery order."""
    if order == "rec":
        return rows
    sc, sd = order.rstrip("0123456789"), order[len(order.rstrip("0123456789")):]
    g = PF[f"{sc}|{sd}|seen_session"].astype(int); rk = {int(x): k for k, x in enumerate(g)}
    return rows[np.argsort([rk[int(r)] for r in rows], kind="stable")]


def run(arch, d, seed, out, epochs=0, orders=ORDERS):
    per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str); t0 = time.time()
    for prot, tests, pub in (("B", ("2", "3"), ("1",)), ("A", ("3",), ("1", "2"))):
        if all((out / f"{prot}_{arch}_{v}-s{seed}.npz").exists() for v in VARS):
            print("exists", prot, arch, seed, flush=True); continue
        got = {v: {o: [] for o in orders} for v in VARS}; idx = []
        for q in ("P04", "P10"):
            te = (per == q) & np.isin(ses, tests); tr = ~te & (d["split"] != "external"); pr = np.flatnonzero((per == q) & np.isin(ses, pub))
            m = P.make(arch, d, seed)
            if epochs:
                m.epochs = epochs
            m.fit(np.flatnonzero(tr)); m.set_k(P.ARCHS[arch][1])
            nets = {v: P.personalise(m, pr, d["y"], *P.VARS[v], seed)[0] for v in VARS}
            for s in tests:
                rows = np.flatnonzero((per == q) & (ses == s)); idx.append(rows)
                for o in orders:
                    r = delivery(o, rows); X = torch.from_numpy(m._aligned(r, causal=True)).to(m.dev); back = np.argsort(r)  # rows ascend (recording order)
                    assert np.array_equal(r[back], rows)
                    with torch.no_grad():
                        for v in VARS:
                            got[v][o].append(torch.softmax(nets[v](X)[:, :3].float().cpu().double(), 1).numpy()[back])
            acc = {o: np.mean(np.concatenate(got["pf200"][o][-len(tests):]).argmax(1) == d["y"][np.concatenate(idx[-len(tests):])]) for o in ("rec", orders[-1])}
            print(f"pftshuf {prot} {arch} s{seed} {q}: pf200 acc {acc}, {time.time() - t0:.0f}s", flush=True)
        idx = np.concatenate(idx)
        for v in VARS:
            np.savez(out / f"{prot}_{arch}_{v}-s{seed}.npz", index=idx, **{o: np.concatenate(got[v][o]) for o in orders})
    print(f"DONE {arch} s{seed} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load(); archs = list(P.ARCHS)
    if sys.argv[1] == "smoke":
        out = WS / "runs/strongpersonal/pftshuf_smoke"; out.mkdir(parents=True, exist_ok=True)
        run("sh", d, 0, out, epochs=1, orders=["rec", "within0", "full0"]); return print("SMOKE OK", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); t = int(sys.argv[1]); run(archs[t // 3], d, t % 3, OUT)


if __name__ == "__main__":
    main()
