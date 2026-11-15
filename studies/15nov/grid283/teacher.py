"""s283 teacher posteriors: REVE rv2 (s280 F2 recipe, reve_fm.FMNet) fine-tuned per held-out person and seed on the nine people's windows, then run on those
SAME training windows (inputs aligned per session as in training, non-causal) -> runs/diag/reve283/teacher_s<seed>_<held>.npz (index, proba, held, plus the
held-out person's causal recording-order posteriors and accuracies as a sanity line against s280 rv2 ~55). The held-out person's windows are never used for training or teacher inference.
Usage: python studies/15nov/grid283/teacher.py TASK (TASK = seed*10 + fold, 30 tasks) | smoke"""
import os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(WS / "studies/15nov/fm280")]
import numpy as np, torch  # noqa: E402
import reve_fm as R  # noqa: E402  (puts experiments/ on the path, patches deep.build for REVE)
protocol, SB = R.protocol, R.SB
OUT = WS / "runs/diag/reve283"


def fold_run(seed, held, d, tr, te, smoke=False):
    fn = OUT / f"teacher_s{seed}_{held}.npz"
    if fn.exists():
        return print("exists", fn.name, flush=True)
    t0 = time.time(); m = R.FMNet(d, "rv2", seed); trr = np.flatnonzero(tr); rows = np.flatnonzero(te)
    if smoke:
        trr = trr[::8]; m.epochs = 1
    assert not (set(np.unique(d["subject"][rows]).tolist()) & set(np.unique(d["subject"][trr]).tolist()))
    m.fit(trr); m.net.eval(); R.check_weights(m, "rv2", d); assert m.n_sbn == 0 and not (set(np.unique(d["subject"][te]).tolist()) & m.fit_subjects)

    def probs(r, causal):
        Xa = m._aligned(r, causal=causal)
        with torch.no_grad():
            return torch.cat([torch.softmax(m.net(torch.from_numpy(Xa[i:i + 64]).to(m.dev))[:, :3].double(), 1).cpu() for i in range(0, len(Xa), 64)]).numpy()
    st = d["stream"][trr]; idx, pr = [], []
    for s in dict.fromkeys(st):
        r = trr[st == s]; idx.append(r); pr.append(probs(r, False))  # the inputs the teacher was trained on
    idx, pr = np.concatenate(idx), np.concatenate(pr); assert np.isfinite(pr).all() and len(idx) == len(trr) and set(idx.tolist()) == set(trr.tolist())
    ts = d["stream"][rows]; hp = np.zeros((len(rows), 3)); pos = {int(g): i for i, g in enumerate(rows)}
    for s in dict.fromkeys(ts):
        r = rows[ts == s]
        for g, p in zip(r, probs(r, True)):
            hp[pos[int(g)]] = p
    acc_tr, acc_te = float(np.mean(pr.argmax(1) == d["y"][idx])), float(np.mean(hp.argmax(1) == d["y"][rows]))
    OUT.mkdir(parents=True, exist_ok=True); np.savez(fn, index=idx, proba=pr, held=str(held), seed=seed, held_rows=rows, held_proba=hp, val_log=np.array(m.val_log))
    print(f"teacher s{seed} held {held} best_ep {m.best_ep}/{len(m.val_log)} acc on own training windows {acc_tr:.3f} held-out {acc_te:.3f} (maxprob mean {pr.max(1).mean():.3f}) {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    d = protocol.load(); fl = list(protocol.folds(d, "all_loso")); assert len(fl) == 10
    if sys.argv[1] == "smoke":
        OUT = WS / "runs/diag/reve283_smoke"; h, tr, te = fl[0]; fold_run(0, h, d, tr, te, True); print("SMOKE OK"); sys.exit()
    t = int(sys.argv[1]); h, tr, te = fl[t % 10]; fold_run(t // 10, h, d, tr, te)
