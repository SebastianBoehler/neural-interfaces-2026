"""OCUFOLD (ledger s244): the deployable form of the eye-embedding view. s234's E view concatenates each fold net's HELD-OUT
embeddings, so every person's rows come from a different net and its PCA mixes nine nets' coordinates: no package has that.
The package ships fixed eye nets trained on the ten people and could fit a PCA offline on their (in-sample) embeddings.
Dev analogue per held-out person q: the fold-q net (trained without q, exactly as ocushuf/ocunet train it: same traces in
the order's delivery standardisation, same seed, same epochs) embeds ALL rows; train rows are in-sample, q's rows held out.
  GPU step  (TASK = seed): SCHEME, PSEED -> runs/diag/ocufold/<order>/fold_s<seed>.npz: emb (10, N, 128) float16 by fold,
            proba (N, 3) of the held-out rows (checked against runs/diag/ocushuf/<order>/oc_nomix-all_loso-s<seed>.npz).
  CPU step  (view ORDER [ORDER ...]): 3 seeds x 128 = 384 columns; per held-out person: standardise and PCA 4 on the training
            rows of the fold net's embedding, project q's rows -> runs/views_emb/Edep_pca4_<order>.npz (key <order>, (N, 4)).
Usage: [SMOKE=1] SCHEME=within PSEED=0 python studies/14nov/stack/ocufold.py SEED | python studies/14nov/stack/ocufold.py view ORDER ..."""
import importlib.util, os, sys
from pathlib import Path
import numpy as np
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3])); SMOKE = bool(os.environ.get("SMOKE"))


def gpu(seed):
    sp = importlib.util.spec_from_file_location("ocushuf", WS / "studies/13nov/ocushuf.py"); O = importlib.util.module_from_spec(sp); sp.loader.exec_module(O)
    OC = O.OC; d = OC.protocol.load(); O.set_order("all_loso"); T = O.traces(d); N = len(d["y"]); allr = np.arange(N)
    order = f"{O.SCHEME}{O.PSEED}"; out = WS / "runs/diag/ocufold" / order; out.mkdir(parents=True, exist_ok=True)
    E, P, folds = [], np.full((N, 3), np.nan), []
    for fold, tr, te in OC.protocol.folds(d, "all_loso"):
        p, e = OC.fit_predict(T, d["y"], np.flatnonzero(tr), allr, seed, False, 1 if SMOKE else 60)  # training is independent of te
        E.append(e.astype(np.float16)); te = np.flatnonzero(te); P[te] = p[te]; folds.append(te); print(f"{order} s{seed} {fold} held-out acc {np.mean(p[te].argmax(1) == d['y'][te]):.3f}", flush=True)
        if SMOKE:
            np.savez(out / "smoke.npz", emb=np.stack(E)); return print("SMOKE OK", E[0].shape, flush=True)
    ref = WS / f"runs/diag/ocushuf/{order}/oc_nomix-all_loso-s{seed}.npz"
    if ref.exists():
        z = np.load(ref); m = ~np.isnan(P[:, 0]); agr = np.mean(P[m].argmax(1) == z["proba"][m].argmax(1)); print(f"held-out argmax agreement with {ref.name}: {agr:.4f}", flush=True)
        assert agr > 0.85, "the fold nets are not the ocushuf nets (GPU nondeterminism alone stays above 0.9)"
    owner = np.full(N, -1)
    for k, te in enumerate(folds):
        owner[te] = k
    np.savez(out / f"fold_s{seed}.npz", emb=np.stack(E), proba=P, owner=owner); print("written", out / f"fold_s{seed}.npz", flush=True)


def view(orders):
    from sklearn.decomposition import PCA
    sys.path.insert(0, str(WS / "experiments")); import protocol  # noqa: E402
    d = protocol.load(); N = len(d["y"])
    for o in orders:
        zs = [np.load(WS / f"runs/diag/ocufold/{o}/fold_s{s}.npz") for s in range(3)]; owner = zs[0]["owner"]; assert all((z["owner"] == owner).all() for z in zs)
        out = np.zeros((N, 4))
        for k in range(zs[0]["emb"].shape[0]):
            X = np.concatenate([z["emb"][k].astype(np.float64) for z in zs], 1); te = owner == k; tr = ~te
            mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-9); m = PCA(4, random_state=0).fit((X[tr] - mu) / sd); out[te] = m.transform((X[te] - mu) / sd)
        np.savez(WS / f"runs/views_emb/Edep_pca4_{o}.npz", **{o: out}); print("written", f"runs/views_emb/Edep_pca4_{o}.npz", out.shape, flush=True)


if __name__ == "__main__":
    view(sys.argv[2:]) if sys.argv[1] == "view" else gpu(int(sys.argv[1]))
