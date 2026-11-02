"""Oracle-first screen for the MI-transfer arm (1 Nov): do the external heads separate PROTEUS classes?

Usage: python scripts/diag/screen_external_heads.py BASE_CACHE OUT_JSON CKPT [CKPT ...]
The m7/m7r pretrained checkpoints (experiments/pretrain.py) were trained on seven external
corpora only (hundreds of participants, no PROTEUS). Their per-corpus heads are applied to the
causally EA-aligned PROTEUS windows (k 5; prior G = mean PROTEUS covariance, a negligible leak).
Reports, on dev_loso_all rows, against R19e from BASE_CACHE (scratchpad base_cache.py):
  - label-free scores, per participant AUC: MI-ness = 1 - p(rest) for PhysionetMI and High-Gamma,
    SUB-ness = p(subtraction) for Shin 2017B, |laterality| from the L/R heads;
  - the same scores session-centred (causal running mean);
  - a LOSO logistic map from the 19 centred head log-probs to the 3 classes: dla alone,
    P(right | R19e wrong), and R19e x teacher^0.5 per seed half (round-4 bar).
Nets pretrained with ALIGN "raw" get the unaligned windows scaled to unit global RMS instead.
NI26_EMB=1 saves each net's trunk embedding (the final layer's flattened input) instead of
the head log-probs, for eval_external_teacher.py NI26_PROBE=1, and skips the report.
"""

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts/diag")]
import braindecode.models as bd  # noqa: E402
import protocol  # noqa: E402
from deep import ARCHS  # noqa: E402
from features import covariances, invsqrtm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.metrics import balanced_accuracy_score as bacc  # noqa: E402

d = protocol.load()
st, y, subj = d["stream"], d["y"], d["subject"]


def causal_ea(X, k=5.0):
    C = np.concatenate([covariances(X[i:i + 512]) for i in range(0, len(X), 512)])
    G, out = C.mean(0), np.empty_like(X, dtype=np.float32)
    for s in dict.fromkeys(st):
        i = np.flatnonzero(st == s)
        n = np.arange(1, len(i) + 1)[:, None, None]
        P = invsqrtm((k * G + np.cumsum(C[i], 0)) / (k + n))
        out[i] = np.einsum("nij,njt->nit", P, X[i].astype(np.float64))
    return out


def heads(path, Xs):
    ck = torch.load(path, weights_only=False)
    meta = json.loads(ck["meta"])
    X = Xs[meta.get("align", "ea")]  # raw-pretrained nets see the unaligned stream (wave 8)
    arch = Path(path).name.split("_")[0]
    cls = getattr(bd, ARCHS[arch])
    blocks = {(m["offset"], m.get("n", len(m["classes"]))): c for c, m in meta["corpora"].items()}
    if meta.get("unified"):  # R17 saves only the shared block's MI / SUB / WG outputs
        blocks = {(0, 3): "unified"}
    n_out = max(o + n for o, n in blocks)
    kw = dict(n_chans=41, n_outputs=n_out, n_times=480)
    if "sfreq" in inspect.signature(cls).parameters:
        kw["sfreq"] = 120.0
    net = cls(**kw)
    net.load_state_dict(ck["state"])
    net.eval()
    emb = []
    if os.environ.get("NI26_EMB"):
        net.final_layer.register_forward_pre_hook(lambda m, a: emb.append(a[0].flatten(1)))
    with torch.no_grad():
        Z = torch.cat([net(torch.from_numpy(X[i:i + 256])) for i in range(0, len(X), 256)]).double()
    if emb:
        E = torch.cat(emb).numpy().astype(np.float32)
        return E, [f"emb{j}" for j in range(E.shape[1])]
    cols, names = [], []
    for (o, n), c in sorted(blocks.items()):
        cols.append(torch.log_softmax(Z[:, o:o + n], 1).numpy())
        names += [f"{c}:{k}" for k in meta["corpora"][c]["classes"]] if c in meta["corpora"] else \
            [f"unified:{k}" for k in ("mi", "sub", "wg")]
    return np.concatenate(cols, 1), names


def centred(F):
    out = np.empty_like(F)
    for s in dict.fromkeys(st):
        i = np.flatnonzero(st == s)
        cm = np.cumsum(F[i], 0) / np.arange(1, len(i) + 1)[:, None]
        out[i] = F[i] - np.r_[F[i][:1] * 0, cm[:-1]]  # mean of windows 0..t-1 (0 at t=0)
    return out


def cellmean(idx, pred):
    out = []
    for s in dict.fromkeys(st[idx]):
        for it in ("Graz", "BrainHero"):
            m = (st[idx] == s) & (d["interface"][idx] == it)
            if m.any():
                out.append(100 * bacc(y[idx][m], pred[m]))
    return float(np.mean(out))


def main():
    base, out_json, ckpts = np.load(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    raw = (d["X"] / np.sqrt((d["X"].astype(np.float64) ** 2).mean())).astype(np.float32)
    Xs = {"ea": causal_ea(d["X"]), "raw": raw}  # raw: global RMS, as the raw members' scale
    feats, fam_names, per_ckpt = {}, {}, {}
    for p in ckpts:
        F, names = heads(p, Xs)
        fam = Path(p).name.rsplit("-s", 1)[0]
        feats.setdefault(fam, []).append(F)
        fam_names[fam], per_ckpt[Path(p).stem] = names, F
    np.savez_compressed(out_json.with_suffix(".npz"), names=np.array(names), **per_ckpt)
    if os.environ.get("NI26_EMB"):
        return print({k: v.shape for k, v in per_ckpt.items()})
    res ={"checkpoints": ckpts, "families": {}}
    idx = base["s05_dev_loso_all_idx"]
    parts = sorted(set(subj[idx]))
    for fam, Fs in feats.items():
        F = np.mean(Fs, 0)
        Fc = centred(F)
        col = {n: j for j, n in enumerate(fam_names[fam])}
        P = np.exp(F)
        lat = [c for c in ("cho2017", "lee2019_mi", "dreyer2023", "shin2017a") if f"{c}:left_hand" in col]
        scores = {} if not ({"physionetmi:rest", "schirrmeister2017:rest", "shin2017b:subtraction"} <= col.keys()
                            and lat) else {
            "mi_phys": 1 - P[:, col["physionetmi:rest"]],
            "mi_hgd": 1 - P[:, col["schirrmeister2017:rest"]],
            "sub_shin": P[:, col["shin2017b:subtraction"]],
            "abs_lat": np.abs(np.mean([F[:, col[f"{c}:left_hand"]] - F[:, col[f"{c}:right_hand"]]
                                       for c in lat], 0)),
        }
        target = {"mi_phys": 0, "mi_hgd": 0, "abs_lat": 0, "sub_shin": 1}
        auc = {}
        for name, sc in scores.items():
            for tag, v in (("raw", sc), ("centred", centred(sc[:, None])[:, 0])):
                per = {p.split("/")[-1]: round(float(roc_auc_score(y[idx][subj[idx] == p] == target[name],
                                                                     v[idx][subj[idx] == p])), 3)
                       for p in parts}
                auc[f"{name}_{tag}"] = {"mean": round(float(np.mean(list(per.values()))), 3), **per}
        # LOSO logistic map over all 10 participants; evaluated on dev_loso_all rows
        T = np.zeros((len(y), 3))
        for p in sorted(set(subj)):
            te = subj == p
            lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(Fc[~te], y[~te])
            T[te] = lr.predict_proba(Fc[te])
        fam_res = {"auc": auc, "teacher_alone_dla": round(cellmean(idx, T[idx].argmax(1)), 2)}
        for half in ("s02", "s35", "s05"):
            ix, R = base[f"{half}_dev_loso_all_idx"], base[f"{half}_dev_loso_all_R19e"]
            wrong = R.argmax(1) != y[ix]
            Q = R * T[ix] ** 0.5
            fam_res[half] = {"R19e": round(cellmean(ix, R.argmax(1)), 2),
                             "poe05": round(cellmean(ix, Q.argmax(1)), 2),
                             "p_right_given_wrong": round(float((T[ix].argmax(1) == y[ix])[wrong].mean()), 3)}
        res["families"][fam] = fam_res
        print(fam, json.dumps(fam_res, indent=1), flush=True)
    out_json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
