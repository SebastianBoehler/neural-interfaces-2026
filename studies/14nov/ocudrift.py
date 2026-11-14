"""OCUDRIFT (ledger s207, input-layer edit of the eye net, correlation-targeted): ocushuf.py with a weight edit after each
fold's training. The first-layer kernels of every input trace channel are scaled by g_c = clip(1 - ALPHA * drift_c /
max(drift), 0.3, 1). drift_c is that channel's within-session lag-1 autocorrelation of window means, measured on the fold's
TRAINING people in recording order (slow drift within a session is what makes the eye view's errors correlated, s188).
ALPHA 0 is the paired baseline: the same retraining, no edit. New people only (all_loso).
Saved to runs/diag/ocudrift/a<ALPHA>/<SCHEME><PSEED>/oc_nomix-all_loso-s<seed>.npz (shufcombo OCU_DIR).
Usage: ALPHA=0.5 SCHEME=full PSEED=0 python studies/14nov/ocudrift.py SEED | smoke"""
import importlib.util, os, sys
from pathlib import Path
import numpy as np
import torch
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); ALPHA = float(os.environ["ALPHA"])
_s = importlib.util.spec_from_file_location("ocushuf", WS / "studies/13nov/ocushuf.py"); OS_ = importlib.util.module_from_spec(_s); _s.loader.exec_module(OS_)
OC = OS_.OC; _fit = OC.fit_predict; D = {}


def drift(T, rows):  # per channel: mean over training sessions of the lag-1 autocorrelation of window means (recording order)
    st = D["d"]["stream"]; mu = T.mean(-1); out = []
    for s in dict.fromkeys(st[rows]):
        r = rows[st[rows] == s]
        if len(r) > 5:
            out.append([np.corrcoef(mu[r[:-1], c], mu[r[1:], c])[0, 1] for c in range(T.shape[1])])
    return np.nan_to_num(np.nanmean(out, 0))


def fit_predict(T, y, tr, te, seed, mix, epochs=60):
    if ALPHA == 0:
        return _fit(T, y, tr, te, seed, mix, epochs)
    dr = np.maximum(drift(D["Trec"], tr), 0); g = np.clip(1 - ALPHA * dr / max(dr.max(), 1e-9), 0.3, 1.0)
    orig = OC.Net.forward  # the edit: after training, before prediction (fit_predict predicts through net.embed / net.fc)
    emb0 = OC.Net.embed

    def embed(self, x):
        if not getattr(self, "_edited", False) and not self.training:
            with torch.no_grad():
                for conv in (self.a, self.b):
                    conv.weight.mul_(torch.as_tensor(g, dtype=conv.weight.dtype, device=conv.weight.device)[None, :, None])
            self._edited = True
            print(f"  drift edit: gains " + " ".join(f"{v:.2f}" for v in g), flush=True)
        return emb0(self, x)
    OC.Net.embed = embed
    try:
        return _fit(T, y, tr, te, seed, mix, epochs)
    finally:
        OC.Net.embed, OC.Net.forward = emb0, orig


OC.fit_predict = fit_predict


def main():
    d = OC.protocol.load(); D["d"] = d; T = OS_.traces(d)
    OS_.RANK[:] = np.arange(len(OS_.RANK)); D["Trec"] = OS_.traces(d)  # recording-order standardisation, for the drift statistic
    OS_.RANK[OS_.ORDER] = np.arange(len(OS_.ORDER))  # back to the delivery order
    out = WS / "runs/diag/ocudrift" / f"a{ALPHA:g}" / f"{OS_.SCHEME}{OS_.PSEED}"
    if sys.argv[1] == "smoke":
        out = out / "smoke"; out.mkdir(parents=True, exist_ok=True); OC.run("nomix", d, T, "all_loso", 0, out, epochs=2); return print("SMOKE PASS", flush=True)
    out.mkdir(parents=True, exist_ok=True); OC.run("nomix", d, T, "all_loso", int(sys.argv[1]), out)


if __name__ == "__main__":
    main()
