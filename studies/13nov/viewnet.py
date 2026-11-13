"""VIEWNET (13 Nov, ledger s158): view nets on label-free signals the mix does not use, built like the eye net
(studies/12nov/ocunet.py: 7 derived traces per window, causal per-session standardisation, 1-D ResNet with channel
attention, leave one person out and the seen protocol), for SessionEM's view inside the combination (s155: a perfect
view of this kind adds +3.8; mix members as view columns hurt, s157).
  KIND=emg  muscle activity: log 30-45 Hz envelopes at TP7, TP8, F5/F6, AF3/AF4, C5, C6 and the all-channel mean
  KIND=scp  slow cortical potentials: < 1 Hz traces at Fz, FCz, Cz, CPz, Pz, C3, C4
  KIND=ocuslow / ocufast / ocuh1 / ocuh2 / asym / alpha / beta / theta: see traces() (s159)
Saved to runs/diag/viewnet/<KIND>/oc_nomix-<proto>-s<seed>.npz (ocunet's format).
Usage (Leonardo): KIND=emg python studies/13nov/viewnet.py SEED | smoke"""
import importlib.util, os, sys
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); KIND = os.environ["KIND"]
KINDS = ("emg", "scp", "ocuslow", "ocufast", "ocuh1", "ocuh2", "asym", "alpha", "beta", "theta"); assert KIND in KINDS, KIND
_spec = importlib.util.spec_from_file_location("ocunet", WS / "studies/12nov/ocunet.py"); OC = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(OC)
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402


def traces(d):
    X = d["X"].astype(np.float64); ix = {str(c): i for i, c in enumerate(d["ch_names"])}; sel = lambda cs: X[:, [ix[c] for c in cs]].mean(1)  # noqa: E731
    bp = lambda a, lo, hi: sosfiltfilt(butter(4, [lo, hi], btype="band", fs=120, output="sos"), a, axis=-1)  # noqa: E731
    lp = lambda a, f: sosfiltfilt(butter(2, f, fs=120, output="sos"), a, axis=-1)  # noqa: E731
    env = lambda a, lo, hi: lp(np.log(np.abs(bp(a, lo, hi)) + 1e-3), 4)  # noqa: E731
    h1, h2 = sel(("F5", "AF3", "F3")) - sel(("F6", "AF4", "F4")), sel(("FC5", "C5")) - sel(("FC6", "C6"))  # the eye net's ocular derivations
    v1, v2 = sel(("AF3", "AF4")) - sel(("P1", "P2", "Pz")), sel(("AF3", "AF4")) - sel(("C1", "C2", "Cz"))
    ocu = np.stack([h1, h2, v1, v2, np.sqrt(h1 ** 2 + v1 ** 2), np.gradient(h1, axis=-1), np.gradient(v1, axis=-1)], 1)
    if KIND == "emg":
        T = np.stack([env(sel(g), 30, 45) for g in (("TP7",), ("TP8",), ("F5", "F6"), ("AF3", "AF4"), ("C5",), ("C6",))]
                     + [lp(np.log(np.abs(bp(X, 30, 45)).mean(1) + 1e-3), 4)], 1)
    elif KIND == "scp":
        T = np.stack([lp(sel((c,)), 1.0) for c in ("Fz", "FCz", "Cz", "CPz", "Pz", "C3", "C4")], 1)
    elif KIND == "ocuslow":  # the ocular derivations below 1 Hz (slow eye position)
        T = lp(ocu, 1.0)
    elif KIND == "ocufast":  # the ocular derivations at 1-8 Hz (saccades, blinks)
        T = bp(ocu, 1.0, 8.0)
    elif KIND == "ocuh1" or KIND == "ocuh2":  # one half of the window only (the other half zeroed)
        T = ocu.copy(); h = T.shape[-1] // 2; T[..., h:] = 0 if KIND == "ocuh1" else T[..., h:]; T[..., :h] = T[..., :h] if KIND == "ocuh1" else 0
    elif KIND == "asym":  # left-right asymmetries, broadband, smoothed
        T = np.stack([lp(sel((a,)) - sel((b,)), 8) for a, b in (("C3", "C4"), ("FC3", "FC4"), ("CP3", "CP4"), ("F3", "F4"), ("P3", "P4"), ("TP7", "TP8"), ("PO7", "PO8"))], 1)
    elif KIND == "alpha":  # 8-13 Hz envelopes, parietal and occipital
        T = np.stack([env(sel((c,)), 8, 13) for c in ("P3", "P4", "Pz", "PO7", "PO8", "P5", "P6")], 1)
    elif KIND == "beta":  # 13-30 Hz envelopes, sensorimotor
        T = np.stack([env(sel((c,)), 13, 30) for c in ("C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4")], 1)
    elif KIND == "theta":  # 4-8 Hz envelopes, frontal midline
        T = np.stack([env(sel((c,)), 4, 8) for c in ("Fz", "FCz", "Cz", "F1", "F2", "AF3", "AF4")], 1)
    st = d["stream"]
    for s in dict.fromkeys(st):  # causal per-session standardisation, as ocunet.traces
        r = np.flatnonzero(st == s); assert (np.diff(r) > 0).all(); n = np.arange(1, len(r) + 1)[:, None]
        m1 = np.cumsum(T[r].mean(-1), 0) / n; m2 = np.cumsum((T[r] ** 2).mean(-1), 0) / n
        T[r] = (T[r] - m1[:, :, None]) / np.sqrt(np.maximum(m2 - m1 ** 2, 1e-12))[:, :, None]
    assert T.shape[1] == 7 and np.isfinite(T).all(); return np.clip(T, -8, 8).astype(np.float32)


def main():
    d = OC.protocol.load(); T = traces(d); out = WS / "runs/diag/viewnet" / KIND
    if sys.argv[1] == "smoke":
        out = out / "smoke"; out.mkdir(parents=True, exist_ok=True); OC.run("nomix", d, T, "seen_session", 0, out, epochs=2); return print("SMOKE PASS", KIND, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for proto in ("all_loso", "seen_session"):
        OC.run("nomix", d, T, proto, int(sys.argv[1]), out)


if __name__ == "__main__":
    main()
