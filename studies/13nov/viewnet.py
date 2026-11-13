"""VIEWNET (13 Nov, ledger s158): view nets on label-free signals the mix does not use, built like the eye net
(studies/12nov/ocunet.py: 7 derived traces per window, causal per-session standardisation, 1-D ResNet with channel
attention, leave one person out and the seen protocol), for SessionEM's view inside the combination (s155: a perfect
view of this kind adds +3.8; mix members as view columns hurt, s157).
  KIND=emg  muscle activity: log 30-45 Hz envelopes at TP7, TP8, F5/F6, AF3/AF4, C5, C6 and the all-channel mean
  KIND=scp  slow cortical potentials: < 1 Hz traces at Fz, FCz, Cz, CPz, Pz, C3, C4
Saved to runs/diag/viewnet/<KIND>/oc_nomix-<proto>-s<seed>.npz (ocunet's format).
Usage (Leonardo): KIND=emg python studies/13nov/viewnet.py SEED | smoke"""
import importlib.util, os, sys
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); KIND = os.environ["KIND"]; assert KIND in ("emg", "scp")
_spec = importlib.util.spec_from_file_location("ocunet", WS / "studies/12nov/ocunet.py"); OC = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(OC)
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402


def traces(d):
    X = d["X"].astype(np.float64); ix = {str(c): i for i, c in enumerate(d["ch_names"])}; sel = lambda cs: X[:, [ix[c] for c in cs]].mean(1)  # noqa: E731
    if KIND == "emg":
        sos = butter(4, [30, 45], btype="band", fs=120, output="sos"); env = lambda a: np.log(np.abs(sosfiltfilt(sos, a, axis=-1)) + 1e-3)  # noqa: E731
        sm = lambda a: sosfiltfilt(butter(2, 4, fs=120, output="sos"), a, axis=-1)  # noqa: E731
        T = np.stack([sm(env(sel((c,)))) for c in ("TP7", "TP8")] + [sm(env(sel(g))) for g in (("F5", "F6"), ("AF3", "AF4"), ("C5",), ("C6",))]
                     + [sm(np.log(np.abs(sosfiltfilt(sos, X, axis=-1)).mean(1) + 1e-3))], 1)
    else:
        sos = butter(2, 1.0, fs=120, output="sos")
        T = np.stack([sosfiltfilt(sos, sel((c,)), axis=-1) for c in ("Fz", "FCz", "Cz", "CPz", "Pz", "C3", "C4")], 1)
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
