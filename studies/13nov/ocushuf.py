"""OCUSHUF (13 Nov, ledger s167): the eye net (studies/12nov/ocunet.py, arm nomix) with its causal per-session trace
standardisation done in a non-chronological delivery order (runs/shuffle_perms.npz, the orders of shuffle.py), for
training and test sessions alike; everything else as ocunet. Saved to runs/diag/ocushuf/<SCHEME><PSEED>/oc_nomix-...
keyed by window index. Usage (Leonardo): SCHEME=full PSEED=0 python studies/13nov/ocushuf.py SEED | smoke"""
import hashlib, importlib.util, os, sys
from pathlib import Path
import numpy as np
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); SCHEME = os.environ["SCHEME"]; PSEED = int(os.environ["PSEED"])
_spec = importlib.util.spec_from_file_location("ocunet", WS / "studies/12nov/ocunet.py"); OC = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(OC)
PERMS = np.load(WS / "runs/shuffle_perms.npz")
# s214 (bugs_scoring P0-a): the delivery order is the PROTOCOL's own. It used to be all_loso's for seen_session too, while the
# stages used seen_session's permutation. set_order(proto) sets it; every output stores the hash of the order it used.
ORDER = PERMS[f"{SCHEME}|{PSEED}|all_loso"]; RANK = np.full(int(ORDER.max()) + 1, -1); RANK[ORDER] = np.arange(len(ORDER))


def set_order(proto):
    global ORDER
    ORDER = PERMS[f"{SCHEME}|{PSEED}|{proto}"]; RANK[:] = -1; RANK[ORDER] = np.arange(len(ORDER))
    return hashlib.sha256(np.ascontiguousarray(ORDER.astype(np.int64)).tobytes()).hexdigest()


def traces(d):
    X = d["X"].astype(np.float64); ix = {str(c): i for i, c in enumerate(d["ch_names"])}; sel = lambda cs: X[:, [ix[c] for c in cs]].mean(1)  # noqa: E731
    sg = lambda a: OC.savgol_filter(a, 25, 3, axis=-1)  # noqa: E731
    h1, h2 = sg(sel(("F5", "AF3", "F3")) - sel(("F6", "AF4", "F4"))), sg(sel(("FC5", "C5")) - sel(("FC6", "C6")))
    v1, v2 = sg(sel(("AF3", "AF4")) - sel(("P1", "P2", "Pz"))), sg(sel(("AF3", "AF4")) - sel(("C1", "C2", "Cz")))
    T = np.stack([h1, h2, v1, v2, np.sqrt(h1 ** 2 + v1 ** 2), np.gradient(h1, axis=-1), np.gradient(v1, axis=-1)], 1)
    st = d["stream"]; A0 = PERMS[f"{SCHEME}|{PSEED}|all_loso"]; rk = np.full(len(st), -1); rk[A0] = np.arange(len(A0))  # all_loso's
    assert (rk >= 0).all(), "the all_loso order does not cover every window"
    cov = ORDER[ORDER < len(st)]; rk[cov] = RANK[cov] + len(st)  # the protocol's own order wins on its (test) sessions
    for s in dict.fromkeys(st):
        r = np.flatnonzero(st == s); r = r[np.argsort(rk[r], kind="stable")]; n = np.arange(1, len(r) + 1)[:, None]  # delivery order
        m1 = np.cumsum(T[r].mean(-1), 0) / n; m2 = np.cumsum((T[r] ** 2).mean(-1), 0) / n
        T[r] = (T[r] - m1[:, :, None]) / np.sqrt(np.maximum(m2 - m1 ** 2, 1e-12))[:, :, None]
    return np.clip(T, -8, 8).astype(np.float32)


def main():  # PROTOS=seen_session re-streams only that protocol; OUT_SUB names the output folder (default ocushuf)
    d = OC.protocol.load(); out = WS / "runs/diag" / os.environ.get("OUT_SUB", "ocushuf") / f"{SCHEME}{PSEED}"
    protos = os.environ.get("PROTOS", "all_loso,seen_session").split(",")
    if sys.argv[1] == "smoke":
        out = out / "smoke"; protos, seed, ep = ["seen_session"], 0, 2
    else:
        seed, ep = int(sys.argv[1]), 60
    out.mkdir(parents=True, exist_ok=True)
    for proto in protos:
        sha = set_order(proto); T = traces(d); OC.run("nomix", d, T, proto, seed, out, epochs=ep)
        f = out / f"oc_nomix-{proto}-s{seed}.npz"; z = dict(np.load(f))
        if "order_sha" not in z:
            np.savez(f, **z, order_sha=np.array(sha), order_proto=np.array(proto))
    if sys.argv[1] == "smoke":
        print("SMOKE PASS", flush=True)


if __name__ == "__main__":
    main()
