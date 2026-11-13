"""TEACHSHUF (ledger s184): the teacher member (slot 10, weight 5) with its causal EA in a delivery order.
Teacher = LogisticRegression(C 0.01, balanced) per left-out person (all_loso) or per fold (seen_session) on the seed-mean
head log-probs of eegnex_ea-m7r and ifnet_ea-m7r (runs/pretrain_a100all, seeds 0-11) applied to causally EA-aligned
windows (scripts/diag/screen_external_heads.py causal_ea, k 5, G = mean PROTEUS covariance; runs/archive_7nov/own/
pe_metric.py teacher()). Package-faithful: the map is fitted on the training people's recording-order features and
applied to the test rows' features aligned in the delivery order. The recording-order rebuild must equal the stored slot 10.
Saves runs/teachshuf.npz: rec|proto and <scheme><seed>|proto, keyed by replay row.
Usage: [ORDERS=within:0,...] python studies/13nov/teachshuf.py"""
import importlib.util, json, os, sys, time
import numpy as np
os.environ["NI26_C"] = "0.01"  # pe_metric.py sets this before importing the teacher map
REPO = os.path.abspath(__file__ + "/../../.."); T0 = time.time()
sys.path[:0] = [REPO + "/replay", REPO + "/experiments", REPO + "/scripts", REPO + "/scripts/diag"]
import torch  # noqa: E402
import env  # noqa: E402
import protocol  # noqa: E402
import eval_external_teacher as E  # noqa: E402
_s = importlib.util.spec_from_file_location("seh", REPO + "/scripts/diag/screen_external_heads.py"); SEH = importlib.util.module_from_spec(_s); _s.loader.exec_module(SEH)
A = env.A; d = SEH.d; st, subj = d["stream"], d["subject"]; DEV = "cuda" if torch.cuda.is_available() else "cpu"
ORDERS = os.environ.get("ORDERS", "within:0,within:1,full:0,runs:0").split(",")
PERMS = np.load(REPO + "/runs/shuffle_perms.npz"); FAMS = ("eegnex_ea-m7r", "ifnet_ea-m7r"); SEEDS = range(12)
C = np.concatenate([SEH.covariances(d["X"][i:i + 512]) for i in range(0, len(d["X"]), 512)]); G = C.mean(0)


def ea(order):
    """causal EA with each session's windows in the given order (global indices); rows outside the order stay 0."""
    out = np.zeros(d["X"].shape, np.float32)
    for s in dict.fromkeys(st[order]):
        i = order[st[order] == s]; n = np.arange(1, len(i) + 1)[:, None, None]
        P = SEH.invsqrtm((5.0 * G + np.cumsum(C[i], 0)) / (5.0 + n)); out[i] = np.einsum("nij,njt->nit", P, d["X"][i].astype(np.float64))
    return out


NETS = {}


def feats(X):
    """seed-mean head log-probs per family, concatenated (screen_external_heads.heads on the device)."""
    F = []
    for f in FAMS:
        acc = 0
        for s in SEEDS:
            p = f"{REPO}/runs/pretrain_a100all/{f}-s{s}.pt"
            if p not in NETS:
                ck = torch.load(p, weights_only=False, map_location="cpu"); meta = json.loads(ck["meta"]); assert meta.get("align", "ea") == "ea"
                cls = getattr(SEH.bd, SEH.ARCHS[os.path.basename(p).split("_")[0]]); blocks = {(m["offset"], m.get("n", len(m["classes"]))): c for c, m in meta["corpora"].items()}
                if meta.get("unified"):
                    blocks = {(0, 3): "unified"}
                kw = dict(n_chans=41, n_outputs=max(o + n for o, n in blocks), n_times=480)
                if "sfreq" in SEH.inspect.signature(cls).parameters:
                    kw["sfreq"] = 120.0
                net = cls(**kw); net.load_state_dict(ck["state"]); NETS[p] = (net.eval().to(DEV), sorted(blocks))
            net, blocks = NETS[p]
            with torch.no_grad():
                Zt = torch.cat([net(torch.from_numpy(X[i:i + 256]).to(DEV)).cpu() for i in range(0, len(X), 256)]).double()
            acc = acc + np.concatenate([torch.log_softmax(Zt[:, o:o + n], 1).numpy() for o, n in blocks], 1)
        F.append(acc / len(SEEDS))
    return np.concatenate(F, 1)


def teacher(Frec, Fte, proto):
    T = np.zeros((len(st), 3))
    if proto == "all_loso":
        for p in sorted(set(subj)):
            te = subj == p; Fc = Frec.copy(); Fc[te] = Fte[te]; T[te] = E.fit_predict([Fc], ~te, te)
    else:
        for _, tr, te in protocol.folds(d, proto):
            Fc = Frec.copy(); Fc[te] = Fte[te]; T[te] = E.fit_predict([Fc], tr, te)
    return T


Frec = feats(SEH.causal_ea(d["X"])); OUT = {}
print(f"TEACHSHUF: recording-order features {Frec.shape} on {DEV} ({time.time() - T0:.0f}s)", flush=True)
for proto in A.PROTOS:
    idx = A.L.Z[f"idx|{proto}"].astype(int); T = teacher(Frec, Frec, proto)[idx]; ref = A.L.Z[f"B2_pool|Ps|{proto}"][10]
    print(f"harness {proto}: rebuilt recording-order teacher against the stored slot 10, max |diff| {np.abs(T - ref).max():.2e}", flush=True)
    assert np.abs(T - ref).max() < 1e-4, "the rebuilt teacher is not the stored slot 10"
    OUT[f"rec|{proto}"] = T
    for o in ORDERS:
        sc, sd = o.split(":"); Fo = feats(ea(PERMS[f"{sc}|{sd}|{proto}"])); To = teacher(Frec, Fo, proto)[idx]; OUT[f"{sc}{sd}|{proto}"] = To
        print(f"  {proto} {sc}{sd}: mean |change| against recording order {np.abs(To - T).mean():.4f}, labels changed {np.mean(To.argmax(1) != T.argmax(1)):.3f} ({time.time() - T0:.0f}s)", flush=True)
SFX = "" if "ORDERS" not in os.environ else "_" + "_".join(o.replace(":", "") for o in ORDERS)
np.savez_compressed(f"{REPO}/runs/teachshuf{SFX}.npz", **OUT); print("saved", f"{time.time() - T0:.0f}s")
