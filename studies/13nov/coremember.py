"""COREMEMBER (ledger s176): core1 as a member (slot c1) streamed in delivery orders. ts_core2.Core with the shipped
config, heads lr0.01 / 0.03 / 0.1 / 0.3; the head that reproduces the stored member in recording order is used (asserted
correlation >= 0.99), then each order of runs/shuffle_perms.npz is streamed through deliver.py. Saves the member per
order keyed by replay row: runs/coremember[_<orders>].npz (rec|proto, <scheme><seed>|proto).
Usage: [ORDERS=within:0,within:1,full:0,runs:0] python studies/13nov/coremember.py"""
import os, sys, time
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/experiments", os.path.dirname(os.path.abspath(__file__))]
import env  # noqa: E402
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
import deliver  # noqa: E402
A = env.A; T0 = time.time(); CFG = os.environ.get("CFG", "car_ov17_fs2_t0.1_le_k2"); MEMBER = os.environ.get("MEMBER", "core1")
HEADS = os.environ.get("HEADS", "lr0.01,lr0.03,lr0.1,lr0.3").split(",")  # s184 Hilbert: CFG=car_fb7_hfs2_t0.1_le_k2 HEADS=lr3 MEMBER=hilbert
ORDERS = [tuple(x.split(":")) for x in os.environ.get("ORDERS", "within:0,within:1,full:0,runs:0").split(",")]
PERMS = np.load(REPO + "/runs/shuffle_perms.npz"); d = protocol.load(); fe = T.Feats(d, T.parse(CFG))
print(f"COREMEMBER: features built ({time.time() - T0:.0f}s); orders {ORDERS}", flush=True)


def stream(proto, heads):
    store = {h: np.full((len(d["y"]), 3), np.nan) for h in heads}; protocol.evaluate(lambda: T.Core(fe, heads, store), d, proto); return store


OUT = {}
for proto in A.PROTOS:
    idx = A.pl.B[f"idx|{proto}"].astype(int); ref = A.C1[proto] if MEMBER == "core1" else 2 * A.L.Z[f"B2_pool|Ps|{proto}"][0] - A.C1[proto]; st = stream(proto, HEADS)
    corr = {h: np.corrcoef(st[h][idx].ravel(), ref.ravel())[0, 1] for h in HEADS}; best = max(corr, key=corr.get)
    print(f"harness {proto}: correlation with the stored {MEMBER} member " + " ".join(f"{h} {c:.4f}" for h, c in corr.items()) + f" -> {best}", flush=True)
    assert corr[best] >= 0.99, f"no head reproduces the stored {MEMBER} member ({corr[best]:.4f})"
    OUT[f"rec|{proto}"] = st[best][idx]
    for sc, sd in ORDERS:
        deliver.install(protocol, deliver.rank_from(PERMS, sc, int(sd), proto))
        try:
            OUT[f"{sc}{sd}|{proto}"] = stream(proto, [best])[best][idx]
        finally:
            deliver.uninstall(protocol)
        print(f"  {proto} {sc}{sd}: mean |change| against recording order {np.abs(OUT[f'{sc}{sd}|{proto}'] - OUT[f'rec|{proto}']).mean():.4f} ({time.time() - T0:.0f}s)", flush=True)
SFX = "" if "ORDERS" not in os.environ else "_" + "_".join(f"{a}{b}" for a, b in ORDERS)  # one file per order set (shufcombo globs them)
np.savez_compressed(REPO + f"/runs/{'coremember' if MEMBER == 'core1' else MEMBER + 'member'}{SFX}.npz", **OUT); print("saved", f"{time.time() - T0:.0f}s")
