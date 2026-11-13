import sys, numpy as np, torch
sys.dont_write_bytecode = True
from pathlib import Path
PKG = Path("/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026/runs/sealed/cand_v52b_pnets")
z = np.load(PKG / "weights.npz")
names = [str(n) for n in z["route_names"]]; print("route_names", names)
idx = z["pnet_member"]; files = z["pnet_files"]; w = z["member_weights"].astype(float); w = w / w.sum()
print("n_nets", int(z["n_nets"]), "pnet members", idx.tolist()); print("sum of normalised weight on the 30 SBN members %.4f; core1 weight %.4f; per-member" % (w[1 + idx].sum(), w[0]), np.round(w[1 + idx], 4).tolist()[:6], "...")
print("keys own_beta", float(z["own_beta"]), "bio_tau", float(z["bio_tau"]), "bio_t0", int(z["bio_t0"]), "route_tau", float(z["route_tau"]), "route_t_start", int(z["route_t_start"]), "route_ramp", float(z["route_ramp"]), "bio_ramp", float(z["bio_ramp"]))
seen = {}
for m, i in enumerate(idx):
    base = str(z[f"net{i}_sjit_file"]); fam = base.rsplit("_s", 1)[0]
    sb = torch.jit.load(str(PKG / base), map_location="cpu").state_dict()
    rows = []
    for j in (0, 3, 9):
        sp = torch.jit.load(str(PKG / str(files[m, j])), map_location="cpu").state_dict()
        assert set(sp) == set(sb), (base, set(sp) ^ set(sb))
        diff = {k: float((sp[k].double() - sb[k].double()).abs().max()) for k in sb if sb[k].shape == sp[k].shape}
        rows.append({k: v for k, v in diff.items() if v > 0})
    if fam not in seen:
        seen[fam] = 1
        print(base, "tensors:", len(sb))
        for j, r in zip((0, 3, 9), rows):
            print("   ", names[j], {k: float(f"{v:.3g}") for k, v in r.items()})
        # requires_grad of parameters after jit.load (relevant to fine-tuning a trace)
        mod = torch.jit.load(str(PKG / base), map_location="cpu")
        print("    parameters (requires_grad):", [(k, p.requires_grad) for k, p in mod.named_parameters()][:30])
        print("    buffers:", [k for k, _ in mod.named_buffers()])
