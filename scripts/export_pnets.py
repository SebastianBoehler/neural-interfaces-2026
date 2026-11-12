"""v5.2 preparation (ledger s96, s102): trace each person's fine-tuned SessionBN nets (runs/pnets, variant pfh60) as
stateful TorchScript members in the package's form (session_bn.to_stream at the package's k, as scripts/export_sbn.py).
Checks per trace: traced stream output equals the eager personal net on the person's own public sessions (< 1e-4).
Usage (Leonardo/Galvani GPU): python scripts/export_pnets.py TASK   (TASK 0-23: cfg = CFGS[TASK // 6], seed = TASK % 6)
Writes runs/pnets_jit/p_<prefix>_k<k>_s<seed>_<person>.pt and a JSON line per trace."""
import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
from export_external_teacher import device_free_trace  # noqa: E402
from export_sbn import trained  # noqa: E402
from session_bn import reset_stream, to_stream  # noqa: E402
# cfg -> (package file prefix, package k values)
CFGS = [("fb_sbn_rot05", "sbn", (2.0, 5.0)), ("fbcnet_raw_sbn", "sbn_fbcnet_raw", (20.0,)),
        ("fblight_ea_rot05_sbn", "sbn_fblight_ea_rot05", (5.0,)), ("sincshallow_ea_ptm7r_rot05_sbn", "sbn_sincshallow_ea_ptm7r_rot05", (20.0,))]
OUT = ROOT / "runs/pnets_jit"


def main():
    task = int(sys.argv[1]); cfg, pre, ks = CFGS[task // 6]; seed = task % 6; d = protocol.load(); OUT.mkdir(parents=True, exist_ok=True)
    per = np.array([s[-3:] for s in d["subject"]]); pub = d["split"] != "external"; st = d["stream"]
    m, info = trained(cfg, d, seed, ROOT / "runs/sbn"); base_sd = {k: v.clone() for k, v in m.net.state_dict().items()}
    for q in sorted(set(per[pub])):
        ck = torch.load(ROOT / f"runs/pnets/{cfg}-s{seed}-{q}-pfh60.pt", map_location="cpu", weights_only=False)
        assert ck["ckpt"] == info["ckpt_sha256"], f"{cfg} s{seed} {q}: personal net was fine-tuned from another base checkpoint"
        m.net.load_state_dict(ck["state"]); m.net.eval()
        rows = np.flatnonzero(pub & (per == q)); first = m._aligned(rows[st[rows] == st[rows][0]][:4], True)
        for k in ks:
            m.set_k(k); ref = protocol.predict_streams(m, d, pub & (per == q), proba=True)
            name = f"p_{pre}_k{k:g}_s{seed}_{q}.pt"; tr = device_free_trace(to_stream(m.net, k), torch.from_numpy(np.ascontiguousarray(first, dtype=np.float32))); reset_stream(tr); tr.save(str(OUT / name))
            mod = torch.jit.load(str(OUT / name)); diff = 0.0
            with torch.no_grad():
                for s in dict.fromkeys(st[rows]):
                    r = rows[st[rows] == s]; reset_stream(mod); X = m._aligned(r, True)
                    P = torch.softmax(mod(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))).double(), 1).numpy()
                    diff = max(diff, float(np.abs(P - ref[np.isin(rows, r)]).max()))
            print(json.dumps({"trace": name, "max_abs_prob_vs_eager": diff, "sessions": len(set(st[rows]))}), flush=True)
            assert diff < 1e-4, f"{name}: traced output differs from the eager personal net by {diff}"
        m.net.load_state_dict(base_sd)
    print(f"DONE {cfg} s{seed}", flush=True)


if __name__ == "__main__":
    main()
