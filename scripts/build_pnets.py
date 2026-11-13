"""Personal SessionBN nets for the package (ledger s96): every shipped SessionBN net (runs/sbn/<cfg>-s<seed>.pt, trained on
all public windows) is fine-tuned for each of the ten public people on all of that person's public windows (pf60: all
parameters, 60 steps, lr 5e-4; pfh60: SessionBN affine + output layer, 60 steps, lr 2e-3), as studies/11nov/ttt/galv/pft.py.
Saves runs/pnets/<cfg>-s<seed>-<person>-<variant>.pt (state dict + losses). Export to TorchScript is a later step.
Usage (Galvani): python scripts/build_pnets.py TASK   (TASK 0-23: cfg = CFGS[TASK // 6], seed = TASK % 6) | smoke
BASE_PKG=<package dir> (review 13 Nov, item 5): the base is the package's own shipped trace, rebuilt exactly as a dev net
(scripts/sbn_trace_base.py), not runs/sbn; pfh60 only; saved to runs/pnets_pkg with the trace's SHA-256 as "ckpt"."""
import importlib.util, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
import os  # noqa: E402
from export_sbn import trained  # noqa: E402
from sbn_trace_base import base as pkg_base  # noqa: E402
spec = importlib.util.spec_from_file_location("pft", ROOT / "studies/11nov/ttt/galv/pft.py"); PFT = importlib.util.module_from_spec(spec); spec.loader.exec_module(PFT)
BASE_PKG = os.environ.get("BASE_PKG")
CFGS = ["fb_sbn_rot05", "fbcnet_raw_sbn", "fblight_ea_rot05_sbn", "sincshallow_ea_ptm7r_rot05_sbn"]
VARS = {"pf60": (60, 5e-4, "all"), "pfh60": (60, 2e-3, "head")}; OUT = ROOT / "runs/pnets"
if BASE_PKG:
    VARS, OUT = {"pfh60": VARS["pfh60"]}, ROOT / "runs/pnets_pkg"


def main():
    d = protocol.load(); smoke = sys.argv[1] == "smoke"; task = 0 if smoke else int(sys.argv[1]); cfg, seed = CFGS[task // 6], task % 6
    OUT.mkdir(parents=True, exist_ok=True); m, info = pkg_base(BASE_PKG, cfg, seed, d) if BASE_PKG else trained(cfg, d, seed, ROOT / "runs/sbn"); m.dev = "cuda" if torch.cuda.is_available() else "cpu"; m.net.to(m.dev)
    per = np.array([s[-3:] for s in d["subject"]]); pub = d["split"] != "external"; people = sorted(set(per[pub]))
    assert len(people) == 10, people
    for q in people[:1] if smoke else people:
        rows = np.flatnonzero(pub & (per == q))
        for v, (steps, lr, scope) in VARS.items():
            f = OUT / f"{cfg}-s{seed}-{q}-{v}.pt"
            if f.exists() and not smoke:
                continue
            t0 = time.time(); net, l0, l1 = PFT.personalise(m, rows, d["y"], 2 if smoke else steps, lr, scope, seed)
            assert smoke or l1 < l0, f"{cfg} s{seed} {q} {v}: loss did not fall ({l0:.4f} -> {l1:.4f})"
            sd = {k: t.detach().cpu() for k, t in net.state_dict().items()}; base = m.net.state_dict()
            changed = sum(int(not torch.equal(sd[k], base[k].cpu())) for k in sd); assert changed > 0, "no parameter changed"
            torch.save({"state": sd, "cfg": cfg, "seed": seed, "person": q, "variant": v, "loss0": l0, "loss1": l1, "rows": len(rows), "ckpt": info["ckpt_sha256"]}, f)
            print(f"{cfg} s{seed} {q} {v}: {len(rows)} rows, loss {l0:.3f} -> {l1:.3f}, {changed} tensors changed, {time.time() - t0:.0f}s", flush=True)
    print("SMOKE OK" if smoke else f"DONE {cfg} s{seed}", flush=True)


if __name__ == "__main__":
    main()
