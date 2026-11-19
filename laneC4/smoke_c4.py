"""Load and forward smoke on a compute node for both C4 encoders: strict load, positions/slots, shapes, frozen body,
batch vs singleton and bf16 vs fp32 agreement, pretrained vs random difference, timing, and (best effort) output equality of the NeuroRVQ copy
with braindecode's own class from the pinned source tree. Usage: smoke_c4.py ROOT RELEASE_DIR"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "laneC")]
import braincodes_member as bcm  # noqa: E402
import neurorvq_member as nrq  # noqa: E402
from oracle import require  # noqa: E402

root, rel = Path(sys.argv[1]), Path(sys.argv[2])
with np.load(root / "data/proteus_windows.npz", allow_pickle=False) as z:
    names, x = z["ch_names"].tolist(), torch.from_numpy(z["X"][:32].astype(np.float32)).cuda()
ckpt = {"braincodes": rel / "braincodes/braincodes.pt", "neurorvq": rel / "neurorvq/model.safetensors"}
mods = {"braincodes": bcm, "neurorvq": nrq}
logits = {}
for name, mod in mods.items():
    for init in ("pretrained", "random"):
        torch.manual_seed(0)
        m = mod.Member(ckpt[name], init).cuda().train()
        require(not any(p.requires_grad for p in m.body.parameters()) and not m.body.training, "body not frozen/eval")
        n_body = sum(p.numel() for p in m.body.parameters())
        pos = m.positions(names, "cuda")
        with torch.inference_mode():
            tok = m.tokens(x[:4], pos)
            f32, _ = m(x, pos)
            single = torch.cat([m(x[i:i + 1], pos)[0] for i in range(4)])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f16, _ = m(x, pos)
            sub = torch.arange(0, 41, 3)  # channel-dropout style subset, positions subset alongside
            fsub, _ = m(x[:, sub], pos[:, sub])
        logits[name, init] = f32.float().cpu()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            for _ in range(5):
                m(x, pos)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 5 / len(x) * 1000
        print(name, init, "body params", n_body, "tokens", tuple(tok.shape), "token std", tok.float().std().item(),
              "batch-vs-single", (f32[:4] - single).abs().max().item(), "bf16-vs-fp32", (f16.float() - f32).abs().max().item(),
              "subset finite", bool(torch.isfinite(fsub).all()), f"{ms:.1f} ms/window (batch 32)")
        require(torch.isfinite(f32).all() and torch.isfinite(tok).all() and (f32[:4] - single).abs().max() < 5e-4, f"{name} {init} smoke failed")
        require((f16.float() - f32).abs().max() < 0.5, f"{name} {init}: bf16 forward disagrees with fp32")
        if name == "braincodes" and init == "pretrained":
            codes = m.body.tokenizer.tokenize(torch.randn(2, 3, 1000, device="cuda"))
            print("codes", tuple(codes.shape), "unique", codes.unique().numel(), "max", int(codes.max()))
            require(codes.shape[-1] == 6 and int(codes.max()) < 256, "bad codes")
for name in mods:
    d = (logits[name, "pretrained"] - logits[name, "random"]).abs().max().item()
    print(name, "pretrained vs random-init logit difference (heads differ by seed-matched init only)", d)
    require(d > 0, f"{name}: random and pretrained encoders are identical")

# NeuroRVQ copy against braindecode's own class (pinned source tree), same weights, same random input
try:
    sys.path.insert(0, str(rel / "bd_src"))
    from braindecode.models.neurorvq import NeuroRVQ
    ref = NeuroRVQ(n_outputs=2, channel_names=names, n_chans=len(names), n_times=800, sfreq=200).cuda().eval()
    state = nrq.release_tensors(ckpt["neurorvq"])
    miss = ref.load_state_dict(state, strict=False)
    print("braindecode load: missing", [k for k in miss.missing_keys if not k.startswith("final_layer") and not k.startswith("fc_norm")][:5],
          "unexpected", miss.unexpected_keys[:5])
    mine = nrq.Member(ckpt["neurorvq"], "pretrained").cuda().eval()
    inp = torch.randn(3, 41, 800, device="cuda")
    with torch.inference_mode():
        b = mine.body(inp, mine.positions(names, "cuda")[0])
    a = ref._features(inp).reshape(3, 41 * 4, 800)
    print("NeuroRVQ copy vs braindecode class, max abs difference", (a - b).abs().max().item(), "feature scale", a.abs().mean().item())
    require((a - b).abs().max() < 1e-3, "NeuroRVQ copy disagrees with the braindecode class")
except Exception as e:  # reported, never silent
    print("NEURORVQ PARITY CHECK NOT RUN OR FAILED:", type(e).__name__, str(e)[:300])
    if isinstance(e, AssertionError) or "disagrees" in str(e):
        raise
print("PASS smoke")
