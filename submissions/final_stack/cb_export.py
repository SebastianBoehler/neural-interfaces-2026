"""Lane C2 CBraMod-C fit (.pt: model, prior, channels) -> compact member blob for CBMember (cb_member.py.inc)."""
import zlib


def export_blob(ckpt, half=False):
    """Leonardo Lane C2 fit (.pt: model, prior, channels) -> compact member blob for CBMember."""
    state, names = ckpt["model"], ckpt["channels"]
    if len(names) != 41:
        raise ValueError("expected the 41-channel PROTEUS montage")
    tag = zlib.crc32(",".join(names).encode()) % (1 << 20)
    w, b = state[f"heads.h{tag}.task.weight"], state[f"heads.h{tag}.task.bias"]
    if w.shape != (4, 41 * 200):
        raise ValueError("unexpected PROTEUS head")
    cast = (lambda t: t.half()) if half else (lambda t: t.float())
    return dict(net={k[len("body.net."):]: cast(v) for k, v in state.items() if k.startswith("body.net.")},
                head_w=cast(w[:3]), head_b=cast(b[:3]), prior=ckpt["prior"].double(), tag=tag)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import torch

    p = argparse.ArgumentParser(description="export reve_full_C_{seed}_{HOLD}.pt (seeds 0-2) as cb_s{seed}.pt")
    p.add_argument("ckpts", type=Path); p.add_argument("out", type=Path); p.add_argument("hold")
    p.add_argument("--fp32", action="store_true", help="keep float32 (default: float16 storage, loaded as float32)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for seed in range(3):
        ckpt = torch.load(a.ckpts / f"reve_full_C_{seed}_{a.hold}.pt", map_location="cpu", weights_only=True)
        torch.save(export_blob(ckpt, half=not a.fp32), a.out / f"cb_s{seed}.pt")
        print("exported", seed, (a.out / f"cb_s{seed}.pt").stat().st_size)
