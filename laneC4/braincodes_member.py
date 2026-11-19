"""Lane C4: BrainCodes (microsoft/BrainCodes) over the frozen BioCodec RVQ tokenizer as a frozen Lane C member.

Pinned sources, all MIT, fetched on a login node by commit-pinned URL and hash-checked here before any import:
BrainCodes commit b075ec09 (api, backbone, channels; weights/braincodes.pt), BioCodec-release commit 575bf6ea
(biocodec/ model, modules, quantization, utils; biocodec_ckpt.pt). Files were read before use: no network, subprocess, eval or
exec in them; both checkpoints load with `torch.load(weights_only=True)`. The released classes are imported as they are
(no reimplementation); tokenisation runs in fp32 without TF32 (discrete codes must not depend on autocast or batch shape).
Head input: the bottleneck grid [C, T_b, 128] of the released backbone, per README the tap of all reported results.
init=random keeps the pretrained codec (and the codebook-seeded code embeddings) and re-draws every backbone weight.
"""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from common import HeadMember
from oracle import require, sha

HASHES = {
    "braincodes/braincodes.pt": "f173669f2924e425a78d1b89ef4464464bdee70239250d6382e7cd23bb0059c2",
    "braincodes/braincodes/api.py": "32d27a87a80252520b85a2cad8a602b7878dca0957c3f8c9728620f680a281fd",
    "braincodes/braincodes/backbone.py": "19304321a801fc201dbd0f3fce44169dc46d2179a2f69bf315391a1c379aa0e7",
    "braincodes/braincodes/channels.py": "f2f62f557d1e1702ea37b894a2cc1f7d943079b2ec4f718e0d7c7cb78fa55926",
    "braincodes/braincodes/classifier.py": "92e49371f2da39eba64a512ea971a4955752b2bc58806f0dcac2d142c10a6107",
    "braincodes/braincodes/__init__.py": "08bf57c72f9b548c899ac8e56667f2fd800be2036daf7b578453c9732cb9ecba",
    "braincodes/braincodes/preprocess.py": "5f17fe390ccb5f108c9a845d5addf13a10ee44ae85e05522d07e73f68c7d5724",
    "biocodec/biocodec_ckpt.pt": "b8b5c864bd92242a2aa70c414ab1abe23231772dc2f2e39a8f509fdded728af1",
    "biocodec/biocodec/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "biocodec/biocodec/model.py": "5ed36675df8c550d76257891cf2154121997e1b0b787daeab668cc56e2b19834",
    "biocodec/biocodec/modules/conv.py": "e7299ce8edfa2605998048f53699454357cfec23f9b20a79c9faeac511776603",
    "biocodec/biocodec/modules/__init__.py": "a0d9cd2f90234348a1f80e30c25e35c4e2898eaaff66b12df730db2ed8b6be2c",
    "biocodec/biocodec/modules/seanet.py": "3d2cc86aa23be09ad74a4f121a55e273406dd0f437f5e5268bc9f3c4756573f2",
    "biocodec/biocodec/quantization/core_vq.py": "8d4848dfa7541a473041ef51d317efb9c409b5fed7be0636c0480008be4993b4",
    "biocodec/biocodec/quantization/__init__.py": "3a2f24b561dbb7e3c542fcaca3a37ce9435a0aa2f5f05e91c677f79fa63df429",
    "biocodec/biocodec/quantization/quantizer.py": "4b6c772d4a52720c549a24d5f3b5b21ef3939f9356ba6d89fb70471e66a4319f",
    "biocodec/biocodec/utils.py": "a19a2674c9ba32fb5f6dae462b4c2b07e55f8800827536f2e7e7a34f3f8aa291",
}
PROBE_NAMES = ["C3", "C4"]  # montage is rebound per batch through set_coords


class Member(HeadMember):
    def __init__(self, checkpoint, init):
        require(init in ("pretrained", "random"), "init must be pretrained or random")
        root = Path(checkpoint).resolve().parent.parent
        require(Path(checkpoint).resolve() == (root / "braincodes/braincodes.pt"), "checkpoint must be braincodes/braincodes.pt")
        for rel, digest in HASHES.items():
            require(sha(root / rel) == digest, f"{rel} differs from its pinned revision")
        sys.path[:0] = [str(root / "braincodes"), str(root / "biocodec")]
        from braincodes import DEFAULT_ARCH, BrainCodesModel
        from braincodes.backbone import BrainCodes
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        model = BrainCodesModel.from_pretrained(PROBE_NAMES, weights=checkpoint, device="cpu",
                                                codec_ckpt=root / "biocodec/biocodec_ckpt.pt")
        body = model.backbone
        if init == "random":
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            a = {**DEFAULT_ARCH, **ckpt.get("arch", {})}
            body = BrainCodes(tokenizer=body.tokenizer, n_channels=2, d_model=a["d_model"], n_heads=a["n_heads"],
                              n_blocks=a["n_blocks"], n_layers_time=a["n_layers_time"], n_layers_chan=a["n_layers_chan"],
                              dropout=a["dropout"], init_from_codebooks=a["init_from_codebooks"],
                              backbone_width=a["backbone_width"], pool_factors=tuple(a["pool_factors"]))
        super().__init__(body, body.D)
        self.init = init

    def positions(self, names, device):
        from braincodes.channels import coords_from_names
        return torch.from_numpy(coords_from_names(list(names))).reshape(1, len(names), 3).to(device)

    def tokens(self, x, pos):
        x = F.interpolate(x.float(), size=round(x.shape[-1] * 250 / 120), mode="linear", align_corners=False)
        with torch.autocast("cuda", enabled=False):
            x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)  # BrainCodesModel.tokenize
            codes = self.body.tokenizer.tokenize(x)
        self.body.set_coords(pos[0])
        return self.body(codes, already_tokenized=True)["bottleneck"].flatten(1, 2)
