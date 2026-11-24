"""Lane I returning arm: raw SessionBN traces with another baked prior count k.

python patch_k.py SRC_PKG_DIR OUT_ROOT WHO    OUT_ROOT/WHO_k{20,10,5}: a copy of the fold package whose six `sbn_fbcnet_raw_k20_s*.pt`
have the StreamBN constant k rewritten (20 -> tau); every other file is a symlink or copy as in the source. k20 must reproduce the source
traces bitwise (checked by load + forward on a probe, and the zip bytes differ only in the one constant)."""
import shutil
import struct
import sys
import zipfile
from pathlib import Path

import torch

TAUS = (20, 10, 5)


def patch(src, dst, tau):
    zin = zipfile.ZipFile(src)
    names = [i.filename for i in zin.infolist() if "/constants/" in i.filename]  # archive prefix is not always the file stem
    hits = [n for n in names if len(zin.read(n)) == 8 and struct.unpack("<d", zin.read(n))[0] == 20.0]
    assert len(hits) == 1, hits  # exactly one 20.0 constant: the prior count of the one StreamBN
    with zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == hits[0]:
                data = struct.pack("<d", float(tau))
            zout.writestr(info, data, compress_type=info.compress_type)


def main(src_pkg, out_root, who):
    src_pkg = Path(src_pkg)
    for tau in TAUS:
        out = Path(out_root) / f"{who}_k{tau}"
        out.mkdir(parents=True)
        for f in sorted(src_pkg.iterdir()):
            if f.name.startswith("sbn_fbcnet_raw_k20_s") and f.suffix == ".pt":
                patch(f.resolve(), out / f.name, tau)
            elif f.name == "submission.py":
                shutil.copy2(f, out / f.name)
            else:
                (out / f.name).symlink_to(f.resolve())
        # probe: patched trace vs original on one input stream, window by window
        a, b = torch.jit.load(str(src_pkg / "sbn_fbcnet_raw_k20_s0.pt")), torch.jit.load(str(out / "sbn_fbcnet_raw_k20_s0.pt"))
        bn = [dict(m.named_modules())["spatial_conv.1"] for m in (a, b)]
        z = torch.randn(40, int(bn[0].rm.shape[0]), 1, 120)
        for m in (a, b):
            for n, buf in m.named_buffers():
                if n.rsplit(".", 1)[-1] in ("s1", "s2", "n"):
                    buf.zero_()
        ya, yb = bn[0](z), bn[1](z)
        same = bool(torch.equal(ya, yb))
        assert same == (tau == 20), (tau, same)
        print("OK", out.name, "bitwise_equal_to_source", same, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
