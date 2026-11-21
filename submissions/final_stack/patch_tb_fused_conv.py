"""Derive a Team B package whose `shallow`/`slow` bodies run one fused conv1d on CUDA (21 Nov profile: the batch-64 prefix
re-embedding of the nine nets was 28 ms/window on the RTX 2080 Ti, 102 of 106 ms of it in these two convs).

usage: patch_tb_fused_conv.py IN.zip IN_SHA256 OUT.zip
Shallow.forward is bn(sc(tc(bandpass(x)))) with tc (1->nf, 1xk, bias) and sc (nf->nf, Cx1, no bias) both linear and nothing in
between, so it equals one conv1d over the C channels with weights M[g,c,k] = sum_f sc[g,f,c] tc[f,k] and constant
sum_{f,c} sc[g,f,c] tc_bias[f] (combined in float64). Only on CUDA; the CPU path is the signed text. Only the Team B encoder text
inside submission.py changes; every other entry is copied byte for byte."""
import ast
import hashlib
import sys
from copy import copy
from pathlib import Path
from zipfile import ZipFile

ANCHORS = [
    ("class Shallow(nn.Module):\n",
     'def fuse_conv(body):\n'
     '    """tc (1 -> nf, 1 x k, bias) and sc (nf -> nf, C x 1) are linear with nothing between them: one conv1d over the C channels."""\n'
     '    w = body.tc.weight.detach().double()[:, 0, 0, :]\n'
     '    s = body.sc.weight.detach().double()[:, :, :, 0]\n'
     '    return torch.einsum("gfc,fk->gck", s, w).float().contiguous(), torch.einsum("gfc,f->g", s, body.tc.bias.detach().double()).float()\n\n\n'
     "class Shallow(nn.Module):\n    fused = None\n\n"),
    ("    def forward(self, x):\n        h = self.bn(self.sc(self.tc(bandpass(x, self.mask)[:, None])))\n",
     "    def forward(self, x):\n"
     "        if self.fused is None:\n"
     "            h = self.bn(self.sc(self.tc(bandpass(x, self.mask)[:, None])))\n"
     "        else:\n"
     "            m, const = self.fused\n"
     "            h = self.bn(F.conv1d(bandpass(x, self.mask), m, const, padding=m.shape[-1] // 2)[:, :, None])\n"),
    ("            self.nets.append(net.to(device).eval())\n",
     "            net = net.to(device).eval()\n"
     "            if torch.device(device).type == \"cuda\" and kind != \"fbc\":\n"
     "                net.body.fused = fuse_conv(net.body)\n"
     "            self.nets.append(net)\n"),
]


def patch(source):
    start = source.index("exec(compile(", source.index("_TEAM_B = {")) + len("exec(compile(")
    stop = source.index(", '<Team B encoder>', 'exec'), _TEAM_B)", start)
    encoder = ast.literal_eval(source[start:stop])
    for old, new in ANCHORS:
        if encoder.count(old) != 1:
            raise ValueError(f"expected exactly one Team B anchor {old[:60]!r}")
        encoder = encoder.replace(old, new)
    compile(encoder, "<Team B encoder>", "exec")
    return source[:start] + repr(encoder) + source[stop:]


def main(src, digest, out):
    src, out = Path(src), Path(out)
    if hashlib.sha256(src.read_bytes()).hexdigest() != digest:
        raise ValueError("input ZIP SHA-256 differs")
    if out.exists():
        raise FileExistsError(out)
    with ZipFile(src) as zin, ZipFile(out, "x") as zout:
        for e in zin.infolist():
            b = zin.read(e.filename)
            if e.filename == "submission.py":
                b = patch(b.decode("utf-8")).encode("utf-8")
                compile(b, "submission.py", "exec")
            zout.writestr(copy(e), b)
    with ZipFile(src) as a, ZipFile(out) as b:
        assert a.namelist() == b.namelist() and b.testzip() is None
        changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
        assert changed == ["submission.py"], changed
    print(hashlib.sha256(out.read_bytes()).hexdigest(), out)


if __name__ == "__main__":
    main(*sys.argv[1:])
