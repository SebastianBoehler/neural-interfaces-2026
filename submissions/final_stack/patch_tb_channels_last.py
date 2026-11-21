"""Derive a Team B package whose conv bodies run channels_last on pre-Ampere GPUs (21 Nov profile: the NCHW cuDNN path of the
`shallow` and `slow` bodies is 3x slower on the RTX 2080 Ti; A100 neutral, so Ampere and newer keep the exact signed path).

usage: patch_tb_channels_last.py IN.zip IN_SHA256 OUT.zip
Only the Team B encoder text inside submission.py changes (one anchor in Encoder.__init__); every other entry is byte for byte."""
import ast
import hashlib
import sys
from copy import copy
from pathlib import Path
from zipfile import ZipFile

MARKER = "_TEAM_B = {"
OLD = "            self.nets.append(net.to(device).eval())\n"
NEW = ("            net = net.to(device).eval()\n"
       "            if torch.device(device).type == \"cuda\" and torch.cuda.get_device_capability(device) < (8, 0) and kind != \"fbc\":\n"
       "                net.to(memory_format=torch.channels_last)  # Turing: NCHW cuDNN convs of these bodies are 3x slower\n"
       "            self.nets.append(net)\n")


def patch(source):
    start = source.index("exec(compile(", source.index(MARKER)) + len("exec(compile(")
    stop = source.index(", '<Team B encoder>', 'exec'), _TEAM_B)", start)
    encoder = ast.literal_eval(source[start:stop])
    if encoder.count(OLD) != 1:
        raise ValueError("expected exactly one Team B net-loading anchor")
    encoder = encoder.replace(OLD, NEW)
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
