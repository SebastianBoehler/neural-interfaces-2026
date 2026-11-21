"""Derive a terminal mixture from the signed fused-convolution Team B lead."""
import argparse
from copy import copy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
import sys

HERE = Path(__file__).resolve().parent
FINAL = HERE.parents[1] / "submissions/final_stack"
sys.path.insert(0, str(FINAL))
from build_ba_terminal import DECISION, TERMINAL
from build_cb_speed import patch as graph_cb
from build_hard_team_b import SOLVER, once, verify_parent

LEAD = "final_stack_hard_safe_st_team_b_nine25_tbgraph_fc.zip"
LEAD_SHA = "7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07"
MIXES = {"spectral20": (0.20, 0.0), "cb25": (0.0, 0.25), "joint": (0.15, 0.15)}


def assets_for(arms, spectral, cb):
    use_spectral = any(MIXES[a][0] for a in arms)
    use_cb = any(MIXES[a][1] for a in arms)
    if (use_spectral and spectral is None) or (use_cb and len(cb) != 3):
        raise ValueError("missing terminal weights")
    result = {"spectral.npz": spectral} if use_spectral else {}
    if use_cb:
        if tuple(p.name for p in cb) != ("cb_s0.pt", "cb_s1.pt", "cb_s2.pt"):
            raise ValueError("unexpected CBraMod roster")
        result.update({p.name: p for p in cb})
    return result


def strict_cb_source():
    member = graph_cb((FINAL / "cb_member.py.inc").read_text().rstrip())
    start = member.index("            if self.graph is None and not self.failed:")
    end = member.index("            self.host.copy_(aligned)", start)
    return member[:start] + (
        "            if self.graph is None:\n"
        "                self.graph, self.host, self.static, self.output = self.capture(forward, aligned, dev)\n"
    ) + member[end:]


def patch(source, arms, signatures, record=False):
    if not arms and not record:
        raise ValueError("empty production mixture")
    spectral = signatures.get("spectral.npz")
    cb = tuple((k, v) for k, v in signatures.items() if k.startswith("cb_s"))
    pieces = []
    if spectral:
        pieces.append((HERE / "spectral.py").read_text())
    if cb:
        pieces.append(strict_cb_source())
    text = (HERE / "terminal.py.inc").read_text()
    for key, value in (("MIXES", {a: MIXES[a] for a in arms}), ("SPECTRAL_SHA", spectral),
                       ("CB_FILES", cb), ("RECORD", record)):
        text = once(text, f"__{key}__", repr(value))
    pieces.append(text)
    out = once(source, DECISION, TERMINAL)
    out = once(out, SOLVER, "\n\n".join(pieces) + "\n\n" + SOLVER)
    out = once(out, "model = TeamBlendStack(z,", "model = BreakthroughStack(z,")
    compile(out, "submission.py", "exec")
    return out


def build(parent, arm, assets, out):
    verify_parent(parent, LEAD_SHA)
    payload = {name: path.read_bytes() for name, path in assets.items()}
    signatures = {name: sha256(blob).hexdigest() for name, blob in payload.items()}
    with TemporaryDirectory(dir=out.parent) as temp, ZipFile(parent) as before:
        stage = Path(temp) / out.name
        text = patch(before.read("submission.py").decode(), (arm,), signatures).encode()
        with ZipFile(stage, "x", compression=ZIP_DEFLATED) as after:
            for item in before.infolist():
                after.writestr(copy(item), text if item.filename == "submission.py" else before.read(item))
            for name, blob in payload.items():
                after.writestr(name, blob)
        with ZipFile(stage) as after:
            if after.testzip() or after.namelist() != before.namelist() + list(payload):
                raise ValueError("terminal archive inventory or CRC differs")
            for name in before.namelist():
                if name != "submission.py" and after.read(name) != before.read(name):
                    raise ValueError(f"unchanged archive entry differs: {name}")
            if after.read("submission.py") != text or any(after.read(n) != b for n, b in payload.items()):
                raise ValueError("terminal archive payload differs")
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    print(sha256(out.read_bytes()).hexdigest(), out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path)
    p.add_argument("arm", choices=MIXES)
    p.add_argument("out", type=Path)
    p.add_argument("--spectral", type=Path)
    p.add_argument("--cb", type=Path, nargs=3, default=())
    a = p.parse_args()
    build(a.parent, a.arm, assets_for((a.arm,), a.spectral, a.cb), a.out)


if __name__ == "__main__":
    main()
