"""Authenticate clean folds and stage the signed fused-conv Solver."""
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("scripts/laneTB", "submissions/final_stack")]
from run import D, DEVICE, stream_session, timed_load
from verify import B, audit, digest, fold_fixture

LEAD = "final_stack_hard_safe_st_team_b_nine25_tbgraph_fc.zip"
LEAD_SHA = "7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07"
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
FOLDS = "production-20261120-32a94ab7/hardcb"
BANKS = "laneTB-21nov/tb"
TAPES = "laneTB-verify-21nov/out/hard"
PINNED_ORDER = 71124
PRECHECK = (71700, 71701)
HELD = (71710, 71711, 71712, 71713)


def read_weights(path):
    with np.load(path, allow_pickle=False) as pack:
        return {key: pack[key] for key in pack.files}


def verify_fold(folder):
    manifest = folder / "pkg.sha256"
    files = (folder / "pkg").iterdir()
    listed = dict(line.strip().split()[::-1] for line in manifest.read_text().splitlines())
    actual = {path.name: digest(path) for path in files}
    if actual != listed:
        raise ValueError(f"fold package SHA-256 mismatch: {folder}")


def replace_once(text, pattern, value):
    text, n = re.subn(pattern, lambda _: value, text, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"expected one fused Solver anchor {pattern!r}")
    return text


def stage(data, fold, inputs, lead, work):
    if digest(lead) != LEAD_SHA:
        raise ValueError("lead archive SHA-256 differs")
    folder = inputs / FOLDS / fold
    verify_fold(folder)
    weights = inputs / BANKS / fold / "weights.pt"
    receipt = audit(data, fold, weights)
    pkg = work / "reference"
    pkg.mkdir()
    with ZipFile(lead) as archive:
        parent = Path(archive.extract("submission.py", work))
    fold_fixture(pkg, parent, folder / "pkg", "hard")
    blob = B.bank(weights, 3)
    source = (pkg / "submission.py").read_text()
    if "def fuse_conv" not in source:
        raise ValueError("fold lost the signed fused-convolution path")
    source = replace_once(source, r"^    TEAM_B_SHA = '[a-f0-9]{64}'$", f"    TEAM_B_SHA = {B.sha(blob)!r}")
    source = replace_once(source, r"^            self\._check_team_b\(dev\)\n", "")
    (pkg / "submission.py").write_text(source)
    (pkg / "team_b.pt").write_bytes(blob)
    receipt.update(manifest_sha=digest(folder / "pkg.sha256"), base_weights_sha=digest(pkg / "weights.npz"),
                   executable_sha=digest(pkg / "submission.py"), lead_sha=LEAD_SHA)
    return pkg, receipt


def enrol(z, replacements):
    """Change only named enrolment slots for names already in the archive."""
    from fit import SLOTS

    names = list(z["route_names"])
    if len(names) != len(set(names)) or set(replacements) - set(names):
        raise ValueError("replacement person is absent or enrolment names are duplicated")
    result = z.copy()
    for name, fitted in replacements.items():
        if set(fitted) != set(SLOTS):
            raise ValueError(f"incomplete replacement for {name}")
        index = names.index(name)
        for key, value in fitted.items():
            if key not in result or result[key].shape[0] != len(names) or value.shape != result[key][index].shape:
                raise ValueError(f"shape mismatch for {name} {key}")
            if result[key] is z[key]:
                result[key] = z[key].copy()
            result[key][index] = value
    return result


def stage_weights(pkg, out, weights):
    out.mkdir()
    for path in pkg.iterdir():
        if path.name != "weights.npz":
            (out / path.name).symlink_to(path.resolve())
    np.savez(out / "weights.npz", **weights)
    return out


def load(pkg, data):
    model, _ = timed_load(pkg, data, DEVICE)
    if ([c.__name__ for c in type(model).__mro__[:2]] != ["TeamBlendStack", "HardWaveEnsemble"]
            or model.routed != 1 or model.cf is None or model.cp is None or not model.tpri
            or model.TEAM_B_MIX != .25 or len(model.team_b.nets) != 9):
        raise ValueError("unexpected signed Solver stages")
    if any(net.body.fused is None for net in model.team_b.nets
           if hasattr(net.body, "fused")) or sum(hasattr(net.body, "fused") for net in model.team_b.nets) != 6:
        raise ValueError("the six Team B fused members did not load")
    return model


def replay(model, signals, rows):
    details = {}

    def check(live, n):
        if (live.cp.dead or live.cf.dead or live.cp.log["err"] or live.cf.log["err"]
                or live.route.log["err"] or live.routeB.log["err"] or len(live.team_b.windows) != n):
            raise ValueError("degraded Solver session")
        if not all(g.graph is not None and not g.failed for g in live.team_b.feature_graphs.values()):
            raise ValueError("Team B CUDA graph path is inactive")
        details.update(routed=live.routeB.log["routed"], own=live.routeB.log["staged"])

    labels, _, meta = stream_session(model, signals, rows, DEVICE, check=check)
    return labels, dict(meta, **details)


def pinned(inputs, fold, arm, stream, rows, weights_sha):
    name = f"{PINNED_ORDER}_{stream.split('|')[1]}"
    tapes = []
    for path in sorted((inputs / TAPES / fold / arm).glob(f"pred_{fold}_*.npz")):
        with np.load(path, allow_pickle=False) as pred:
            if f"rows_{name}" in pred:
                np.testing.assert_array_equal(pred[f"rows_{name}"], rows)
                if str(pred["weights_sha"]) != weights_sha:
                    raise ValueError(f"pinned Team B weights differ for {path}")
                tapes.append(pred[f"labels_{name}"])
    if len(tapes) != 1:
        raise ValueError(f"expected one pinned {fold}/{arm} tape, found {len(tapes)}")
    return tapes[0]
