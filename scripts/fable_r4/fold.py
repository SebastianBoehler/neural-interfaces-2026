"""Pin the fused executable and use the existing cap250 and Team B fold builders."""
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend(str(ROOT / path) for path in ("scripts/fable_round2", "scripts/laneTB", "cleanslate/harness"))
from folds import DATA_SHA, FOLDS, ORDERS, pinned, streams
from grow_sources import once
from verify import B, D, DEVICE, FOLDS as PRODUCTION, W, WEIGHTS, audit, digest, fold_fixture, timed_load

LEAD = W / "fc-qual-21nov/zips/final_stack_hard_safe_st_team_b_nine25_tbgraph_fc.zip"
LEAD_SHA = "7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07"
ORACLE = tuple(range(71700, 71704))
PSEUDO = tuple(range(71704, 71708))


def stage(archive, data, fold, work):
    if digest(archive) != LEAD_SHA:
        raise ValueError("fused lead SHA-256 differs")
    weights = WEIGHTS / fold / "weights.pt"
    bank_receipt = audit(data, fold, weights)
    blob = B.bank(weights, 3)
    with ZipFile(archive) as z:
        original = Path(z.extract("submission.py", work))
        parent_bank = B.sha(z.read("team_b.pt"))
    pkg = work / "fold"
    pkg.mkdir()
    fold_fixture(pkg, original, PRODUCTION / fold / "pkg", "hard")
    source = pkg / "submission.py"
    text = once(source.read_text(), "            self._check_team_b(dev)\n", "")
    text = once(text, repr(parent_bank), repr(B.sha(blob)))
    source.write_text(text)
    (pkg / "team_b.pt").write_bytes(blob)
    model, _ = timed_load(pkg, data, DEVICE)
    if type(model).__name__ != "TeamBlendStack" or model.TEAM_B_MIX != .25 or model.perm is not None:
        raise ValueError("unexpected lead class, Team B weight or channel mapping")
    return model, dict(fold=fold, source_sha=digest(source), weights_sha=digest(pkg / "weights.npz"),
                       lead_sha=LEAD_SHA, data_sha=DATA_SHA, bank_sha=B.sha(blob),
                       bank_source_sha=bank_receipt["weights_sha"])
