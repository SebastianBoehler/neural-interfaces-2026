"""Stage signed, fold-excluded graph packages used by the bank growth replay."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from dose import DATA_SHA, V
from grow_sources import team_a_donor
from patch_team_b_bank import COUNTS, PARENT_SHA, bank_for, once, patch_bank, with_manas

LEAD = V.W / "ba-terminal-21nov/zips/final_stack_hard_safe_st_team_b_nine25_tbgraph.zip"
ALT_WEIGHTS = V.W / "laneTB-grow-21nov/b"
REFERENCE = "nine25"
SCREEN = tuple(range(71500, 71508))
HELD = tuple(range(71508, 71524))
ARMS = tuple(COUNTS)


def stage(root, fold, data, data_path):
    packs = root / "fixtures"
    packs.mkdir(parents=True, exist_ok=True)
    out = packs / fold
    if out.exists():
        raise FileExistsError(out)
    if V.digest(LEAD) != PARENT_SHA or V.digest(data_path) != DATA_SHA:
        raise ValueError("signed lead or PROTEUS data digest differs")
    origins = [base / fold / "weights.pt" for base in (V.WEIGHTS, ALT_WEIGHTS)]
    audits = [V.audit(data, fold, path) for path in origins]
    if audits[0]["weights_sha"] == audits[1]["weights_sha"]:
        raise ValueError("seed-1 bank is identical to seed 0")
    original, extra = [path.read_bytes() for path in origins]
    donor, partner = team_a_donor(fold, data, DATA_SHA)
    with TemporaryDirectory(dir=packs) as temp:
        scratch = Path(temp)
        target = scratch / fold
        target.mkdir()
        with ZipFile(LEAD) as z:
            raw = Path(z.extract("submission.py", scratch))
        hard = scratch / "hard"
        hard.mkdir()
        V.fold_fixture(hard, raw, V.FOLDS / fold / "pkg", "hard")
        base = (hard / "submission.py").read_text()
        assets = {path.name: V.digest(path) for path in hard.iterdir()}
        receipts = {}
        for arm in ARMS:
            blob = bank_for(arm, original, extra)
            text = patch_bank(base, arm, blob)
            pkg = target / arm
            pkg.mkdir()
            for path in hard.iterdir():
                if path.name != "submission.py":
                    (pkg / path.name).symlink_to(path.resolve())
            if arm == "ba_six":
                text = with_manas(text, donor, partner["weights_sha"], (donor / "LICENSE-MANAS.txt").read_text())
                (pkg / "team_a.pt").symlink_to(donor / "weights.pt")
            (pkg / "submission.py").write_text(text)
            (pkg / "team_b.pt").write_bytes(blob)
            banks = audits if arm in ("swap25", "fbc12") else audits[:1]
            receipts[arm] = dict(data_sha=DATA_SHA, lead_sha=PARENT_SHA, fold=fold, assets=assets, banks=banks,
                                 source_sha=V.B.sha(text.encode()), bank_sha=V.B.sha(blob), members=COUNTS[arm],
                                 partner=partner if arm == "ba_six" else None)
        (target / "receipt.json").write_text(json.dumps(receipts, indent=1) + "\n")
        target.rename(out)
    print(f"Prepared {fold}: {', '.join(ARMS)}", flush=True)


def open_fold(root, fold):
    pkg = root / "fixtures" / fold
    receipts = json.loads((pkg / "receipt.json").read_text())
    for arm, receipt in receipts.items():
        if (receipt["lead_sha"] != PARENT_SHA or receipt["data_sha"] != DATA_SHA or receipt["fold"] != fold or
                V.digest(pkg / arm / "submission.py") != receipt["source_sha"] or
                V.digest(pkg / arm / "team_b.pt") != receipt["bank_sha"]):
            raise ValueError(f"{fold} {arm}: staged package differs")
    return pkg, receipts


def prepare_zero(lead, target):
    target.mkdir()
    for path in lead.iterdir():
        if path.name != "submission.py":
            (target / path.name).symlink_to(path.resolve())
    source = (lead / "submission.py").read_text()
    (target / "submission.py").write_text(once(source, "    TEAM_B_MIX = 0.25\n", "    TEAM_B_MIX = 0.0\n"))
