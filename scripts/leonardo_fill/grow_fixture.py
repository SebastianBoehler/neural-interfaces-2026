"""Stage signed, fold-excluded graph packages used by the bank growth replay."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from dose import DATA_SHA, V
from grow_sources import team_a_donor
from patch_team_b_bank import COUNTS, MANAS_ARMS, PARENT_SHA, bank_count, bank_for, once, patch_bank, with_manas

LEAD = V.W / "ba-terminal-21nov/zips/final_stack_hard_safe_st_team_b_nine25_tbgraph.zip"
ALT_WEIGHTS = V.W / "laneTB-grow-21nov/b"
SEED2_WEIGHTS = V.W / "finalists-21nov/b_s2"
REFERENCE = "nine25"
WAVES = {
    "growth": (("nine25", "swap25", "fbc12", "ba_six"), tuple(range(71500, 71508)), tuple(range(71508, 71524))),
    "roster": (("nine25", "triseed25", "ba_seven", "ba_eight"), tuple(range(71700, 71716)), tuple(range(71716, 71732))),
}


def stage(root, fold, data, data_path, arms):
    packs = root / "fixtures"
    packs.mkdir(parents=True, exist_ok=True)
    out = packs / fold
    if out.exists():
        raise FileExistsError(out)
    if V.digest(LEAD) != PARENT_SHA or V.digest(data_path) != DATA_SHA:
        raise ValueError("signed lead or PROTEUS data digest differs")
    sources = (V.WEIGHTS, ALT_WEIGHTS, SEED2_WEIGHTS)[:max(bank_count(arm) for arm in arms)]
    origins = [base / fold / "weights.pt" for base in sources]
    audits = [V.audit(data, fold, path) for path in origins]
    if len({info["weights_sha"] for info in audits}) != len(audits):
        raise ValueError("identical Team B seed banks")
    blobs = [path.read_bytes() for path in origins]
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
        for arm in arms:
            blob = bank_for(arm, *blobs)
            text = patch_bank(base, arm, blob)
            pkg = target / arm
            pkg.mkdir()
            for path in hard.iterdir():
                if path.name != "submission.py":
                    (pkg / path.name).symlink_to(path.resolve())
            if arm in MANAS_ARMS:
                text = with_manas(text, donor, partner["weights_sha"], (donor / "LICENSE-MANAS.txt").read_text())
                (pkg / "team_a.pt").symlink_to(donor / "weights.pt")
            (pkg / "submission.py").write_text(text)
            (pkg / "team_b.pt").write_bytes(blob)
            receipts[arm] = dict(data_sha=DATA_SHA, lead_sha=PARENT_SHA, fold=fold, assets=assets,
                                 banks=audits[:bank_count(arm)],
                                 source_sha=V.B.sha(text.encode()), bank_sha=V.B.sha(blob), members=COUNTS[arm],
                                 partner=partner if arm in MANAS_ARMS else None)
        (target / "receipt.json").write_text(json.dumps(receipts, indent=1) + "\n")
        target.rename(out)
    print(f"Prepared {fold}: {', '.join(arms)}", flush=True)


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
