"""Make signed fused-lead fold copies without changing fitted assets."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from fit import DATA_SHA, roster
from patch_terminal import LEAD, LEAD_SHA, assets_for, patch, once, verify_parent
import verify as V

TEAM_B_PIN = V.W / "laneTB-verify-21nov/out/hard"
SCREEN = tuple(range(71600, 71608))
HELD = tuple(range(71608, 71624))
ORDER_PIN = 71124


def visits(data, fold):
    mode, person = fold.split("_")
    sessions = ("2", "3") if mode == "ret" else sorted(set(data["session"][data["person"] == person]))
    return [f"{person}|{visit}" for visit in sessions]


def bank_inputs(run, data, fold, pkg, arms):
    # The immutable core package is linked below; all independent banks use its fold exclusion.
    spectral = run / "fits" / f"{fold}.npz"
    weights = V.WEIGHTS / fold / "weights.pt"
    team_b = V.audit(data, fold, weights)
    files = assets_for(arms, spectral, tuple(sorted(pkg.glob("cb_s*.pt"))))
    signatures = {name: V.digest(path) for name, path in files.items()}
    if "spectral.npz" in files:
        rows, people, first = roster(fold, data)
        expected = dict(fold=fold, people=people, first_only=first, streams=sorted(set(data["stream"][rows])),
                        rows=len(rows), spectral_sha=signatures["spectral.npz"], data_sha=DATA_SHA)
        if json.loads(spectral.with_suffix(".json").read_text()) != expected:
            raise ValueError(f"{fold}: spectral training roster or digest differs")
    return V.B.bank(weights, 3), team_b, files, signatures


def stage(run, data, fold, out, arms, *, observe):
    signed = run / "zips" / LEAD
    verify_parent(signed, LEAD_SHA)
    pkg = V.FOLDS / fold / "pkg"
    if len(tuple(pkg.glob("cb_s*.pt"))) != 3 or not (pkg / "weights.npz").is_file():
        raise ValueError(f"incomplete production fold {fold}: {pkg}")
    bank, audit, files, hashes = bank_inputs(run, data, fold, pkg, arms)
    with TemporaryDirectory(dir=out.parent) as temp, ZipFile(signed) as z:
        raw = Path(z.extract("submission.py", temp))
        out.mkdir()
        V.fold_fixture(out, raw, pkg, "hard")
        fixture = out / "submission.py"
        text = fixture.read_text()
        text = once(text, "            self._check_team_b(dev)\n", "")
        old = z.read("team_b.pt")
        text = once(text, repr(V.B.sha(old)), repr(V.B.sha(bank)))
        if observe:
            text = patch(text, arms, hashes, record=True)
        fixture.write_text(text)
        (out / "team_b.pt").write_bytes(bank)
        for name, path in files.items():
            (out / name).symlink_to(path.resolve())
    return dict(fold=fold, lead_sha=LEAD_SHA, data_sha=DATA_SHA, team_b_audit=audit,
                team_b_bank=V.B.sha(bank), experts=hashes, assets={p.name: V.digest(p) for p in out.iterdir()},
                arms=list(arms), observe=observe)
