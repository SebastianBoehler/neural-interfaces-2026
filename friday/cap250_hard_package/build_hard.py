"""Build the deterministic self-contained hard-only trial from signed sources."""

import argparse
from copy import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


HERE = Path(__file__).resolve().parent
STOCK = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
CODE = "37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f"
LOCK = "0292022abd39067cefebd0e3a0bd8f1834a04bfc67d7d41081cfe6abaae8ebd2"
ACTORS = (
    "ec33684850e902479f3bf28e472a9ea590e64fa759b0b03704daf7aff1ea4018",
    "ed83b6b5b2a2765f82afa4a2fcb3f2c526d24a32e8b789b527268bde67b5401e",
)
MARKER = "class Solver(CompetSolver):"
FACTORY = 'model = Ensemble6(z, str(meta.get("device", "cpu")), meta["submission_dir"])'


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def actor_assets(pupils, lock):
    if sha(lock) != LOCK:
        raise ValueError("F32x actor lock differs")
    content = json.loads(lock.read_text())
    if tuple(content["actors"]["hard"]) != ACTORS or content["device"] != "cpu" or not content["f32e_source_only"]:
        raise ValueError("expected exact source-qualified CPU hard actors")
    assets = {}
    for seed, wanted in enumerate(ACTORS):
        path = pupils / f"raw_s1c_s{seed}.pt"
        if sha(path) != wanted:
            raise ValueError(f"wrong signed actor {seed}")
        assets[f"hard_s1c_s{seed}.pt"] = path.read_bytes()
    return assets


def revised(original):
    if hashlib.sha256(original).hexdigest() != CODE:
        raise ValueError("cap250 source bytes differ")
    source = original.decode("utf-8")
    if source.count(MARKER) != 1 or source.count(FACTORY) != 1:
        raise ValueError("expected one official Solver factory")
    fragment = (HERE / "hard_model.py.inc").read_text().rstrip()
    source = source.replace(MARKER, f"{fragment}\n\n\n{MARKER}")
    source = source.replace(FACTORY, FACTORY.replace("Ensemble6", "HardWaveEnsemble"))
    compile(source,"submission.py","exec")
    return source.encode("utf-8")


def assemble(frozen,pupils,lock,out):
    if sha(frozen) != STOCK:
        raise ValueError("unexpected frozen cap250 ZIP")
    if out.exists() or out.resolve() == frozen.resolve():
        raise FileExistsError(out)
    assets = actor_assets(pupils,lock)
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as scratch, ZipFile(frozen) as original:
        names = original.namelist()
        if (len(names) != 39 or len(set(names)) != 39 or {Path(s).name for s in names} != set(names)
                or any(n in names for n in assets) or original.testzip() is not None):
            raise ValueError("unexpected stock archive manifest")
        staged = Path(scratch) / "trial.zip"
        with ZipFile(staged,"x") as pending:
            for entry in original.infolist():
                b = original.read(entry.filename)
                pending.writestr(copy(entry), revised(b) if entry.filename == "submission.py" else b)
            for name, b in assets.items():
                entry = ZipInfo(name)
                entry.compress_type = ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                pending.writestr(entry,b)
        with ZipFile(staged) as trial:
            if trial.namelist() != names + list(assets) or trial.testzip() is not None:
                raise ValueError("invalid trial archive manifest or CRC")
            for name in names:
                if name != "submission.py" and original.read(name) != trial.read(name):
                    raise ValueError(f"stock asset changed: {name}")
            for name,b in assets.items():
                if trial.read(name) != b:
                    raise ValueError(f"signed embedded actor changed: {name}")
        os.link(staged,out)
    return sha(out)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("stock","pupils","lock","output"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    print(assemble(a.stock,a.pupils,a.lock,a.output))
