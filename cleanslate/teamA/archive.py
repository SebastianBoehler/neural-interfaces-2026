"""Create a root-level competition ZIP from a completed production fit."""
import argparse
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from data import PEOPLE, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    p.add_argument("archive", type=Path)
    a = p.parse_args()
    training = json.loads((a.package / "training.json").read_text())
    if training["smoke"] or set(training["people"]) != set(PEOPLE):
        raise ValueError("A production archive requires every public training person")
    model = a.package / "weights.pt"
    if sha(model) != training["weights_sha"]:
        raise ValueError("Weights differ from the training receipt")
    names = ["submission.py", "weights.pt"]
    if training["arch"] == "manas":
        names.append("LICENSE-MANAS.txt")
    with ZipFile(a.archive, mode="x", compression=ZIP_STORED) as archive:
        for name in names:
            archive.write(a.package / name, name)
    with ZipFile(a.archive) as archive:
        if archive.namelist() != names or archive.testzip() is not None:
            raise ValueError("Package ZIP verification failed")
    print(f"{a.archive} SHA-256 {sha(a.archive)}", flush=True)


if __name__ == "__main__":
    main()
