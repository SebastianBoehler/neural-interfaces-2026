"""Assemble a deterministic experimental package from the immutable cap250 ZIP."""
import argparse
from copy import copy
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
ZIP_HASH = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
CODE_HASH = "37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f"
SOLVER_MARKER = "class Solver(CompetSolver):"
ORIGINAL_FACTORY = 'model = Ensemble6(z, str(meta.get("device", "cpu")), meta["submission_dir"])'


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def revised(source):
    if hashlib.sha256(source).hexdigest() != CODE_HASH:
        raise ValueError("unexpected frozen cap250 submission source")
    code = source.decode("utf-8")
    if code.count(SOLVER_MARKER) != 1 or code.count(ORIGINAL_FACTORY) != 1:
        raise ValueError("expected exactly one solver factory")
    body = HERE.joinpath("fbl2_model.py.inc").read_text().rstrip()
    code = code.replace(SOLVER_MARKER, f"{body}\n\n\n{SOLVER_MARKER}")
    code = code.replace(ORIGINAL_FACTORY, ORIGINAL_FACTORY.replace("Ensemble6", "FBLightEnsemble"))
    compile(code, "submission.py", "exec")
    return code.encode("utf-8")


def assemble(frozen, output, patch=revised):
    if digest(frozen) != ZIP_HASH:
        raise ValueError("frozen cap250 ZIP digest differs")
    if frozen.resolve() == output.resolve() or output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch, ZipFile(frozen) as original:
        names = original.namelist()
        if len(names) != 39 or len(set(names)) != 39 or any(Path(name).name != name for name in names):
            raise ValueError("unexpected frozen archive manifest")
        if original.testzip() is not None:
            raise ValueError("frozen archive CRC check failed")
        temporary = Path(scratch) / output.name
        with ZipFile(temporary, "x") as pending:
            for entry in original.infolist():
                content = original.read(entry.filename)
                if entry.filename == "submission.py":
                    content = patch(content)
                pending.writestr(copy(entry), content)
        with ZipFile(temporary) as verified:
            if verified.namelist() != names or verified.testzip() is not None:
                raise ValueError("experimental archive CRC or manifest differs")
            for name in names:
                if name != "submission.py" and verified.read(name) != original.read(name):
                    raise ValueError(f"frozen asset changed: {name}")
        os.link(temporary, output)
    return digest(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(assemble(args.frozen, args.output))
