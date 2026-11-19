"""Append a small Lane D switch to a copy of the signed cap250 archive."""

from copy import copy
import hashlib
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
STOCK_SHA = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
SOURCE_SHA = "37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build(stock, directory):
    if digest(stock) != STOCK_SHA:
        raise ValueError("expected signed cap250 ZIP")
    policy = (HERE / "balance.py").read_bytes()
    hook = (HERE / "package_hook.py").read_bytes()
    marker = b'LANE_D_ARM = "off"'
    if hook.count(marker) != 1:
        raise ValueError("expected one package mode marker")
    directory.mkdir(parents=True, exist_ok=True)
    with stock.open("rb") as source, (directory / "off.zip").open("xb") as disabled:
        shutil.copyfileobj(source, disabled)
    with ZipFile(stock) as original:
        submission = original.read("submission.py")
        if hashlib.sha256(submission).hexdigest() != SOURCE_SHA:
            raise ValueError("expected signed cap250 source")
        replacement = hook.replace(marker, b'LANE_D_ARM = "output"')
        with ZipFile(directory / "output.zip", "x") as trial:
            for item in original.infolist():
                data = original.read(item)
                if item.filename == "submission.py":
                    data = b"\n\n".join((data, policy, replacement))
                    compile(data, "output/submission.py", "exec")
                trial.writestr(copy(item), data)


if __name__ == "__main__":
    build(*map(Path, sys.argv[1:]))
