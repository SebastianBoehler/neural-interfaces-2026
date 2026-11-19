"""Build Lane B test archives from the signed cap250 inputs (no new weights)."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile


ARMS = ("off", "B1", "B2", "B3", "B4", "B5")
STOCK_SHA = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
SOURCE_SHA = "37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f"
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def build(stock, source, out):
    if digest(stock) != STOCK_SHA:
        raise ValueError("Expected the signed cap250 archive")
    code = source.read_bytes()
    marker = b'LANE_B_ARM = "off"'
    if code.count(marker) != 1:
        raise ValueError("Expected one all-off arm in the copied source")
    out.mkdir(parents=True, exist_ok=True)
    with ZipFile(stock) as original:
        if hashlib.sha256(original.read("submission.py")).hexdigest() != SOURCE_SHA:
            raise ValueError("Cap250 source does not match the source pin")
        assets = [(item, original.read(item)) for item in original.infolist()]
        for arm in ARMS:
            path = out / f"{arm}.zip"
            replacement = code.replace(marker, f'LANE_B_ARM = "{arm}"'.encode(), 1)
            with ZipFile(path, "x") as trial:
                for item, data in assets:
                    trial.writestr(item, replacement if item.filename == "submission.py" else data)
            print(f"{digest(path)}  {path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stock", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--source", type=Path, default=ROOT / "submissions/lane_b_cap250/submission.py")
    args = p.parse_args()
    build(args.stock, args.source, args.out)
