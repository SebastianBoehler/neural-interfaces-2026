"""Isolate assembler parity by splitting the authenticated stock ZIP into temporary family artifacts."""

import argparse
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from assemble import FIXED, assemble, load_arrays
from components import ORDINARY, SBN, Variant, assets, part_path
from source import PEOPLE, default_data, file_digest
from verify import SEALED


def extract_parts(archive, parts):
    z = load_arrays(archive / "weights.npz")
    for family, seed in Variant().fits():
        part = part_path(parts, family, seed)
        part.mkdir(parents=True)
        if family in ORDINARY or family in SBN or family == "teacher":
            if family in ORDINARY:
                ids = (ORDINARY[family] + seed,)
            elif family in SBN:
                ids = tuple(index + seed for index, _ in SBN[family])
            else:
                ids = (34,)
            np.savez(part / "weights.npz", **{k: v for k, v in z.items()
                                            if any(k.startswith(f"net{i}_") for i in ids)})
        elif family in FIXED:
            np.savez(part / "weights.npz", **{key: z[key] for key in FIXED[family]})
        for file in assets(family, seed):
            if re.fullmatch(r"rest_s\d+\.npz", file):
                np.savez(part / file, **{key: z["v6_rest_" + key] for key in ("align", "k", "G", "scale", "shrink")})
            elif family == "geometry":
                np.savez(part / file, tpri=z["v6_tpri"], names=z["route_names"], sessions=z["bio_m"])
            elif family == "count":
                np.savez(part / file, **{key.removeprefix("v6_"): value for key, value in z.items()
                                         if key.startswith("v6_cf_") and key != "v6_cf_on"})
            else:
                shutil.copyfile(archive / file, part / file)


def rebuild(sealed, out, data):
    if file_digest(sealed) != SEALED:
        raise ValueError("sealed reference ZIP SHA-256 changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as scratch, ZipFile(sealed) as zip_file:
        root = Path(scratch)
        archive, parts = root / "archived", root / "parts"
        zip_file.extractall(archive)
        extract_parts(archive, parts)
        assemble(PEOPLE, (), data, parts, out, Variant(), require_receipts=False)
        old, new = (load_arrays(p / "weights.npz") for p in (archive, out))
        diff = [k for k in old if k not in new or new[k].dtype != old[k].dtype or not np.array_equal(new[k], old[k])]
        payload_diff = [file.name for file in archive.iterdir() if file.name != "weights.npz" and
                        file_digest(file) != file_digest(out / file.name)]
        if diff or new.keys() != old.keys() or payload_diff:
            raise ValueError(f"archived part assembly differs: arrays={diff}, payload={payload_diff}")
        print(f"PASS {len(old)} arrays and {len(list(archive.iterdir())) - 1} payload files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=default_data())
    args = parser.parse_args()
    rebuild(args.sealed, args.out, args.data)
