"""Isolate a fresh-vs-archive decision change by restoring one fitted family at a time."""

import argparse
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np

from archive_check import rebuild
from assemble import ALIGN, FIXED, load_arrays
from components import FAMILIES, ORDINARY, SBN, Variant, assets
from source import PEOPLE, file_digest
from verify import SEALED, harness, labels

ORDERS = (None, 71000)
FAMILIES_STOCK = tuple(dict.fromkeys(family for family, _ in Variant().fits()))


def payload(family, arrays):
    seeds = range(FAMILIES[family].seeds)
    files = {name for seed in seeds for name in assets(family, seed) if name.endswith(".pt")}
    if family in ORDINARY or family in SBN or family == "teacher":
        bases = [ORDINARY[family]] if family in ORDINARY else (
            [index for index, _ in SBN[family]] if family in SBN else [34])
        prefixes = tuple(f"net{index + seed}_" for index in bases for seed in seeds)
        keys = {key for key in arrays if key.startswith(prefixes)}
    elif family == "rest":
        keys = {f"v6_rest_{key}" for key in ALIGN}
    elif family == "count":
        keys = {key for key in arrays if key.startswith("v6_cf_")} - {"v6_cf_on"}
    else:
        keys = FIXED.get(family, set()) | ({"v6_tpri"} if family == "geometry" else set())
    return keys, files


def prepare(run, fresh, sealed, data):
    if file_digest(sealed) != SEALED:
        raise ValueError("sealed reference ZIP SHA-256 changed")
    run.mkdir(exist_ok=True)
    archive = run / "archive"
    archive.mkdir()
    with ZipFile(sealed) as zip_file:
        zip_file.extractall(archive)
    rebuild(sealed, run / "reassembled", data)
    old, new = (load_arrays(path / "weights.npz") for path in (archive, fresh))
    for line in fresh.with_suffix(".sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if file_digest(fresh / name) != digest:
            raise ValueError(f"fresh package digest changed: {name}")
    files = {path.name for path in archive.iterdir()}
    if new.keys() != old.keys() or {path.name for path in fresh.iterdir()} != files:
        raise ValueError("fresh and archived stock artifacts have different inventories")
    payloads = {family: payload(family, old) for family in FAMILIES_STOCK}
    assigned_keys = set().union(*(keys for keys, _ in payloads.values()))
    assigned_files = set().union(*(names for _, names in payloads.values()))
    changed_keys = {key for key in old if old[key].dtype != new[key].dtype or not np.array_equal(old[key], new[key])}
    changed_files = {name for name in files - {"weights.npz"}
                     if file_digest(archive / name) != file_digest(fresh / name)}
    if changed_keys - assigned_keys or changed_files - assigned_files:
        raise ValueError(f"differences outside fitted families: {changed_keys - assigned_keys}, "
                         f"{changed_files - assigned_files}")
    shutil.copytree(fresh, run / "fresh", copy_function=lambda src, dst: Path(dst).hardlink_to(src))
    for family, (keys, names) in payloads.items():
        restore = run / f"restore-{family}"
        swap(restore, fresh, archive, new, old, keys, names)
        if family == "sbn_fbc_raw":
            swap(run / "raw-only", archive, fresh, old, new, keys, names)


def swap(out, base, donor, base_arrays, donor_arrays, keys, files):
    out.mkdir()
    for path in base.iterdir():
        if path.name != "weights.npz":
            (out / path.name).hardlink_to(donor / path.name if path.name in files else path)
    np.savez(out / "weights.npz", **{**base_arrays, **{key: donor_arrays[key] for key in keys}})


def variants():
    return ("archive", "reassembled", "fresh", "raw-only", *(f"restore-{name}" for name in FAMILIES_STOCK))


def replay(run, data_path, task):
    variant, person = [(variant, person) for variant in variants() for person in PEOPLE][task]
    data = harness.load(data_path)
    predictions = labels(run / variant, data, ORDERS, person)
    np.savez(run / f"{variant}-{person}.npz", **{
        f"{order}|{visit}|{field}": value
        for order, sessions in predictions.items() for visit, (rows, decisions, _) in sessions.items()
        for field, value in (("rows", rows), ("labels", decisions))})


def compare(run, original):
    summary = {str(order): {tag: {"vs_archive": 0, "vs_fresh": 0, "restored": 0}
                            for tag in variants()} for order in ORDERS}
    for person in PEOPLE:
        with np.load(original / f"{person}.npz", allow_pickle=False) as rows:
            observed = {key: rows[key] for key in rows.files}
        ref = {tag: load_arrays(run / f"{tag}-{person}.npz") for tag in variants()}
        archive, fresh = ref["archive"], ref["fresh"]
        expected_keys = {f"{key}|{field}" for key in observed for field in ("rows", "labels")}
        if any(value.keys() != expected_keys for value in ref.values()):
            raise ValueError(f"attribution key coverage mismatch for {person}")
        for key, changed in observed.items():
            rows = archive[f"{key}|rows"]
            old, new = archive[f"{key}|labels"], fresh[f"{key}|labels"]
            if not np.array_equal(rows[old != new], changed):
                raise ValueError(f"fresh replay did not reproduce the original mismatch rows: {key}")
            for tag, predictions in ref.items():
                delivered, decisions = predictions[f"{key}|rows"], predictions[f"{key}|labels"]
                if not np.array_equal(rows, delivered):
                    raise ValueError(f"attribution delivery mismatch for {tag}: {key}")
                counts = summary[key.split("|", 1)[0]][tag]
                counts["vs_archive"] += int((old != decisions).sum())
                counts["vs_fresh"] += int((new != decisions).sum())
                counts["restored"] += int(((old != new) & (old == decisions)).sum())
    (run / "attribution.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    for order, by_tag in summary.items():
        if by_tag["reassembled"]["vs_archive"]:
            raise ValueError(f"archive reconstruction changed public decisions: {order}")
        observed = by_tag["fresh"]["vs_archive"]
        if (by_tag["restore-sbn_fbc_raw"]["restored"] != observed
                or by_tag["raw-only"]["vs_fresh"]
                or any(by_tag[f"restore-{family}"]["restored"] for family in FAMILIES_STOCK
                       if family != "sbn_fbc_raw")):
            raise ValueError(f"public decision mismatch was not isolated to raw SessionBN: {order}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, arguments in {"prepare": ("run", "fresh", "sealed", "data"),
                            "replay": ("run", "data"), "compare": ("run", "original")}.items():
        command = commands.add_parser(name)
        for argument in arguments:
            command.add_argument(argument, type=Path)
        if name == "replay":
            command.add_argument("task", type=int)
    kwargs = vars(parser.parse_args())
    if kwargs["command"] == "replay":
        kwargs["data_path"] = kwargs.pop("data")
    globals()[kwargs.pop("command")](**kwargs)
