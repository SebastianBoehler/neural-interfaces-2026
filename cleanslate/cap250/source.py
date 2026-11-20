"""Select the public source rows before a cap250 family learns any statistics."""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get("CAP250_ASSETS", ROOT)).resolve()
PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))


def default_data():
    return Path(os.environ.get("CS_DATA", ASSETS / "data/proteus_windows.npz")).resolve()


def roster(value):
    people = value.split(",") if value else []
    if len(people) != len(set(people)) or set(people) - set(PEOPLE):
        raise argparse.ArgumentTypeError(f"invalid PROTEUS roster: {value!r}")
    return tuple(sorted(people))


def add_scope(parser):
    parser.add_argument("--people", type=roster, required=True)
    parser.add_argument("--first-session-only", type=roster, default=())
    parser.add_argument("--data", type=Path, default=default_data())


def validate_scope(people, first_session_only):
    if not people or set(first_session_only) - set(people):
        raise ValueError("the source roster must include every first-session-only person")


def family_args(description, seeds):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=("train",))
    add_scope(parser)
    parser.add_argument("--seed", required=True, type=int, choices=seeds)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validate_scope(args.people, args.first_session_only)
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error(f"output must be empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def selection(archive, people, first_session_only=()):
    validate_scope(people, first_session_only)
    person = np.array([str(s).rsplit("/", 1)[-1] for s in archive["subject"]])
    session = archive["session"].astype(str)
    keep = np.isin(person, people)
    keep &= ~(np.isin(person, first_session_only) & (session != "1"))
    rows = np.flatnonzero(keep)
    if set(person[rows]) != set(people):
        raise ValueError("one or more source people are missing from the dataset")
    return rows, person[rows], session[rows]


def source_rows(people, first_session_only=(), path=None):
    with np.load(path or default_data(), allow_pickle=False) as archive:
        return selection(archive, people, first_session_only)[0]


def load_source(people, first_session_only=(), path=None):
    """Return an experiments.protocol-format dictionary with source rows only."""
    with np.load(path or default_data(), allow_pickle=False) as archive:
        rows, person, session = selection(archive, people, first_session_only)
        data = {
            key: archive[key][rows]
            for key in archive.files
            if key not in {"ch_names"}
        }
        data["ch_names"] = archive["ch_names"]
    data.update(person=person, session=session, index=rows, w=np.ones(len(rows)),
                nolow=np.zeros(len(rows), dtype=bool), tag="")
    data["stream"] = np.char.add(np.char.add(data["subject"], "|"), session)
    if data["X"].shape[1:] != (41, 480) or data["X"].dtype != np.float32:
        raise ValueError("expected the official float32 41×480 waveform")
    return data


def row_digest(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def scope_args(people, first_session_only, data):
    args = ["--people", ",".join(people), "--data", str(Path(data).resolve())]
    if first_session_only:
        args += ["--first-session-only", ",".join(first_session_only)]
    return args
