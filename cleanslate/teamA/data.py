"""Data selection, causal alignment and source-only virtual subjects."""
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path

import numpy as np

from eeg import center, covariance, matrix_power, update_transform

PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
CORE = ("physionetmi", "cho2017", "lee2019_mi", "dreyer2023", "schirrmeister2017",
        "shin2017a", "shin2017b", "shin2018")
EXT = ("scherer2015", "ds007221g", "ds007221lr", "netbci")
TASK = {"left_hand": 0, "right_hand": 0, "hands": 0, "feet": 0, "tongue": 0,
        "subtraction": 1, "WG": 2, "S16": 2, "rest": 3, "baseline": 3}
EXPECTED = {
    "shin2017a": {"left_hand", "right_hand"}, "shin2017b": {"rest", "subtraction"},
    "shin2018": {"baseline", "WG"}, "scherer2015": {"hands", "subtraction", "S16"},
    "ds007221g": {"feet", "left_hand", "rest", "right_hand"},
    "ds007221lr": {"left_hand", "right_hand"}, "netbci": {"rest", "right_hand"},
}


@dataclass
class Trials:
    x: np.ndarray
    y: np.ndarray
    person: np.ndarray
    visit: np.ndarray
    interface: np.ndarray
    names: tuple
    classes: tuple = ()

    def subset(self, rows):
        return Trials(self.x[rows], self.y[rows], self.person[rows], self.visit[rows],
                      self.interface[rows], self.names, self.classes)

    def sessions(self):
        keys = np.array([f"{p}/{v}" for p, v in zip(self.person, self.visit)])
        return (np.flatnonzero(keys == key) for key in np.unique(keys))


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_public(path):
    with np.load(path, allow_pickle=False) as f:
        return Trials(f["X"], f["y"],
                      np.array([p.rsplit("/", 1)[-1] for p in f["subject"]]),
                      f["session"].astype(str), f["interface"], tuple(str(c).removeprefix("EEG ").split("-", 1)[0] for c in f["ch_names"]))


def source_rows(data, people, enroll_first):
    if not people or set(people) - set(data.person):
        raise ValueError(f"Requested people outside PROTEUS: {people}")
    mask = np.isin(data.person, people)
    mask &= ~np.isin(data.person, enroll_first) | (data.visit == "1")
    return np.flatnonzero(mask)


def external_data(corpus, directories, names, rng, excluded=(), smoke=False):
    for directory in directories:
        folder = directory / corpus
        files = sorted(folder.glob("windows_*.npz"))
        single = folder / "windows.npz"
        if not files and single.is_file():
            files = [single]
        if files:
            break
    if not files:
        raise FileNotFoundError(f"{corpus} has no window exports in {directories}")
    fields, hashes, classes = [], {}, None
    for path in files[:1] if smoke else files:
        hashes[f"{corpus}/{path.name}"] = sha(path)
        with np.load(path, allow_pickle=False) as f:
            found = tuple(f["classes"]) if "classes" in f else ()
            if corpus == "scherer2015" and not found:
                if hashes[f"{corpus}/{path.name}"] != "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91":
                    raise ValueError("Scherer label order requires the audited export")
                found = ("hands", "subtraction", "S16")
            if not found or (corpus in EXPECTED and set(found) != EXPECTED[corpus]):
                raise ValueError(f"{corpus}: unexpected labels {found}")
            if classes is None:
                classes = found
            if classes != found:
                raise ValueError(f"{corpus}: inconsistent index mapping across shards")
            y, people, visits = f["y"], f["subject"].astype(str), f["session"].astype(str)
            if y.min() < 0 or y.max() >= len(classes):
                raise ValueError(f"{corpus}: label outside class map")
            mask = ~np.isin(people, excluded)
            if "win" in f:
                mask &= f["win"] == 0
            selected = []
            for person in np.unique(people[mask]):
                for label in np.unique(y[mask & (people == person)]):
                    rows = np.flatnonzero(mask & (people == person) & (y == label))
                    selected.extend(rng.choice(rows, min(len(rows), 6 if smoke else 64), replace=False))
            if not selected:
                continue
            rows = np.sort(selected)
            original = [str(c).removeprefix("EEG ").split("-", 1)[0] for c in f["ch_names"]]
            if set(original) != set(names):
                raise ValueError(f"{corpus}: montage differs from PROTEUS export")
            x = f["X"][rows][:, [original.index(c) for c in names]]
            fields.append((x, y[rows], people[rows], visits[rows]))
    if not fields:
        raise ValueError(f"{corpus}: empty after selection")
    x, y, people, visits = map(np.concatenate, zip(*fields))
    expected_people = {"ds007221g": 14, "ds007221lr": 48, "netbci": 19}.get(corpus)
    if not smoke and expected_people is not None and len(np.unique(people)) != expected_people:
        raise ValueError(f"{corpus}: incomplete B6 roster, expected {expected_people} people")
    if x.shape[1:] != (41, 480) or not np.isfinite(x).all():
        raise ValueError(f"{corpus}: invalid model tensor")
    return Trials(x, y, people, visits, np.repeat(corpus, len(y)), names, classes), hashes


def prior(data):
    cov = covariance(center(data.x))
    return np.mean([cov[rows].mean(0) for rows in data.sessions()], axis=0, dtype=np.float64).astype(np.float32)


def align_trials(data, source, rng):
    raw = center(data.x)
    cov = covariance(raw)
    aligned = np.empty_like(raw)
    for rows in data.sessions():
        order = rng.permutation(rows)
        totals = np.cumsum(cov[order], axis=0, dtype=np.float64)
        ages = np.arange(1, len(rows) + 1, 8)
        maps = update_transform(totals[ages - 1], ages[:, None, None], source)
        aligned[order] = maps[np.arange(len(rows)) // 8] @ raw[order]
    return aligned


def virtual_subjects(data, limit=None):
    cov = covariance(center(data.x))
    sessions = list(data.sessions())
    if limit and len(sessions) > limit:
        sessions = [sessions[i] for i in np.linspace(0, len(sessions) - 1, limit, dtype=int)]
    spectra = np.stack([cov[rows].mean(0) for rows in sessions])
    std = np.trace(spectra, axis1=-2, axis2=-1) / 41
    roots = matrix_power(spectra / std[:, None, None], 0.5)
    inv = matrix_power(spectra / std[:, None, None], -0.5)
    return (roots[:, None] @ inv[None, :]).reshape(-1, 41, 41).astype(np.float32)
