"""Stage two causally aligned orders of one session per external corpus person."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from base import D
from context import MAX_PREFIX
from episodes import aligned, encode

CORPORA = (
    "physionetmi", "cho2017", "lee2019_mi", "dreyer2023", "schirrmeister2017",
    "shin2017a", "shin2017b", "ds007221g", "ds007221lr", "netbci",
)
B6 = {
    "ds007221g": (14, ["feet", "left_hand", "rest", "right_hand"]),
    "ds007221lr": (48, ["left_hand", "right_hand"]),
    "netbci": (19, ["rest", "right_hand"]),
}


def stage(corpus, data, output, smoke):
    if not torch.cuda.is_available():
        raise RuntimeError("external staging requires a Slurm GPU")
    with np.load(D.DATA) as proteus:
        channels = proteus["ch_names"]
    files = sorted((data / corpus).glob("windows_*.npz"))
    if not files:
        raise FileNotFoundError(f"missing corpus {corpus} in {data}")
    out = output / corpus
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    manifest.unlink(missing_ok=True)
    rng = np.random.default_rng([1675, CORPORA.index(corpus)])
    people, parts, classes, counts = set(), [], None, None
    for f in files[:1] if smoke else files:
        with np.load(f, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["ch_names"], channels)
            if not np.all(z["split"] == "external"):
                raise ValueError(f"non-external rows in {f}")
            current = z["classes"].tolist()
            if classes is None:
                classes, counts = current, np.zeros(len(current), dtype=int)
            if current != classes or (corpus in B6 and current != B6[corpus][1]):
                raise ValueError(f"class map mismatch for {f}: {current}")
            x, y, subject, sessions = (z[k] for k in ("X", "y", "subject", "session"))
        if x.shape != (len(y), 41, 480) or not np.all((0 <= y) & (y < len(classes))):
            raise ValueError(f"invalid EEG/labels in {f}")
        picked = sorted(set(subject))
        if people.intersection(picked) or any(p.rsplit("/", 1)[-1] in D.PEOPLE for p in picked):
            raise ValueError(f"non-disjoint corpus roster at {f}")
        picked = picked[:1] if smoke else picked
        people.update(picked)
        xs, ys, lengths = [], [], []
        for person in picked:
            mine = subject == person
            session = rng.choice(np.unique(sessions[mine]))
            rows = np.flatnonzero(mine & (sessions == session))
            if len(rows) < 2:
                raise ValueError(f"too few external windows for {person} in {session}")
            for _ in range(2):
                p = rng.permutation(rows)[:MAX_PREFIX + 1]
                a = aligned(x[p]).cpu().numpy()
                if not np.isfinite(a).all():
                    raise ValueError(f"nonfinite aligned EEG in {person} {session}")
                xs.append(a)
                ys.append(y[p])
                lengths.append(len(p))
                counts += np.bincount(y[p], minlength=len(classes))
        np.savez(out / f.name, X=np.concatenate(xs), y=np.concatenate(ys), lengths=lengths)
        with f.open("rb") as source:
            parts.append({"name": f.name, "sha256": file_digest(source, "sha256").hexdigest()})
    if not smoke and (not counts.all() or (corpus in B6 and len(people) != B6[corpus][0])):
        raise ValueError(f"incomplete {corpus} external coverage: {len(people)} people, {counts}")
    receipt = dict(people=len(people), classes=classes, counts=counts.tolist(), parts=parts, smoke=smoke)
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    print(corpus, json.dumps(receipt), flush=True)


def read_episodes(base, root, corpus, smoke):
    directory = root / corpus
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["smoke"] != smoke:
        raise ValueError(f"external stage mismatch for {corpus}")
    episodes = []
    for part in manifest["parts"]:
        with np.load(directory / part["name"], allow_pickle=False) as z:
            x, y, lengths = (z[k] for k in ("X", "y", "lengths"))
        cuts = np.cumsum(lengths)[:-1]
        episodes.extend(encode(base, a, b) for a, b in zip(np.split(x, cuts), np.split(y, cuts)))
    if len(episodes) != 2 * manifest["people"]:
        raise ValueError(f"incomplete episodes for {corpus}")
    return episodes, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", type=int)
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    stage(CORPORA[args.task], args.data, args.output, args.smoke)


if __name__ == "__main__":
    main()
