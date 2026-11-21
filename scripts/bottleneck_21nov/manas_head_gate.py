"""Audit native MANAS A1 gates and freeze a relaxed gate using fit orders only."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from manas_head_reachability import DATA_SHA, PEOPLE, RETURNING, ZIP_SHA, digest, plans, signed_gate
from tapes import pinned, stream_ids

ORDERS = {"fit": tuple(range(71800, 71808)), "held": tuple(range(71808, 71816))}
GRID = tuple(product((1.0, 1.15, 1.30, 1.50), (0.80, 0.85, 0.90, 0.95)))
NATIVE = GRID[0]


def read_bank(path, channels):
    pkg = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if pkg["recipe"] != "A1" or pkg["arch"] != "manas" or tuple(pkg["names"]) != channels:
        raise ValueError(f"wrong MANAS A1 bank: {path}")
    bank = pkg["bank"]
    if not bank["enrolled"].numpy().all():
        raise ValueError(f"unexpected unenrolled MANAS slot: {path}")
    return tuple(np.asarray(bank[k]) for k in ("gallery", "identity_scale", "identity_radius"))


def reconstruct(gallery, names):
    g = np.stack([gallery[name] for name in names])
    scale = 1 / np.maximum(g.std(0), 0.08)
    dist = np.sqrt(np.mean(((g[None] - g[:, None]) * scale) ** 2, axis=-1))
    return g, scale, np.quantile(dist[~np.eye(len(g), dtype=bool)], 0.10) * 0.8


def verify_bank(exact, rebuilt):
    np.testing.assert_allclose(exact[0], rebuilt[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(exact[1], rebuilt[1], rtol=2e-6, atol=0)
    np.testing.assert_allclose(exact[2], rebuilt[2], rtol=0, atol=1e-6)


def witness(features, rows, names, bank, settings):
    g, scale, radius = bank
    ages = np.arange(12, len(rows) + 1, 4)
    means = np.cumsum(features[rows], axis=0, dtype=np.float64)[ages - 1] / ages[:, None]
    distances = (((g[None] - means[:, None]) * scale) ** 2).mean(-1) ** 0.5
    ranks = np.argsort(distances, axis=1)
    nearest = ranks[:, 0]
    d1, d2 = np.take_along_axis(distances, ranks[:, :2], axis=1).T
    gap = d1 / np.maximum(d2, 1e-6)
    result = {}
    for multiplier, margin in settings:
        yes = (d1 < radius * multiplier) & (gap < margin)
        ok = np.flatnonzero(yes[1:] & yes[:-1] & (nearest[1:] == nearest[:-1]))
        if len(ok):
            i = ok[0] + 1
            result[multiplier, margin] = (int(ages[i]), names[nearest[i]])
        else:
            result[multiplier, margin] = (-1, "")
    failures = dict(top=Counter(names[i] for i in nearest), min_distance=float(d1.min()),
                    min_ratio=float(gap.min()), radius_pass=int(np.sum(d1 < radius)),
                    margin_pass=int(np.sum(gap < NATIVE[1])))
    return result, failures


def summarise(records, setting):
    counts = Counter()
    for fold, visit, n, runs in records:
        age, identity = runs[setting]
        counts[f"{fold.split('_')[0]}/v{visit}/total"] += 1
        if age < 0:
            continue
        counts[f"{fold.split('_')[0]}/v{visit}/active"] += 1
        if fold.startswith("ret_") and identity == fold.split("_")[1]:
            counts["correct"] += 1
            counts["active_deliveries"] += n - age + 1
        else:
            counts["impostors"] += 1
    return dict(counts)


def audit(a):
    if digest(a.zip) != ZIP_SHA or digest(a.data) != DATA_SHA:
        raise ValueError("unexpected signed ZIP or EEG data")
    if a.phase == "fit":
        settings = GRID
    else:
        frozen = json.loads((a.out / "fit.json").read_text())
        selected = tuple(frozen["selected"]) if frozen["selected"] else None
        settings = (NATIVE,) if selected is None or selected == NATIVE else (NATIVE, selected)
        if frozen["exact_folds"] != bool(a.fixtures):
            raise ValueError("fit and held banks differ")
    with ZipFile(a.zip) as archive, TemporaryDirectory() as tmp:
        gate = signed_gate(archive.read("submission.py").decode())
        weights = Path(tmp) / "team_a.pt"
        with archive.open("team_a.pt") as source, weights.open("wb") as target:
            shutil.copyfileobj(source, target)
        with np.load(a.data, allow_pickle=False) as z:
            people = np.array([x.rsplit("/", 1)[-1] for x in z["subject"]])
            visits = z["session"]
            channels = tuple(str(x).removeprefix("EEG ").split("-", 1)[0] for x in z["ch_names"])
            raw = gate["center"](z["X"])
        features = gate["fingerprint"](raw, gate["covariance"](raw))
        gallery = {p: features[(people == p) & (visits == "1")].mean(0) for p in PEOPLE}
        signed = read_bank(weights, channels)
    verify_bank(signed, reconstruct(gallery, PEOPLE))
    data = {"person": people, "session": visits, "stream": np.char.add(np.char.add(people, "|"), visits)}
    records, details, radii = [], {}, {}
    for cohort, roster in (("ret", RETURNING), ("new", PEOPLE)):
        for person in roster:
            fold = f"{cohort}_{person}"
            names = tuple(p for p in PEOPLE if cohort == "ret" or p != person)
            rebuilt = reconstruct(gallery, names)
            bank = rebuilt
            if a.fixtures:
                pkg = a.fixtures / fold
                receipt = json.loads((pkg / "receipt.json").read_text())["ba_six"]
                path = pkg / "ba_six/team_a.pt"
                if digest(path) != receipt["partner"]["weights_sha"]:
                    raise ValueError(f"MANAS bank differs from fold receipt: {fold}")
                bank = read_bank(path, channels)
                verify_bank(bank, rebuilt)
            radii[fold] = float(bank[2])
            details[fold] = {}
            for order in ORDERS[a.phase]:
                for stream, rows in plans(data, order, stream_ids(data, cohort, person)).items():
                    visit = stream[-1]
                    key = f"{order}_{visit}"
                    saved, _, receipt = pinned(a.pins, fold, "ba_six", order, key)
                    np.testing.assert_array_equal(rows, saved)
                    if a.fixtures and json.loads(receipt) != json.loads((pkg / "receipt.json").read_text())["ba_six"]:
                        raise ValueError(f"pinned rows used different fold: {fold}")
                    runs, failure = witness(features, rows, names, bank, settings)
                    records.append((fold, visit, len(rows), runs))
                    prod = witness(features, rows, PEOPLE, signed, (NATIVE,))[0][NATIVE] if cohort == "ret" else None
                    details[fold][key] = (runs, failure, prod)
    stats = {setting: summarise(records, setting) for setting in settings}
    if a.phase == "fit":
        selected = min((s for s in settings if not stats[s].get("impostors")),
                       key=lambda s: (-stats[s].get("correct", 0), -stats[s].get("active_deliveries", 0), *s),
                       default=None)
    output = dict(exact_folds=bool(a.fixtures), orders=ORDERS[a.phase], selected=selected, radii=radii,
                  signed_radius=float(signed[2]), native=stats[NATIVE], relaxed=stats.get(selected),
                  grid={str(k): v for k, v in stats.items()}, streams={})
    for fold, streams in details.items():
        output["streams"][fold] = {k: dict(native=r[NATIVE], relaxed=r.get(selected), diagnostics=d, signed=p)
                                   for k, (r, d, p) in streams.items()}
    target = a.out / f"{a.phase}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(a.phase, "exact folds", bool(a.fixtures), "signed radius", output["signed_radius"],
          "native", output["native"], "selected", selected, output["relaxed"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=ORDERS)
    for name in ("zip", "data", "pins", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--fixtures", type=Path)
    audit(p.parse_args())
