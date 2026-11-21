"""Read frozen data metadata and enforce fold, delivery and reference parity."""
from dataclasses import dataclass
from functools import cached_property
from hashlib import file_digest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cleanslate/harness"))
import data as protocol

ROSTERS = ("nine25", "ba_six")
PEOPLE = {"new": tuple(protocol.PEOPLE), "ret": tuple(protocol.RETURNING)}
POSITIONS = ("1–10", "11–20", "21–40", "41+")
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"


@dataclass(frozen=True)
class Case:
    cohort: str
    person: str
    visit: str
    order: int
    rows: np.ndarray
    y: np.ndarray
    interface: np.ndarray
    scores: dict

    @cached_property
    def weights(self):
        result = np.zeros(len(self.y))
        for interface in np.unique(self.interface):
            in_cell = self.interface == interface
            classes = np.unique(self.y[in_cell])
            for label in classes:
                mask = in_cell & (self.y == label)
                result[mask] = 1 / (mask.sum() * len(classes))
        np.testing.assert_allclose(result.sum(), len(np.unique(self.interface)), rtol=0, atol=1e-12)
        return result

    @cached_property
    def position(self):
        return np.digitize(np.arange(1, len(self.y) + 1), (11, 21, 41))

    @property
    def key(self):
        return self.cohort, self.order, self.person, self.visit


def metadata(path):
    with path.open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != DATA_SHA:
            raise ValueError("unexpected PROTEUS export")
    with np.load(path, allow_pickle=False) as tape:
        d = {key: tape[key] for key in ("subject", "session", "interface", "y")}
    d["person"] = np.array([value.rsplit("/", 1)[-1] for value in d["subject"]])
    d["stream"] = np.char.add(np.char.add(d["person"], "|"), d["session"])
    return d


def stream_ids(data, cohort, person):
    visits = ("2", "3") if cohort == "ret" else sorted(set(data["session"][data["person"] == person]))
    return [f"{person}|{visit}" for visit in visits]


def read_bank(data, orders, read, *, deliveries=None):
    if deliveries is None:
        streams = np.unique(data["stream"])
        deliveries = {order: protocol.plans(data, order, streams) for order in orders}
    result = []
    for cohort, people in PEOPLE.items():
        for person in people:
            for order in orders:
                for stream in stream_ids(data, cohort, person):
                    visit = stream.split("|")[1]
                    fold, key = f"{cohort}_{person}", f"{order}_{visit}"
                    rows, scores = read(fold, order, key)
                    np.testing.assert_array_equal(rows, deliveries[order][stream], err_msg=f"{fold}/{key}")
                    if any(len(value) != len(rows) or not np.isfinite(value).all() for value in scores.values()):
                        raise ValueError(f"missing or nonfinite scores for {fold}/{key}")
                    result.append(Case(cohort, person, visit, order, rows, data["y"][rows],
                                       data["interface"][rows], scores))
    return result


def pinned(leo, fold, roster, order, key):
    path = leo / fold / roster / f"pred_{fold}_{order - order % 2}.npz"
    with np.load(path, allow_pickle=False) as tape:
        return tape[f"rows_{key}"], tape[f"labels_{key}"], str(tape["receipt"])


def labels(data, leo, orders):
    def read(fold, order, key):
        scores = {}
        for roster in ROSTERS:
            rows, expected, receipt = pinned(leo, fold, roster, order, key)
            block = "fit" if order < 71808 else "held"
            with np.load(leo / block / roster / fold / f"{order - order % 2}.npz", allow_pickle=False) as tape:
                if str(tape["receipt"]) != receipt:
                    raise ValueError(f"fixture receipt mismatch: {fold}/{key}/{roster}")
                np.testing.assert_array_equal(rows, tape[f"rows_{key}"])
                np.testing.assert_array_equal(expected, tape[f"w25_{key}"])
            scores[roster] = expected
        return rows, scores
    return read_bank(data, orders, read)


def stages(data, work, roster, orders):
    def read(fold, order, key):
        path = work / "capture" / f"{roster}-{fold}-{order - order % 2}.npz"
        rows, expected, receipt = pinned(work / "leo", fold, roster, order, key)
        with np.load(path, allow_pickle=False) as tape:
            if str(tape["receipt"]) != receipt:
                raise ValueError(f"wrong capture fixture: {path}")
            np.testing.assert_array_equal(rows, tape[f"rows_{key}"])
            np.testing.assert_array_equal(expected, tape[f"label_{key}"])
            fields = {field.removesuffix(f"_{key}"): tape[field] for field in tape.files
                      if field.endswith(f"_{key}") and not field.startswith("rows_")}
            names = tape["route_names"].tolist()
            fields["gate_person"] = np.array([int(names[i][1:]) if i >= 0 else 0 for i in fields["gate_id"]])
        return rows, fields
    return read_bank(data, orders, read)


def subsets(cases, cohort=None, interface=None, position=None):
    for case in cases:
        if cohort is not None and case.cohort != cohort:
            continue
        mask = np.ones(len(case.y), dtype=bool)
        if interface is not None:
            mask &= case.interface == interface
        if position is not None:
            mask &= case.position == position
        if mask.any():
            yield case, mask


def mass(cases, values, cohort, **filters):
    tally = weight = 0.0
    for case, mask in subsets(cases, cohort=cohort, **filters):
        w = case.weights[mask]
        tally += w @ values(case)[mask]
        weight += w.sum()
    if not weight:
        raise ValueError(f"empty {cohort} stratum: {filters}")
    return tally, weight


def score(cases, predict, cohort, **filters):
    tally, weight = mass(cases, lambda case: predict(case) == case.y, cohort, **filters)
    return 100 * tally / weight


def joint(cases, predict, **filters):
    return np.mean([score(cases, predict, cohort, **filters) for cohort in PEOPLE])


def paired(cases, candidate, reference, orders, *, nested_orders=False):
    rng_o = np.random.default_rng(20261121)
    rng_p = np.random.default_rng(71120)
    index = rng_o.integers(len(orders), size=(20_000, len(orders)))
    out, boot = {}, []
    for cohort, people in PEOPLE.items():
        values = np.zeros((len(orders), len(people)))
        counts = np.zeros_like(values)
        for case, _ in subsets(cases, cohort=cohort):
            i, j = orders.index(case.order), people.index(case.person)
            delta = (candidate(case) == case.y).astype(int) - (reference(case) == case.y)
            values[i, j] += 100 * case.weights @ delta
            counts[i, j] += case.weights.sum()
        np.testing.assert_allclose(counts, np.tile(counts[0], (len(orders), 1)), rtol=0, atol=1e-10)
        if (counts <= 0).any():
            raise ValueError("paired draw misses a person/order")
        draws_p = rng_p.integers(len(people), size=(len(index), len(people)))
        if nested_orders:
            draws_o = rng_o.integers(len(orders), size=(*draws_p.shape, len(orders)))
            sampled = values[draws_o, draws_p[:, :, None]]
        else:
            sampled = values[index[:, :, None], draws_p[:, None, :]]
        draw = sampled.sum((1, 2)) / (len(orders) * counts[0, draws_p].sum(1))
        out[cohort] = (values.sum() / counts.sum(), *np.quantile(draw, (0.1, 0.9)))
        boot.append(draw)
    out["J"] = ((out["new"][0] + out["ret"][0]) / 2, *np.quantile(np.mean(boot, axis=0), (0.1, 0.9)))
    return out
