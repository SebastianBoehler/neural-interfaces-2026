"""Build enrolled references from the latest visit allowed by each fold's roster."""
from hashlib import sha256

import numpy as np
import torch

from settings import D


def aligners(model):
    parts = [s for s, _ in model.nets.groups if s.align]
    if model.rest.stream.align:
        parts.append(model.rest.stream)
    return parts


def latest_visits(data, gallery, fold):
    mode, held = fold.split("_")
    people = D.PEOPLE if mode == "ret" else [p for p in D.PEOPLE if p != held]
    first_only = [held] if mode == "ret" else []
    if set(gallery) != set(people) or len(gallery) != len(people):
        raise ValueError(f"{fold} route gallery differs from the allowed roster")
    train = D.train_rows(data, people, first_only)
    for person in gallery:
        own = train[data["person"][train] == person]
        visit = str(max(map(int, data["session"][own])))
        rows = own[data["session"][own] == visit]
        if not len(rows) or (person == held and (mode != "ret" or visit != "1")):
            raise ValueError(f"invalid enrollment for {fold}/{person}/{visit}")
        yield person, visit, rows


@torch.no_grad()
def compute(model, data, fold):
    result = {}
    cov_inputs = aligners(model)
    reps = {part.shrink: part for part in cov_inputs}
    for j, (person, visit, rows) in enumerate(latest_visits(data, model.round3_names, fold)):
        x = data["X"][rows]
        mats = []
        for core in (model.core1, model.core2):
            logs = [core.round3_cov(w - w.mean(0, keepdims=True))[1] for w in x]
            mats.append(np.mean(logs, axis=0))
        arithmetic = {k: np.mean([part.round3_cov(w) for w in x], axis=0)
                      for k, part in reps.items()}
        mats.extend(arithmetic[part.shrink] for part in cov_inputs)
        team = [model.team_b.round3_enrol_cov(torch.from_numpy(w)).cpu().numpy() for w in x]
        mats.append(np.mean(team, axis=0))
        assert all(a.shape[-2:] == (41, 41) and np.isfinite(a).all() for a in mats)
        result.update({f"{j}_{i}": a for i, a in enumerate(mats)})
        result[f"visit_{j}"] = visit
        result[f"rows_sha_{j}"] = sha256(rows.tobytes()).hexdigest()
        print(f"{fold} source {person} visit {visit} {len(rows)} windows", flush=True)
    return result


def attach(model, tape):
    names = tuple(tape["names"])
    if names != model.round3_names:
        raise ValueError("enrolled reference names do not match model gallery")
    count = len(aligners(model)) + 3
    model.round3_bank = {j: tuple(tape[f"{j}_{i}"] for i in range(count)) for j in range(len(names))}
    assert all(np.isfinite(a).all() for mats in model.round3_bank.values() for a in mats)
