"""Train a frozen six-person F25 teacher and emit only held-source embeddings."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, schedule, sha, split
from phase_source import F16_ACTOR, authenticated
from probe import diagnostics, fit, install
from produce_distill import publish
from relational import DIM, F25_ACTOR, HERE, TOL, ids, tag_for, teacher_rows, unit
from source_targets import normalizer


F14 = HERE.parent / "friday_s1c_training_20261118"
F16 = HERE.parent / "friday_s1c_distill_20261118"
F25 = HERE.parent / "friday_s1c_modulation_20261119"


@torch.no_grad()
def features(model, x, stream, rows, scale):
    from encoder312 import fingerprint

    model.eval()
    before = fingerprint(model)
    bank = np.full((len(rows), DIM), np.nan, dtype=np.float32)
    visits = [np.flatnonzero(stream[rows] == s) for s in dict.fromkeys(stream[rows])]
    maximum, first_checks = 0., 0
    for i, indices in enumerate(visits):
        raw = torch.as_tensor(x[rows[indices]] / scale, device="cuda")
        other = visits[(i + 1) % len(visits)]
        foreign = torch.as_tensor(x[rows[other[:10]]] / scale, device="cuda")
        complete = unit(model.representation(raw))
        prefix = unit(model.representation(raw[:10]))
        model.representation(foreign)
        if not torch.equal(prefix, unit(model.representation(raw[:10]))):
            raise ValueError("frozen teacher retained another source-person session")
        difference = float((complete[:len(prefix)] - prefix).abs().max())
        if difference > TOL or not np.isfinite(difference):
            raise ValueError("teacher representation depended on a future trial")
        maximum = max(maximum, difference)
        first_checks += 1
        bank[indices] = complete.cpu().numpy()
    if (not np.isfinite(bank).all() or np.max(abs(np.linalg.norm(bank, axis=1) - 1)) > TOL or
            fingerprint(model) != before):
        raise ValueError("incomplete, unnormalized or mutable teacher bank")
    return bank, dict(restarts=first_checks, prefix=maximum)


def run(held, seed, group, smoke):
    from encoder312 import fingerprint
    import braindecode
    import model322
    from source_labels import read_training_labels

    if (sha(F16 / "actor.sha256") != F16_ACTOR or sha(F25 / "actor.sha256") != F25_ACTOR or
            sha(F14 / "runtime.lock") != RUNTIME or sha(F14 / "original/source323.sha256") != ORIGIN_SHA or
            Path(model322.__file__).resolve() != F14 / "original/model322.py"):
        raise ValueError("pinned F16/F25 training ancestry differs")
    device = torch.cuda.get_device_name()
    if "H100" not in device or braindecode.__version__ != "1.8.1":
        raise ValueError("teacher requires exact H100 training platform")
    torch.set_num_threads(2)
    data = F14 / "proteus_windows.npz"
    if sha(data) != DATA_SHA:
        raise ValueError("official raw tensor differs")
    with np.load(data, allow_pickle=False) as z:
        eeg, subject, visit = (z[k] for k in ("X", "subject", "session"))
    person = ids(subject)
    stream = np.char.add(np.char.add(subject, "|"), visit)
    fit_rows, query = teacher_rows(person, held, group)
    src, src_stream, roster = split(subject, visit, held)
    ref, _ = authenticated(F16, F16, held, seed, smoke, src, src_stream, roster)
    batches, rows_sha = schedule(fit_rows, stream[fit_rows], seed, ref["epochs"])
    y = read_training_labels(data, fit_rows, len(eeg))
    d = dict(X=eeg[fit_rows], y=y, stream=stream[fit_rows])
    x, y, scale, mu, sd, weights, prepared = model322.setup(d, np.arange(len(fit_rows)), "cuda", epochs=ref["epochs"])
    if (x.shape[1:] != (41, 480) or not torch.isfinite(x).all() or
            any(not np.array_equal(a, b) for a, b in zip(
                (r for ep in batches for r in ep), (r for ep in prepared for r in ep), strict=True))):
        raise ValueError("six-person teacher model inputs or exact source work differs")
    model = model322.Factor("session", False, "cuda")
    normalizer(model)
    if fingerprint(model) != ref["initial"]:
        raise ValueError("teacher original zero-q state differs")
    sample = x[batches[0][0][:32]]
    layer = install(model, "multi", sample)
    stats = fit(model, layer, (x, y, mu, sd, weights, batches),
                dict(steps=sum(map(len, batches)), initial=ref["initial"]))
    activity = diagnostics(model, layer, sample, "multi")
    bank, guards = features(model, eeg, stream, query, scale)
    meta = dict(kind="multi", held=held, seed=seed, group=group, epochs=ref["epochs"], scale=scale, data=DATA_SHA,
                runtime=RUNTIME, device=device, origin=ORIGIN_SHA, initial=ref["initial"], actor=sha(HERE/"teacher.sha256"),
                f25=F25_ACTOR, source_people=sorted(set(person[fit_rows])), query_people=sorted(set(person[query])),
                source_rows=row_sha(fit_rows), query_rows=row_sha(query), batches=rows_sha, teacher_query_labels_opened=False,
                held_labels_opened=False, q_gradients_nonzero=0, **stats, **activity, **guards)
    stem = HERE / "teachers" / f"teacher_{tag_for(held, seed, smoke)}_{group}"
    publish(stem.with_suffix(".pt"), model, meta)
    with stem.with_suffix(".npz").open("xb") as out:
        np.savez_compressed(out, index=query, embedding=bank, meta=json.dumps(meta, sort_keys=True))
    side = dict(teacher=sha(stem.with_suffix(".pt")), receipt=sha(stem.with_suffix(".json")),
                bank=sha(stem.with_suffix(".npz")))
    with stem.with_suffix(".seal").open("x") as out:
        json.dump(side, out, sort_keys=True)
        out.write("\n")
    print(f"PASS F28 teacher {stem.name} {len(fit_rows)} source and {len(query)} unseen query rows", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", choices=PEOPLE)
    cli.add_argument("seed", type=int, choices=(0, 1))
    cli.add_argument("group", type=int, choices=(0, 1, 2))
    cli.add_argument("--smoke", action="store_true")
    a = cli.parse_args()
    run(a.held, a.seed, a.group, a.smoke)
