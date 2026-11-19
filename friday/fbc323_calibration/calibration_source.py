"""Bind F16 and F17 source states and the independently reconstructed pairing tape."""
from anchors import DATA_SHA, ORIGIN_SHA, RUNTIME, row_sha, sha, split
from calibration import HERE, QUERY, STRENGTH, plan
from hierarchy_source import F17_BANK, controls
from phase_source import F16_BANK, authenticated


PARENT_BANKS = dict(hard=F16_BANK, jitter=F17_BANK)


def parent_roots(independent):
    prefix = "ni26_friday_s1c_" if independent else "friday_s1c_"
    return dict(hard=HERE.parent / (prefix + "distill_20261118"),
                jitter=HERE.parent / (prefix + "phase_20261118"))


def context(held, seed, smoke, subject, visit, independent):
    roots = parent_roots(independent)
    folders = {k: p / "checkpoints" if independent else p for k, p in roots.items()}
    source, streams, people = split(subject, visit, held)
    refs, batches, direction, masks, jitter_sha = controls(
        roots["jitter"], folders["jitter"], held, seed, smoke, source, streams, people)
    anchor, _ = authenticated(roots["hard"], folders["hard"], held, seed, smoke, source, streams, people)
    if (anchor["checkpoint"] != refs["hard"]["checkpoint"] or
            any(sha(roots[k] / "before_labels.sha256") != digest for k, digest in PARENT_BANKS.items() if independent)):
        raise ValueError("exact F16 and F17 source or blind bank controls differ")
    entries, selections, picked, context_sha = plan(batches, streams)
    proof = dict(held=held, seed=seed, epochs=anchor["epochs"], scale=anchor["scale"], initial=anchor["initial"],
                 batches=anchor["batches"], source_rows=row_sha(source), source_people=people,
                 actor=sha(HERE / "actor.sha256"), data=DATA_SHA, origin=ORIGIN_SHA, runtime=RUNTIME,
                 train_rows_seen=sum(map(len, masks)), jitter_sha=jitter_sha, context_sha=context_sha,
                 diagnostic_steps=picked.tolist(), diagnostic_rows=row_sha([source[entries[i]] for i in picked]),
                 parents={k: m["checkpoint"] for k, m in refs.items()}, rng_isolated=True,
                 held_labels_opened=False, torch=anchor["torch"], braindecode=anchor["braindecode"],
                 q_gradients_nonzero=0, parameters=347903)
    return refs, source, streams, batches, direction, masks, entries, selections, picked, proof


def arm_fields(kind, selections, masks):
    augmented = kind != "hard"
    adaptive = kind in ("within", "cross")
    steps = sum(selections) if adaptive else 0
    return dict(kind=kind, strength=STRENGTH if adaptive else 0., aux_steps=steps, aux_queries=2 * QUERY * steps,
                perturbed_rows=sum(int(m.sum()) for m in masks) if augmented else 0,
                feed_calls=len(masks) if augmented else 0)
