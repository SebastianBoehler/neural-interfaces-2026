"""Produce the label-blind teacher bank and a schedule-matched within-class control."""

import numpy as np
import torch

from anchors import row_sha


TOL = 2e-5


def raw_logits(model, x):
    features = model.representation(x)
    return model.task(torch.cat((features, features.new_zeros((len(features), 287))), 1))


def normalizer(model):
    from session_bn import SessionBN

    layers = [layer for layer in model.body.modules() if type(layer) is SessionBN]
    if len(layers) != 1 or (layers[0].m, layers[0].k, layers[0].k_train, layers[0].causal) != (32, 20., 20., False):
        raise ValueError("authentic §323 S1C SessionBN path differs")
    return layers[0]


@torch.no_grad()
def teacher_bank(path, meta, x, rows, stream):
    from encoder312 import fingerprint
    import model322

    model322.SEED = meta["seed"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if any(payload[k] != meta[k] for k in ("kind", "held", "seed", "scale")):
        raise ValueError("checkpoint payload does not identify the F15 teacher")
    actor = model322.Factor("session", True, "cuda")
    actor.load_state_dict(payload["state"], strict=True)
    actor.eval()
    normalizer(actor)
    before = fingerprint(actor)
    if before != meta["state"] or not actor.skip:
        raise ValueError("teacher is not the frozen original full-q F15 checkpoint")
    logits = np.full((len(x), 3), np.nan, dtype=np.float32)
    groups = {name: np.flatnonzero(stream == name) for name in dict.fromkeys(stream)}
    source_streams = list(groups)
    trace, checks, prefix_error = {}, 0, 0.
    for i, (name, idx) in enumerate(groups.items()):
        other = groups[source_streams[(i + 1) % len(groups)]]
        sequence = torch.as_tensor(x[idx] / meta["scale"], device="cuda")
        prefix = actor(sequence[:10])
        all_rows = actor(sequence)
        native = all_rows[:10]
        torch.testing.assert_close(prefix, native, rtol=TOL, atol=TOL)
        if not torch.equal(prefix.argmax(1), native.argmax(1)):
            raise ValueError("later source windows changed a teacher prefix decision")
        actor(torch.as_tensor(x[other[:10]] / meta["scale"], device="cuda"))
        if not torch.equal(prefix, actor(sequence[:10])):
            raise ValueError("teacher carried state across source sessions")
        trace[name] = row_sha(rows[idx])
        checks += 1
        prefix_error = max(prefix_error, float((prefix - native).abs().max()))
        logits[idx] = all_rows.cpu().numpy()
    if not np.isfinite(logits).all() or fingerprint(actor) != before or len(trace) < 9:
        raise ValueError("incomplete or non-frozen teacher source coverage")
    return logits, dict(sessions=trace, restarted=checks, prefix_max_abs=prefix_error)


def donor_plan(batches, rows, stream, y, held, seed):
    """Preserve padded training exposures, and forbid same-recording-row donors."""
    rng = np.random.default_rng(16016 + 100 * int(held[1:]) + seed)
    source, donor, group_counts = [], [], []
    for batches_this_epoch in batches:
        canonical = np.concatenate(batches_this_epoch)
        controls = np.full(len(canonical), -1)
        count = 0
        for name in np.unique(stream):
            for target in (0, 1, 2):
                group = np.flatnonzero((stream[canonical] == name) & (y[canonical] == target))
                if not len(group):
                    continue
                observed = canonical[group]
                unique, reverse, frequency = np.unique(observed, return_inverse=True, return_counts=True)
                shift = int(frequency.max())
                if 2 * shift > len(group):
                    raise ValueError("source session/class exposure cannot be deranged")
                ordered = np.argsort(rng.permutation(len(unique))[reverse], kind="stable")
                indexes = group[ordered]
                controls[indexes] = np.roll(canonical[indexes], shift)
                if np.any(controls[group] == observed) or not np.array_equal(np.sort(controls[group]), np.sort(observed)):
                    raise ValueError("incorrect within-group derangement or padded marginal")
                count += 1
        if (controls < 0).any() or not np.array_equal(y[canonical], y[controls]) or not np.array_equal(
                stream[canonical], stream[controls]):
            raise ValueError("derangement crossed a source class or session")
        source.append(rows[canonical])
        donor.append(rows[controls])
        group_counts.append(count)
    return np.stack(source), np.stack(donor), group_counts
