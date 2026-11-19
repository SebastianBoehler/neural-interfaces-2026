"""Observe the live CSP likelihood without changing its result or fit."""

import hashlib

import numpy as np

from tails import personalize


def capture(final, windows, raw, filtered, rows):
    factory = final.CP.CausalPers
    probs = np.zeros((len(rows), 3))
    valid = np.zeros(len(rows), dtype=bool)
    activity = []
    feature_hash = hashlib.sha256()

    def observe(config):
        actor = factory(config)
        predict, feats = actor.predict, actor.feats
        assert (actor.cap, actor.R, actor.w0) == (.10, 120, 90)

        def read_likelihood(x):
            q = predict(x)
            age = actor.t - 1
            activity.append(age)
            probs[age] = q
            valid[age] = actor.model is not None
            return q

        def read_features(x):
            c = feats(x)
            feature_hash.update(np.asarray(c, dtype="<f8", order="C").tobytes())
            return c

        actor.predict, actor.feats = read_likelihood, read_features
        return actor

    final.CP.CausalPers = observe
    try:
        scores, proof = personalize(final, windows, raw, filtered, rows)
    finally:
        final.CP.CausalPers = factory
    assert activity == list(range(90, len(rows)))
    assert np.isfinite(probs).all() and (probs >= 0).all()
    np.testing.assert_allclose(probs[90:].sum(1), 1., rtol=0, atol=1e-14)
    w = .10 * np.clip((np.arange(90, len(rows)) - 89) / 120, 0., 1.)
    expected = np.log(np.clip(filtered[rows[90:]], 1e-6, 1.)) + w[:, None] * np.log(np.clip(probs[90:], 1e-6, 1.))
    np.testing.assert_array_equal(expected, scores[90:])
    return dict(scores=scores, q=probs, valid=valid), dict(
        **proof, covariance_sha=feature_hash.hexdigest(), real_model=int(valid.sum()))
