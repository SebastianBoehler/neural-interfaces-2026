"""Matched cognitive-pair CSP filters, with and without context-stable selection."""

import hashlib

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARMS = ("stable_csp", "pooled_csp")
PAIR_WEIGHT = .30


def bootstrap_cells(indices, labels, people, interfaces, seed, fold):
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
    cells = [indices[(people[indices] == person) & (interfaces[indices] == context)
                     & (labels[indices] == label)]
             for person in np.unique(people[indices])
             for context in ("BrainHero", "Graz")
             for label in (1, 2)]
    if not cells or any(len(cell) == 0 for cell in cells):
        raise ValueError("Missing subtraction or word examples in a source interface cell")
    return tuple(rng.choice(cell, size=len(cell), replace=True) for cell in cells)


def log_variance(cov, filters):
    variances = ((cov @ filters) * filters).sum(axis=-2)
    return np.log(variances.clip(1e-8))


def fit(cov, labels, cells):
    rows = np.concatenate(cells)
    sizes = np.array([len(cell) for cell in cells])
    splits = np.cumsum(sizes)[:-1]
    classes = np.array([cov[rows].mean(axis=0, dtype=np.float64) for rows in cells])
    context_means = classes.reshape(-1, 2, 2, *cov.shape[1:])
    pooled = context_means.mean(axis=(0, 1))
    filters = {arm: [] for arm in ARMS}
    scores = []
    for band, (subtraction, word) in enumerate(zip(*pooled, strict=True)):
        _, vectors = eigh(subtraction - word, (subtraction + word) * .5)
        c = vectors.shape[1]
        trial_power = log_variance(cov[rows, band], vectors)
        means = np.array([block.mean(axis=0) for block in np.split(trial_power, splits)]).reshape(-1, 2, 2, c)
        difference = np.median(means[:, :, 0] - means[:, :, 1], axis=0)
        lower = np.arange(c // 2)
        upper = np.arange(c // 2, c)
        word = np.min(-difference[:, lower], axis=0)
        sub = np.min(difference[:, upper], axis=0)
        stable = np.r_[lower[np.argsort(-word, kind="stable")[:2]],
                       upper[np.argsort(-sub, kind="stable")[:2]]]
        extreme = np.r_[0, 1, c-2, c-1]
        for arm, chosen in zip(ARMS, (stable, extreme), strict=True):
            filters[arm].append(vectors[:, chosen])
        scores.append({"stable_indices": stable.tolist(), "pooled_indices": extreme.tolist()})
    # Equal total mass in each person/interface/class cell, for both arms.
    weights = np.repeat(len(rows) / (len(cells) * sizes), sizes)
    model = {}
    for arm, axes in filters.items():
        axes = np.stack(axes)
        x = log_variance(cov[rows], axes).reshape(len(rows), -1)
        head = make_pipeline(StandardScaler(), LogisticRegression(C=.10, max_iter=500))
        head.fit(x, labels[rows], standardscaler__sample_weight=weights,
                 logisticregression__sample_weight=weights)
        if head[-1].classes_.tolist() != [1, 2] or head[-1].n_iter_.max() >= head[-1].max_iter:
            raise ValueError("Invalid cognitive-pair readout")
        model[arm] = (axes, head)
    digest = hashlib.sha256(rows.tobytes()).hexdigest()
    return model, {"bootstrap_sha256": digest, "selected_filters": scores}


def predict(models, cov):
    return {arm: head.predict_proba(log_variance(cov, axes).reshape(len(cov), -1))[:, 0]
            for arm, (axes, head) in models.items()}


def digest(models):
    h = hashlib.sha256()
    for axes, (scaler, head) in models.values():
        for x in (axes, scaler.mean_, scaler.scale_, head.coef_, head.intercept_, head.classes_):
            h.update(x.tobytes())
    return h.hexdigest()


def mix(base, subtraction):
    if subtraction.shape != (len(base),) or not np.all((0 <= subtraction) & (subtraction <= 1)):
        raise ValueError("Expected aligned finite cognitive probabilities")
    eligible = base[:, 0] < base[:, 1:].min(axis=1)
    post = base.copy()
    pair_mass = base[eligible, 1:].sum(axis=1)
    post[eligible, 1] = (1 - PAIR_WEIGHT) * base[eligible, 1] + PAIR_WEIGHT * pair_mass * subtraction[eligible]
    post[eligible, 2] = pair_mass - post[eligible, 1]
    if not np.array_equal(base.argmax(1) == 0, post.argmax(1) == 0):
        raise ValueError("Cognitive-pair correction changed a motor-imagery decision")
    return post
