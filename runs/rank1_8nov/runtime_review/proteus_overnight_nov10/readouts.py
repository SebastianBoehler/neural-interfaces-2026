"""Source-fold readouts with identical fit and score samples across arms."""

import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectral import fit_head


def logits(x, draw, held, y):
    return fit_head(x[draw], y[draw]).decision_function(x[held])


def ordinary(bank, draw, held, y, people, interface, seed, fold):
    return {key: softmax(logits(x, draw, held, y), axis=1) for key, x in bank.items()}


def pairwise(bank, draw, held, y, people, interface, seed, fold):
    result = {}
    for name, x in bank.items():
        result[f"{name}_multi"] = softmax(logits(x, draw, held, y), axis=1)
        combined = np.zeros((int(held.sum()), 3))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            fit = draw[(y[draw] == a) | (y[draw] == b)]
            head = make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=500))
            head.fit(x[fit], y[fit])
            classifier = head[-1]
            if classifier.classes_.tolist() != [a, b] or classifier.n_iter_.max() >= classifier.max_iter:
                raise ValueError("Invalid fitted class-pair classifier")
            log_odds = head.decision_function(x[held])
            combined[:, a] -= log_odds
            combined[:, b] += log_odds
        result[f"{name}_pair"] = softmax(combined / 3, axis=1)
    return result


def prototype_logits(train, labels, x):
    means = np.stack([train[labels == label].mean(axis=0) for label in range(3)])
    residual = train - means[labels]
    variance = .5 + .5 * np.mean(residual**2, axis=0)
    return (x / variance) @ means.T - .5 * ((means**2) / variance).sum(axis=1)


def center_rms(z):
    z = z - z.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(z*z))


def prototypes(bank, draw, held, y, people, interface, seed, fold):
    result = {}
    for key, x in bank.items():
        linear = fit_head(x[draw], y[draw])
        scaler = StandardScaler().fit(x[draw])
        train, test = scaler.transform(x[draw]), scaler.transform(x[held])
        raw = prototype_logits(train, y[draw], test)
        scale = center_rms(linear.decision_function(x[draw])) / max(
            center_rms(prototype_logits(train, y[draw], train)), 1e-8)
        result[f"{key}_template"] = softmax(raw * scale, axis=1)
        result[f"{key}_logistic"] = softmax(linear.decision_function(x[held]), axis=1)
    return result


def shuffle_tiles(x, seed):
    order = np.argsort(np.random.default_rng(seed).random(x.shape[:2]), axis=1)
    return np.take_along_axis(x, order[..., None], axis=1)


def markov_scores(train, test, y, n_states, ordered):
    result = []
    for label in range(3):
        z = train[y == label]
        state = np.bincount(z.ravel(), minlength=n_states) + 1
        state = np.log(state / state.sum())
        if ordered:
            count = np.bincount((z[:, :-1] * n_states + z[:, 1:]).ravel(), minlength=n_states**2)
            count = count.reshape(n_states, n_states) + 1
            transition = np.log(count / count.sum(axis=1, keepdims=True))
            score = state[test[:, 0]] + transition[test[:, :-1], test[:, 1:]].sum(axis=1)
        else:
            score = state[test].sum(axis=1)
        result.append(score / np.sqrt(test.shape[1]))
    return softmax(np.column_stack(result), axis=1)


def markov(bank, draw, held, y, people, interface, seed, fold):
    x = bank["tiles"]
    n, steps, width = x.shape
    scaler = StandardScaler().fit(x[draw].reshape(-1, width))
    normalized = scaler.transform(x.reshape(-1, width))
    centers = KMeans(n_clusters=8, n_init=1, max_iter=50, random_state=seed+fold)
    centers.fit(normalized.reshape(n, steps, width)[draw].reshape(-1, width))
    state = centers.predict(normalized).reshape(n, steps)
    shuffled = shuffle_tiles(state[..., None], seed+fold+38)[..., 0]
    return {"ordered": markov_scores(state[draw], state[held], y[draw], 8, True),
            "bag": markov_scores(state[draw], state[held], y[draw], 8, False),
            "shuffled": markov_scores(shuffled[draw], shuffled[held], y[draw], 8, True),
            "pooled": softmax(logits(x.mean(axis=1), draw, held, y), axis=1)}
