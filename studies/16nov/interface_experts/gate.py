"""Nested person-disjoint source-only interface gating."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def strata_weights(people, labels, interface):
    people, labels, interface = map(np.asarray, (people, labels, interface))
    if not (len(people) == len(labels) == len(interface)) or not len(people):
        raise ValueError('invalid source metadata')
    w = np.empty(len(people), dtype=np.float64)
    for person in np.unique(people):
        for label in range(3):
            for domain in (0, 1):
                mask = (people == person) & (labels == label) & (interface == domain)
                if not mask.any(): raise ValueError(f'missing source stratum {person}/{label}/{domain}')
                w[mask] = 1. / mask.sum()
    w *= len(w) / w.sum()
    return w


def train_gate(x, people, labels, interface):
    w = strata_weights(people, labels, interface)
    scaler = StandardScaler().fit(x, sample_weight=w)
    model = LogisticRegression(C=.1, max_iter=1000, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(scaler.transform(x), interface, sample_weight=w)
    return scaler, model


def logits(model, x):
    scaler, classifier = model
    return classifier.decision_function(scaler.transform(x))


def calibrator(scores, people, labels, interface):
    w = strata_weights(people, labels, interface)
    model = LogisticRegression(C=1., max_iter=1000, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(np.asarray(scores).reshape(-1, 1), interface, sample_weight=w)
    return model


def nested(x, people, labels, interface, held_person):
    """Return source OOF gates and a deployed gate; no held person appears in x."""
    x, people, labels, interface = map(np.asarray, (x, people, labels, interface))
    ids = sorted(set(people)); n = len(x)
    if len(ids) != 9 or held_person in ids or x.shape != (n, 246):
        raise ValueError('expected nine source people excluding outer held person')
    oof, raw, traces = np.full(n, np.nan), np.full(n, np.nan), {}
    for person in ids:
        target = people == person; fit = ~target
        raw[target] = logits(train_gate(x[fit], people[fit], labels[fit], interface[fit]), x[target])
        inner = np.flatnonzero(fit); scores = np.full(len(inner), np.nan); inner_people = people[inner]
        folds = {}
        for other in ids:
            if other == person: continue
            validate = inner_people == other; train = ~validate
            fold_ids = sorted(set(inner_people[train]))
            if person in fold_ids or other in fold_ids: raise ValueError('nested fold leakage')
            scores[validate] = logits(train_gate(x[inner[train]], inner_people[train], labels[inner[train]],
                                                 interface[inner[train]]), x[inner[validate]])
            folds[str(other)] = fold_ids
        cal = calibrator(scores, inner_people, labels[inner], interface[inner])
        oof[target] = cal.predict_proba(raw[target, None])[:, 1]
        traces[str(person)] = dict(gate_fit_people=sorted(set(people[fit])),
                                   calibration_people=sorted(set(inner_people)), inner_fits=folds)
    if not np.isfinite(oof).all() or not np.isfinite(raw).all():
        raise ValueError('incomplete source gate crossfit')
    final = train_gate(x, people, labels, interface)
    final_cal = calibrator(raw, people, labels, interface)
    return oof, (final, final_cal), traces


def predict(model, x):
    base, cal = model
    p = cal.predict_proba(logits(base, x).reshape(-1, 1))[:, 1]
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('invalid gate probability')
    return p
