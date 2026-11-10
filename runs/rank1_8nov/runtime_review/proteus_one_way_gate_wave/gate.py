"""Source confidence fitting and challenge-metric scoring for one-way switches."""

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.metrics import balanced_accuracy, group_scores

THRESHOLDS = (.60, .65, .70, .75, .80, .85, .90, .95)


def eligibility(base):
    return (base[:, 2] > base[:, 1]) & (base[:, 1] > base[:, 0])


def describe(base, csp):
    sub, word = np.maximum(base[:, 1], 1e-12), np.maximum(base[:, 2], 1e-12)
    return np.column_stack((np.log(sub) - np.log(word),
                            logit(csp["stable_csp"].clip(1e-6, 1 - 1e-6)),
                            logit(csp["pooled_csp"].clip(1e-6, 1 - 1e-6))))


def fit_confidence(base, features, labels, person, interface):
    rows = eligibility(base) & np.isin(labels, [1, 2])
    counts = [int(np.count_nonzero(rows & (labels == label))) for label in (1, 2)]
    if min(counts) < 5 or len(np.unique(person[rows])) < 3:
        raise ValueError(f"Insufficient eligible subtraction/word source rows: {counts}")
    _, group, sizes = np.unique(np.column_stack((person, interface, labels)), axis=0,
                                return_inverse=True, return_counts=True)
    weight = 1.0 / sizes[group[rows]]
    weight *= rows.sum() / weight.sum()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.10, max_iter=500))
    model.fit(features[rows], labels[rows], standardscaler__sample_weight=weight,
              logisticregression__sample_weight=weight)
    if model[-1].classes_.tolist() != [1, 2] or model[-1].n_iter_.max() >= model[-1].max_iter:
        raise ValueError("Invalid source confidence fit")
    return model, {"fit_count": counts, "eligible_people": np.unique(person[rows]).tolist()}


def decide(base, confidence, threshold):
    if (confidence.shape != (len(base),) or not np.all((0 <= confidence) & (confidence <= 1))):
        raise ValueError("Expected one finite confidence per baseline row")
    labels = base.argmax(axis=1)
    if threshold is None:
        return labels
    return np.where(eligibility(base) & (confidence >= threshold), 1, labels)


def official_section(meta, rows, before, after):
    true = meta["y"][rows]
    context = meta["interface"][rows]
    people = np.array([name.rsplit("/", 1)[-1] for name in meta["subject"][rows]])
    cells = np.column_stack((meta["stream"][rows], context))
    scopes = {"overall": np.ones(len(rows), dtype=bool),
              **{k: context == k for k in ("BrainHero", "Graz") if np.any(context == k)}}
    if "P10" in people:
        scopes["P10"] = people == "P10"
    result = {}
    for scope, at in scopes.items():
        a, b = [100 * float(group_scores(balanced_accuracy, true[at], p[at], cells[at]).mean())
                for p in (before, after)]
        changed = at & (before != after)
        result[scope] = {"baseline": a, "candidate": b, "delta": b-a, "switches": int(changed.sum()),
                         "rescues": int(np.sum(changed & (true == 1))),
                         "harms": int(np.sum(changed & (true == 2))),
                         "neither": int(np.sum(changed & (true == 0)))}
    return result


def choose(meta, rows, forecasts):
    people = meta["subject"][rows]
    grid = []
    for threshold in THRESHOLDS:
        by_seed, switches = {}, []
        for seed, (base, confidence) in forecasts.items():
            labels = base.argmax(1)
            new = decide(base, confidence, threshold)
            switches.append(labels != new)
            by_seed[str(seed)] = official_section(meta, rows, labels, new)
        any_switch = np.any(switches, axis=0)
        means = {k: float(np.mean([x[k]["delta"] for x in by_seed.values()]))
                 for k in ("overall", "BrainHero", "Graz")}
        checks = {"overall_at_least_0p25": means["overall"] >= .25,
                  "brainhero_at_least_0p15": means["BrainHero"] >= .15,
                  "graz_nonnegative": means["Graz"] >= -1e-12,
                  "two_positive_seeds": sum(x["overall"]["delta"] > 0 for x in by_seed.values()) >= 2,
                  "at_least_8_distinct_switches": int(any_switch.sum()) >= 8,
                  "at_least_3_people_switch": len(np.unique(people[any_switch])) >= 3}
        grid.append({"threshold": threshold, "official_mean_deltas": means, "by_seed": by_seed,
                     "distinct_switches": int(any_switch.sum()), "people_switching": len(np.unique(people[any_switch])),
                     "checks": checks, "pass": all(checks.values())})
    viable = [x for x in grid if x["pass"]]
    chosen = max(viable, key=lambda x: (round(x["official_mean_deltas"]["overall"], 12), x["threshold"])) if viable else None
    return (chosen["threshold"] if chosen is not None else None), grid
