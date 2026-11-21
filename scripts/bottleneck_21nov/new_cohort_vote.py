"""Causal terminal vote in a space of already computed model outputs."""
import numpy as np

START_WINDOW = 30
GRID = tuple((k, w) for w in (0.3, 0.6) for k in (7, 15))


def neighbour_log_scores(past_features, query, past_terminal, terminal, k, weight):
    center = past_features.mean(axis=0)
    centered = past_features - center
    query = query - center
    norm = np.linalg.norm(centered, axis=1) * np.linalg.norm(query)
    similarity = np.divide(centered @ query, norm, out=np.zeros_like(norm), where=norm > 0)
    nearest = np.argsort(-similarity, kind="stable")[:k]
    vote = np.maximum(past_terminal[nearest].mean(axis=0), 1e-8)
    return np.log(np.maximum(terminal, 1e-8)) + weight * np.log(vote)


def vote(features, terminal, recognised, k, weight):
    scores = np.log(np.maximum(terminal, 1e-8))
    for t in range(START_WINDOW - 1, len(terminal)):
        if not recognised[t]:
            scores[t] = neighbour_log_scores(features[:t], features[t], terminal[:t], terminal[t], k, weight)
    return scores


def count_terminal(scores, bypass):
    """Reapply the captured CSP evidence with a different count argument."""
    routed, count, csp = scores["own"], scores["count"], scores["csp"]
    select = bypass & np.any(count != routed, axis=1)
    changed = csp.copy()
    inactive = select & ~scores["csp_active"]
    changed[inactive] = routed[inactive] / routed[inactive].sum(axis=1, keepdims=True)
    active = select & scores["csp_active"]
    evidence = csp[active] * (routed[active].clip(1e-6, 1) / count[active].clip(1e-6, 1))
    changed[active] = evidence / evidence.sum(axis=1, keepdims=True)
    return 0.9 * changed + 0.1 * scores["manas"]
