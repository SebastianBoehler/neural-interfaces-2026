"""Label-free output changes shared by the review scorer and package patch."""

import numpy as np

DOSES = (0.0, 0.35, 0.70, 1.0)


def balance_output(raw, history, recognised, strength):
    if recognised or strength == 0.0:
        return raw
    history.append(np.asarray(raw, dtype=np.float64))
    if len(history) < 20:
        return raw
    p = np.maximum(history, 1e-12)
    scale = np.ones(3)
    for _ in range(12):
        q = p * scale
        q /= q.sum(axis=1, keepdims=True)
        mass = q.mean(axis=0)
        if np.max(np.abs(mass - 1 / 3)) <= 1e-5:
            break
        scale /= 3 * mass
        scale /= scale.max()
    log_scale = np.log(scale)
    return raw * np.exp(strength * np.clip(log_scale - log_scale.mean(), -0.35, 0.35))


def word_output(raw, team, manas, age, recognised, strength):
    if recognised or age < 41 or strength == 0.0:
        return raw
    word = 2
    rival = int(raw.argmax())
    if rival == word or team.argmax() != word or manas.argmax() != word:
        return raw
    team_odds = team[word] / max(team[rival], 1e-12)
    manas_odds = manas[word] / max(manas[rival], 1e-12)
    log_odds = np.clip(np.log(min(team_odds, manas_odds)), 0, 1)
    q = raw.copy()
    q[word] *= np.exp(strength * log_odds)
    return q
