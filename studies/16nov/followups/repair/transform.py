"""Fixed label-free temperature and strictly-prior session centering."""
import numpy as np


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def apply(p, sessions, mode, value):
    p, sessions = np.asarray(p), np.asarray(sessions)
    if p.ndim != 2 or p.shape[1] != 3 or sessions.shape != (len(p),): raise ValueError('invalid shape')
    if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError('repair input must be strictly positive probabilities')
    if mode == 'temp':
        if value not in (.5, 1., 2.): raise ValueError('invalid registered temperature')
        out = p.copy() if value == 1 else softmax(np.log(p) / value)
        if not np.array_equal(out.argmax(1), p.argmax(1)): raise ValueError('temperature changed argmax')
        return out
    if mode != 'center' or value not in (0., .25, .5): raise ValueError('invalid registered centering')
    if value == 0: return p.copy()
    out = p.copy(); seen = set(); start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sessions[end] == sessions[start]: end += 1
        key = str(sessions[start])
        if key in seen: raise ValueError('noncontiguous session')
        seen.add(key); logs = np.log(p[start:end]); n = end - start
        if n > 20:
            prefix = np.cumsum(logs, axis=0)[:-1] / np.arange(1, n)[:, None]
            centered = prefix - prefix.mean(axis=1, keepdims=True)
            out[start+20:end] = softmax(logs[20:] - value * centered[19:])
        start = end
    return out


def check_causal(p, sessions, value):
    out = apply(p, sessions, 'center', value)
    cuts = np.flatnonzero(sessions[1:] != sessions[:-1]) + 1
    bounds = np.r_[0, cuts, len(p)]
    eligible = 0; tested = False
    for start, end in zip(bounds[:-1], bounds[1:]):
        np.testing.assert_array_equal(out[start:min(start+20,end)], p[start:min(start+20,end)])
        separate = apply(p[start:end], sessions[start:end], 'center', value)
        np.testing.assert_array_equal(out[start:end], separate)
        if end-start > 20:
            eligible += end-start-20
        if end-start > 22 and not tested:
            cut = start + 21; altered = p.copy(); altered[cut:end] = altered[cut:end, ::-1]
            np.testing.assert_array_equal(apply(altered, sessions, 'center', value)[:cut], out[:cut])
            tested = True
    if not eligible or not tested or np.array_equal(out, p): raise ValueError('centering has no eligible changed rows')
    return out
