"""Untouched sklearn fit, observing its actual SciPy optimizer result."""
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression, _logistic


def fit(x, y):
    original, recorded = _logistic.optimize.minimize, []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        recorded.append(dict(nfev=int(result.nfev), njev=int(result.njev), nit=int(result.nit),
                             message=str(result.message), options=kwargs['options']))
        return result
    with patch.object(_logistic.optimize, 'minimize', observe):
        model = LogisticRegression(C=.3, max_iter=2000, tol=1e-5).fit(x, y)
    if len(recorded) != 1:
        raise ValueError('expected one multinomial reference optimizer')
    return model, recorded[0]
