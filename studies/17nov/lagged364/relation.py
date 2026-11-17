"""Matched disjoint-channel relation blocks and a frozen-offset residual."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

LAG = 6
FREQUENCIES = np.fft.rfftfreq(480, 1 / 120)
BAND = (FREQUENCIES >= 8) & (FREQUENCIES < 30)
LIMIT = .5


def correlation(left, right):
    left = left - left.mean(axis=-1, keepdims=True)
    right = right - right.mean(axis=-1, keepdims=True)
    left /= np.linalg.norm(left, axis=-1, keepdims=True).clip(1e-12)
    right /= np.linalg.norm(right, axis=-1, keepdims=True).clip(1e-12)
    return (left @ right.transpose(0, 2, 1)).reshape(len(left), 400)


def blocks(windows):
    if windows.shape[1:] != (41, 480) or not np.isfinite(windows).all():
        raise ValueError('expected finite [N,41,480] EEG')
    pieces = []
    for batch in np.array_split(windows, (len(windows) + 63) // 64):
        freq = np.fft.rfft(batch[:, :40])
        freq[..., ~BAND] = 0
        filtered = np.fft.irfft(freq, n=480).astype(np.float64)
        even, odd = filtered[:, ::2], filtered[:, 1::2]
        pieces.append(np.stack([
            correlation(even[..., 3:-3], odd[..., 3:-3]),
            correlation(even[..., :-LAG], odd[..., LAG:]),
        ]))
    return np.concatenate(pieces, axis=1)


def adjust(base, effect):
    if base.shape != effect.shape or not np.isfinite(effect).all():
        raise ValueError('invalid residual')
    if not effect.any():
        return base.copy()
    return softmax(np.log(base.clip(1e-6)) + effect, axis=1)


def fit(features, y, base):
    mu = features.mean(axis=0)
    sd = features.std(axis=0).clip(.05)
    z = (features - mu) / sd
    targets = np.eye(3)[y]
    weights = 1 / (3 * targets.sum(axis=0)[y])
    offset = np.log(base.clip(1e-6))

    def objective(values):
        w = values.reshape(400, 3)
        t = np.tanh(z @ w / LIMIT)
        effect = LIMIT * t
        out = softmax(offset + effect, axis=1)
        loss = -(weights * np.log((out * targets).sum(axis=1))).sum()
        gradient = z.T @ ((out - targets) * (1 - t * t) * weights[:, None])
        return loss + .05 * np.square(w).sum(), (gradient + .1 * w).ravel()

    initial = np.zeros(1200)
    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                      options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-7})
    if (not result.success or not np.isfinite(result.fun)
            or result.fun >= objective(initial)[0] or not np.linalg.norm(result.x)):
        raise ValueError(f'residual fit failed: {result.message}')
    return mu, sd, result.x.reshape(400, 3), result.nit


def test():
    from scipy.signal import chirp
    t = np.arange(480) / 120
    x = np.zeros((1, 41, 480), dtype=np.float32)
    x[0, 0] = chirp(t, 10, t[-1], 24)
    x[0, 1] = np.roll(x[0, 0], LAG)
    q = blocks(x)
    if q.shape != (2, 1, 400) or q[1, 0, 0] <= .98 or abs(q[0, 0, 0]) >= .5:
        raise ValueError('known delayed disjoint-channel pair not detected')
    if np.count_nonzero(q) != 2:
        raise ValueError('relation entries escaped the disjoint pair')
    p = np.array([[.03, .12, .85]])
    if not np.array_equal(adjust(p, np.zeros_like(p)), p):
        raise ValueError('zero residual changed frozen inputs')


if __name__ == '__main__':
    test()
    print('OK: matched shape and delayed cross-channel signal, bitwise residual-off')
