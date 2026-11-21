"""Independent multiband EEG geometry for the frozen terminal mixture."""
import numpy as np


class SpectralGeometry:
    CHANNELS = ("Fz", "FC3", "FC4", "C3", "C4", "Cz", "CP3", "CP4", "CPz",
                "P3", "P4", "Pz", "PO7", "PO8")
    BANDS = ((2, 4), (4, 8), (8, 12), (12, 17), (17, 25), (25, 35), (35, 45))
    PRIOR = 12
    CENTER = 0.75

    def __init__(self, channels):
        self.indices = [list(channels).index(name) for name in self.CHANNELS]
        self.upper = np.triu_indices(len(self.indices), 1)
        self.identity = np.eye(len(self.indices))
        self.taper = np.hanning(240)
        freq = np.fft.rfftfreq(240, 1 / 120)
        self.bins = [np.flatnonzero((freq >= lo) & (freq < hi)) for lo, hi in self.BANDS]

    def features(self, raw):
        if raw.shape != (41, 480):
            raise ValueError("spectral geometry requires a native EEG window")
        x = raw.astype(np.float64)
        x -= x.mean(axis=0)
        segments = np.stack((x[:, :240], x[:, 120:360], x[:, 240:]))
        segments -= segments.mean(axis=-1, keepdims=True)
        spectrum = np.fft.rfft(segments * self.taper, axis=-1)
        power = np.abs(spectrum) ** 2
        bands = np.stack([power[..., bins].mean(axis=-1) for bins in self.bins])
        relative = np.log(np.maximum(bands, 1e-8))
        relative -= np.log(np.maximum(bands.mean(axis=0, keepdims=True), 1e-8))
        spatial = []
        for bins in self.bins:
            z = spectrum[:, self.indices][:, :, bins].transpose(1, 0, 2).reshape(len(self.indices), -1)
            cross = z @ z.conj().T
            scale = np.sqrt(np.maximum(cross.diagonal().real, 1e-8))
            corr = cross / (scale[:, None] * scale)
            eig, vec = np.linalg.eigh(0.85 * corr.real + 0.15 * self.identity)
            log = (vec * np.log(np.maximum(eig, 0.15))) @ vec.T
            spatial.extend((log[self.upper], corr.imag[self.upper]))
        return np.concatenate([relative.mean(axis=1).ravel(),
                               (relative[:, -1, self.indices] - relative[:, 0, self.indices]).ravel(),
                               *spatial])


class SpectralMember:
    def __init__(self, path, channels):
        self.geometry = SpectralGeometry(channels)
        with np.load(path, allow_pickle=False) as z:
            self.center, self.scale, self.weight, self.bias = (z[k] for k in ("center", "scale", "weight", "bias"))
            if tuple(z["channels"]) != tuple(channels):
                raise ValueError("spectral montage differs from its fit")
        n = self.center.size
        if (self.weight.shape != (3, n) or self.scale.shape != (n,) or self.bias.shape != (3,) or
                not all(np.isfinite(x).all() for x in (self.center, self.scale, self.weight, self.bias)) or
                np.any(self.scale <= 0)):
            raise ValueError("invalid spectral coefficients")
        self.reset()

    def reset(self):
        self.total = np.zeros_like(self.center)
        self.n = 0

    def prob(self, raw):
        x = self.geometry.features(raw) - self.center
        self.n += 1
        self.total += x
        x -= self.geometry.CENTER * self.total / (self.n + self.geometry.PRIOR)
        p = self.weight @ np.clip(x / self.scale, -6, 6) + self.bias
        p = np.exp(p - p.max())
        return p / p.sum()
