"""Label-free fixed source7 projection/contrast statistics, not per-window adaptation."""
import json
import numpy as np
import torch
from display import Display, LAYOUTS
from shared import digest


def fit(root, d, rows):
    a = d['X'][rows]
    scale = float(np.quantile(np.abs(a), .995))
    matrix = a.transpose(1, 0, 2).reshape(41, -1).astype(np.float64)
    _, vectors = np.linalg.eigh(matrix @ matrix.T / matrix.shape[1])
    projection = vectors[:, -16:][:, ::-1].T.copy()
    for row in projection:
        if row[np.argmax(np.abs(row))] < 0: row *= -1
    display = Display(scale, projection).eval()
    means, sds = [], []
    with torch.no_grad():
        for layout in LAYOUTS:
            total = torch.zeros(14, dtype=torch.float64); second = total.clone(); n = 0
            for r in np.array_split(rows, max(1, (len(rows) + 7) // 8)):
                z = display.logpower(torch.from_numpy(d['X'][r]), layout).double()
                total += z.sum((0, 1, 3)); second += z.square().sum((0, 1, 3))
                n += z.shape[0] * z.shape[1] * z.shape[3]
            mean = total / n; sd = (second / n - mean.square()).clamp_min(1e-12).sqrt()
            means.append(mean.numpy()); sds.append(sd.numpy())
    means, sds = np.stack(means), np.stack(sds)
    metadata = dict(fit_people=sorted(set(d['subject'][rows])), excluded=['P01', 'P02', 'P03'],
        rows_sha256=digest(rows), input_sha256=digest(a), projection_sha256=digest(projection),
        contrast='source per-frequency moments,pooled training channels/time;no per-window stats', labels_opened=False)
    with (root / 'renderer.npz').open('xb') as out:
        np.savez(out, scale=scale, projection=projection, spectral_mean=means, spectral_sd=sds, metadata=json.dumps(metadata))
    (root / 'renderer.npz').chmod(0o444)
    return Display(scale, projection, means, sds).eval(), metadata


def load(root, rows):
    with np.load(root / 'renderer.npz') as z:
        meta = json.loads(str(z['metadata'])); projection = z['projection']
        if meta['rows_sha256'] != digest(rows) or meta['projection_sha256'] != digest(projection) or meta['labels_opened']:
            raise ValueError('source-fixed renderer provenance changed')
        display = Display(float(z['scale']), projection, z['spectral_mean'], z['spectral_sd'])
    return display.eval(), meta
