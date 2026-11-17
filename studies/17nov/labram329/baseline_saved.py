"""Unchanged source7-trained raw SessionBN reference with causal order/reset guards."""
import sys
import numpy as np
import torch
from shared import BASE, digest

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def predict(d, fit, val, yfit, artifact_path):
    sys.path.insert(0, str(BASE / 'experiments'))
    import session_bn
    data = dict(d)
    data['y'] = np.full(len(d['X']), -1, dtype=np.int64); data['y'][fit] = yfit
    data['w'] = np.ones(len(d['X'])); data['nolow'] = np.zeros(len(d['X']), bool); data['tag'] = ''
    model = session_bn.make('fbcnet_raw_sbn', data, 0)
    model.fit(fit); model.set_k(20.)
    torch.save(dict(state_dict=model.net.state_dict(), G=model.G, scale=model.scale, k=20.,
        fit_rows=fit, fit_subjects=sorted(model.fit_subjects), seed=0, recipe='fbcnet_raw_sbn;40epochs'),artifact_path)
    if model.fit_subjects != set(d['subject'][fit]) or model.n_sbn <= 0:
        raise ValueError('strong SessionBN source/control path differs')
    output = {}; prefix_error = 0.
    before = {n: p.detach().cpu().numpy().copy() for n, p in model.net.state_dict().items()}
    with np.load(BASE / 'runs/shuffle_perms.npz') as z, torch.no_grad():
        for order in ORDERS:
            perm = z[f'{order[:-1]}|{order[-1]}|all_loso']
            if not np.array_equal(np.sort(perm), np.arange(len(d['X']))): raise ValueError('bad baseline order')
            rank = np.empty(len(perm), dtype=int); rank[perm] = np.arange(len(perm))
            result = np.full((len(val), 3), np.nan)
            for session in dict.fromkeys(d['stream'][val]):
                local = np.flatnonzero(d['stream'][val] == session)
                local = local[np.argsort(rank[val[local]], kind='stable')]
                rows = val[local]
                p = model.predict_proba_stream(rows)
                q = model.predict_proba_stream(rows[:7])
                prefix_error = max(prefix_error, float(np.max(np.abs(p[:7] - q))))
                if prefix_error > 1e-4: raise ValueError('raw SessionBN prefix/batch-shape guard failed')
                if not np.array_equal(p, model.predict_proba_stream(rows)):
                    raise ValueError('raw SessionBN repeated/reset inference changed')
                result[local] = p
            if not np.isfinite(result).all(): raise ValueError('baseline rows missing')
            output[order] = result
    if any(not np.array_equal(before[n], p.detach().cpu().numpy()) for n, p in model.net.state_dict().items()):
        raise ValueError('raw source reference changed at inference')
    meta = dict(recipe='unchanged fbcnet_raw_sbn;k20;40epochs', fit_people=sorted(model.fit_subjects),
        n_sbn=model.n_sbn, prefix_max_abs=prefix_error, source_rows_sha256=digest(fit))
    return output, meta
