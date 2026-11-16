"""P01 training-only feature/optimizer parity investigation; no held score."""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'experiments')]
os.environ['NI26_ROOT'] = str(ROOT)
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
from bridge import Native  # noqa: E402


def main():
    out = ROOT / 'runs/s301_audit/head.json'
    if out.exists():
        raise FileExistsError(out)
    d = protocol.load()
    folds = [(tr, te) for held, tr, te in protocol.folds(d, 'all_loso') if held == 'P01']
    if len(folds) != 1:
        raise ValueError('P01 fold not found')
    tr, te = folds[0]
    fit = np.flatnonzero(tr)
    if np.any(te[fit]):
        raise ValueError('held-person row reached training diagnostics')
    fe = T.Feats(d, T.parse('car_ov17_fs2_t0.1_le_k2'))
    original = np.ascontiguousarray(fe.F[fit])
    rows, parts = [], []
    for i in range(2, 11):
        with np.load(ROOT / f'runs/s301_native_inputs/P{i:02d}.npz', allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            if meta['person'] != f'P{i:02d}' or meta['held_accuracy_computed'] is not False:
                raise ValueError('native training inputs have wrong identity')
            rows.append(z['rows'].copy()); parts.append(z['features'].copy())
    index = np.concatenate(rows); order = np.argsort(index)
    if not np.array_equal(index[order], fit):
        raise ValueError('training rows differ')
    native_features = np.ascontiguousarray(np.concatenate(parts)[order])
    diff = native_features - original
    scaler = StandardScaler().fit(original)
    native_scaler = StandardScaler().fit(native_features)
    reference = Path(os.environ['S301_ALL_REFERENCES']) / 'P01.npz'
    with np.load(reference, allow_pickle=False) as z:
        saved = {k: np.ascontiguousarray(z[k]) for k in z.files}
    if not np.array_equal(saved['fit_rows'], fit):
        raise ValueError('saved model training fold differs')
    lib = Native()
    x_native = lib.standardize(native_features, saved['mean'], saved['scale'])
    x_original = lib.standardize(original, saved['mean'], saved['scale'])
    y = np.ascontiguousarray(d['y'][fit], dtype=np.int64)
    theta = np.ascontiguousarray(np.column_stack((saved['coef'], saved['intercept'])))
    saved_native_loss, saved_native_grad = lib.objective(x_native, y, theta)
    saved_original_loss, saved_original_grad = lib.objective(x_original, y, theta)
    report = {'training_feature_max_abs': float(np.max(np.abs(diff))),
              'training_feature_rel_l2': float(np.linalg.norm(diff) / np.linalg.norm(original)),
              'scaler_mean_max_abs': float(np.max(np.abs(native_scaler.mean_ - scaler.mean_))),
              'scaler_scale_max_abs': float(np.max(np.abs(native_scaler.scale_ - scaler.scale_))),
              'saved_scaler_mean_match': bool(np.array_equal(saved['mean'], scaler.mean_)),
              'saved_scaler_scale_match': bool(np.array_equal(saved['scale'], scaler.scale_)),
              'saved_head_original_objective': saved_original_loss,
              'saved_head_native_feature_objective': saved_native_loss,
              'saved_head_original_max_gradient': float(np.max(np.abs(saved_original_grad))),
              'saved_head_native_feature_max_gradient': float(np.max(np.abs(saved_native_grad))),
              'held_accuracy_computed': False, 'fits': {}}
    for name, x, tolerance in (('default_original', x_original, 1e-4), ('strict_native', x_native, 1e-8)):
        with warnings.catch_warnings(record=True) as caught:
            start = time.monotonic()
            model = LogisticRegression(C=.3, max_iter=2000, tol=tolerance).fit(x, y)
            elapsed = time.monotonic() - start
        theta2 = np.ascontiguousarray(np.column_stack((model.coef_, model.intercept_)))
        loss, gradient = lib.objective(x_native, y, theta2)
        report['fits'][name] = {'seconds': elapsed, 'iterations': int(model.n_iter_.max()),
            'objective_on_native_features': loss, 'max_gradient_on_native_features': float(np.max(np.abs(gradient))),
            'warnings': [str(w.message) for w in caught]}
        print(f'{name} training-only fit complete; no score', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True); f.write('\n')
    print(f'training-only objective audit: {out}', flush=True)


if __name__ == '__main__':
    main()
