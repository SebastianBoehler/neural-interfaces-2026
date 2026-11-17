"""Fixed source-only representation factorial; both validation people were previously exposed."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from calibrate import load
from display import LAYOUTS, VIEWS
from shared import build, check, digest, features, labels, load_data, offline, sha, state_sha


def main():
    root = Path(os.environ['VISUAL_ROOT']).resolve(strict=True)
    guard = json.loads((root / 'guard.json').read_text())
    if not guard['passed'] or guard['labels_opened'] or guard['source_sha256'] != sha(root / 'source.sha256'):
        raise ValueError('fresh passed no-score guard with matching source required')
    if guard['renderer_sha256'] != sha(root / 'renderer.npz') or guard['feature_guard_sha256'] != sha(root / 'guard_features.npz'):
        raise ValueError('renderer or guarded features changed')
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise ValueError('explicit registered A100 route required;no fallback')
    torch.set_num_threads(6); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    check(root); d, fit, val, people = load_data(); rows = np.r_[fit, val]
    display, calibration = load(root, fit); display.cuda().eval()
    banks = {}; costs = {}; parity = {}; started_all = time.monotonic()
    # No validation labels are read during feature production or checks.
    with torch.no_grad():
        raw = np.concatenate([display.raw_meanpower(torch.from_numpy(d['X'][r:r+1]).cuda()).cpu().numpy() for r in rows])
        for layout in LAYOUTS:
            for view in VIEWS:
                name = f'{layout}_{view}'
                banks[f'direct_{name}'] = np.concatenate([display.direct(torch.from_numpy(d['X'][r:r+1]).cuda(), layout, view).cpu().numpy() for r in rows])
                if not np.array_equal(banks[f'direct_{name}'][:, -287:], raw): raise ValueError('direct raw block changed')
    banks['raw287'] = raw
    for variant in ('pretrained', 'random'):
        model = build(root, variant, 'cuda'); before = state_sha(model)
        for layout in LAYOUTS:
            for view in VIEWS:
                started = time.monotonic(); values = []
                with torch.no_grad(), np.load(root / 'guard_features.npz') as reference:
                    signal = torch.from_numpy(d['X'][guard['source_windows'][0]:guard['source_windows'][0]+1]).cuda()
                    actual = features(model, display(signal, layout, view), layout * 35).cpu().numpy()
                    key = f'{layout}|{view}|{variant}'; error = float(np.max(np.abs(actual-reference[key])))
                    if error > .005: raise ValueError('CPU/A100 new-view source-feature parity failed')
                    parity[key] = error
                with torch.no_grad(), offline():
                    for r in rows:
                        image = display(torch.from_numpy(d['X'][r:r+1]).cuda(), layout, view)
                        values.append(features(model, image, layout * 35).cpu().numpy()[0])
                name = f'{variant}_{layout}_{view}'; array = np.stack(values)
                banks[name] = array; banks[name + '_raw287'] = np.concatenate([array, raw], 1)
                if not np.array_equal(banks[name + '_raw287'][:, -287:], raw): raise ValueError('prior-specific raw skip changed')
                costs[name] = time.monotonic() - started
        if state_sha(model) != before: raise ValueError('frozen backbone changed')
        del model
    if any(not np.isfinite(v).all() or len(v) != len(rows) for v in banks.values()): raise ValueError('feature matrix guard failed')
    # Fit labels only after no-score guard and full feature banks; no P01 label access.
    yfit = labels(fit, len(d['X']))
    from baseline import predict, ORDERS
    baseline, baseline_meta = predict(d, fit, val, yfit)
    # Only now open selected P02/P03 labels; both are acknowledged development people.
    yval = labels(val, len(d['X'])); selected = {}; n = len(fit)
    for name, array in banks.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=2000))
        model.fit(array[:n], yfit)
        if int(model[-1].n_iter_.max()) >= 2000: raise ValueError('fixed source head did not converge')
        p = model.predict_proba(array[n:])
        stats = {}
        for person in ('P02', 'P03'):
            mask = people[val] == person
            stats[person] = dict(balanced_accuracy=float(balanced_accuracy_score(yval[mask], p[mask].argmax(1))),
                logloss=float(log_loss(yval[mask], p[mask], labels=[0, 1, 2])))
        selected[name] = dict(features=array.shape[1], people=stats,
            person_mean_ba=float(np.mean([v['balanced_accuracy'] for v in stats.values()])),
            iterations=int(model[-1].n_iter_.max()), feature_sha256=digest(array))
    base_stats = {}
    for person in ('P02', 'P03'):
        mask = people[val] == person
        by_order = {o: float(balanced_accuracy_score(yval[mask], p[mask].argmax(1))) for o, p in baseline.items()}
        base_stats[person] = dict(by_order=by_order, scheme_equal=float(np.mean([np.mean([v for o, v in by_order.items() if o.startswith(s)]) for s in ('within', 'full', 'runs')])))
    report = dict(source_only=True, P01_labels_opened=False, source_fit_people=calibration['fit_people'],
        validation_people=['P02', 'P03'], label_scope='both previously exposed development people;not pristine outer validation',
        source_rows_sha256=digest(fit), validation_rows_sha256=digest(val), raw_skip_sha256=digest(raw),
        results=selected, strong_reference=base_stats, strong_reference_metadata=baseline_meta,
        CPU_A100_source_feature_max_abs=parity, torch_version=torch.__version__,
        peak_gpu_allocated=torch.cuda.max_memory_allocated(), peak_gpu_reserved=torch.cuda.max_memory_reserved(),
        fixed_C=.01, metric='pooled BA per source validation person;scheme-equal baseline',
        extraction_seconds=costs, total_seconds=time.monotonic()-started_all,
        source_sha256=sha(root / 'source.sha256'), renderer_sha256=sha(root / 'renderer.npz'),
        limit='no current-system/outer/sealed result;no automatic larger matrix')
    with (root / 'source_results.json').open('x') as out: json.dump(report, out, indent=2)
    with (root / 'source_features.npz').open('xb') as out: np.savez(out, index=rows, **banks)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
