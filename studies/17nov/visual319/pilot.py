"""Frozen visual-prior source-person pilot; P01 labels are never opened."""
import importlib.util
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
from assets import sha, verify
from data319 import DATA, DATA_SHA, digest, load, load_display
from encoder import build, features, offline, state_sha
from render import Display, controls

LABEL_HELPER = Path('/mnt/lustre/work/gehler/gfh527/ni26_s306_expert_cpu_v2_20261116/source_labels.py')


def main():
    root = Path(os.environ['VISUAL_ROOT']).resolve(strict=True)
    report = json.loads((root / 'preflight_cuda.json').read_text())
    layout = report['layouts']['16']
    if (not report['passed'] or max(v['median'] for v in layout['latency'].values()) > .15 or
            layout['gpu_peak_allocated'] > 8 * 1024**3 or 'A100' not in torch.cuda.get_device_name(0)):
        raise ValueError('registered bounded A100 engineering guards fail;no source pilot')
    torch.set_num_threads(6); verify(root)
    if sha(DATA) != DATA_SHA:
        raise ValueError('source tensor hash differs')
    x, subject, names, fit, val = load()
    scale, projection, source = load_display(root, fit)
    rows = np.r_[fit, val]
    if any(str(p).endswith('/P01') for p in subject[rows]):
        raise ValueError('P01 entered source feature extraction')
    display = Display(scale, projection).cuda().eval()
    arrays = {}; costs = {}
    for variant in ('pretrained', 'random'):
        model = build(root, variant, 'cuda'); before = state_sha(model)
        started = time.monotonic(); values = []
        with torch.no_grad(), offline():
            for r in rows:
                image = display(torch.from_numpy(x[r:r+1]).cuda(), 16)
                values.append(features(model, image, 560).cpu().numpy()[0])
        arrays[variant] = np.stack(values); costs[variant] = time.monotonic() - started
        if state_sha(model) != before:
            raise ValueError('frozen encoder changed in source extraction')
        del model
    with torch.no_grad():
        arrays['direct'] = np.concatenate([controls(display, torch.from_numpy(x[r:r+1]).cuda()).cpu().numpy() for r in rows])
    # Reuse the owner helper, which seeks only selected source-label entries.
    if sha(LABEL_HELPER) != '35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc':
        raise ValueError('pinned source-only label helper changed')
    spec = importlib.util.spec_from_file_location('visual319_source_labels', LABEL_HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    yfit = module.read_training_labels(DATA, fit, len(x))
    yval = module.read_training_labels(DATA, val, len(x))
    if set(yfit) != {0, 1, 2} or set(yval) != {0, 1, 2}:
        raise ValueError('source fitting/validation lacks classes')
    results = {}; n = len(fit)
    for name, a in arrays.items():
        if not np.isfinite(a).all():
            raise ValueError('nonfinite source features')
        candidates = []
        for c in (.01, .1, 1.):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            model.fit(a[:n], yfit); p = model.predict_proba(a[n:])
            if int(model[-1].n_iter_.max()) >= 2000:
                raise ValueError('source readout failed its fixed convergence budget')
            if not np.array_equal(model[-1].classes_, [0, 1, 2]):
                raise ValueError('head label convention differs')
            candidates.append(dict(C=c, logloss=float(log_loss(yval, p, labels=[0, 1, 2])),
                balanced_accuracy=float(balanced_accuracy_score(yval, p.argmax(1))), iterations=int(model[-1].n_iter_.max())))
        best = min(candidates, key=lambda z: z['logloss'])
        results[name] = dict(features=a.shape[1], candidates=candidates, selected=best, feature_sha256=digest(a))
    result = dict(source_only=True, outer_P01_labels_opened=False, source_fit_people=source['fit_people'],
        validation_people=['P02'], source_rows_sha256=digest(fit), validation_rows_sha256=digest(val),
        source_manifest_sha256=sha(root / 'source.sha256'), assets_manifest_sha256=sha(root / 'assets/manifest.json'),
        renderer_sha256=sha(root / 'renderer.npz'), source_label_helper_sha256=sha(LABEL_HELPER),
        data_sha256=DATA_SHA, results=results, extraction_seconds=costs,
        metric='pooled balanced accuracy on one source validation person;not official-cell or outer generalization',
        limit='C selected on same P02;optimistic source diagnostic,no full-system/package claim')
    with (root / 'source_pilot.json').open('x') as out:
        json.dump(result, out, indent=2)
    with (root / 'source_features.npz').open('xb') as out:
        np.savez(out, index=rows, **arrays)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
