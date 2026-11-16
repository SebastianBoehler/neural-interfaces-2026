"""Real-data native implementation smoke; held predictions are never scored."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('NI26_WS', Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(ROOT / 'studies/16nov/lane5'), str(ROOT / 'experiments')]
import protocol  # noqa: E402
from bridge import Native, Session  # noqa: E402
from parity import session_checks  # noqa: E402
from run import ARMS, stream, validate_fit_status  # noqa: E402


def main():
    device = int(os.environ['LANE5_DEVICE'])
    native = Native(os.environ['LANE5_LIB'])
    d = protocol.load()
    held, train, test = next(iter(protocol.folds(d, 'all_loso')))
    if held != 'P01':
        raise ValueError('expected fixed P01 smoke fold')
    rows = []
    for person in sorted(set(d['subject'][train])):
        for c in (0, 1, 2):
            candidates = np.flatnonzero(train & (d['subject'] == person) & (d['y'] == c))
            if len(candidates) < 2:
                raise ValueError(f'missing smoke training examples for {person}/{c}')
            rows.extend(candidates[:2])
    rows = np.asarray(rows)
    ids = np.array([int(p.split('/')[-1][1:]) for p in d['subject'][rows]], np.int32)
    session = next(iter(dict.fromkeys(d['stream'][test])))
    held_rows = np.flatnonzero(test & (d['stream'] == session))[:61]
    if len(held_rows) != 61:
        raise ValueError('smoke needs 61 held session windows')
    windows = d['X'][held_rows]
    report = {'held': held, 'held_labels_read_for_prediction': False, 'fit_rows': len(rows), 'arms': {}}
    for name, architecture, consistency in ARMS:
        t0 = time.monotonic()
        model = native.create(architecture, device, 0)
        model.fit(d['X'][rows], d['y'][rows], ids, 1, 1, consistency)
        fit_status = model.fit_status()
        validate_fit_status(fit_status, architecture, consistency)
        direct = model.predict(windows[:2])
        handle = Session(model)
        streamed = np.stack([handle.step(x, False) for x in windows[:2]])
        diff = float(np.max(np.abs(direct - streamed)))
        if not np.allclose(direct, streamed, atol=2e-5, rtol=2e-5):
            raise ValueError(f'{name}: static batch/session mismatch {diff}')
        with tempfile.TemporaryDirectory(prefix='ni26_lane5_smoke_') as tmp:
            path = Path(tmp) / 'anchor.pt'; model.save(path)
            restored = native.create(architecture, device, 99); restored.load(path)
            if not np.allclose(model.predict(windows[:2]), restored.predict(windows[:2]), atol=2e-5, rtol=2e-5):
                raise ValueError(f'{name}: checkpoint restore changed logits')
            isolation = session_checks(restored, windows)
            restored.close()
        _, _, status = stream(model, windows)
        report['arms'][name] = {'elapsed_s': time.monotonic() - t0, 'batch_stream_max_abs': diff, 'fit_status': fit_status,
                                'isolation': isolation, 'stream': status}
        handle.solver.close(); model.close()
    if report['arms']['hybrid_ce']['fit_status']['shuffle_hash'] != report['arms']['hybrid_consistency']['fit_status']['shuffle_hash']:
        raise ValueError('hybrid training order differs between matched controls')
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
