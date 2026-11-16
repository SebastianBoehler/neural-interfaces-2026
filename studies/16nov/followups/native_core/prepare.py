"""Produce native per-person covariances and whole-session training features."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt_zi

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'experiments')]
import protocol  # noqa: E402
from bridge import Native  # noqa: E402
from test_preprocess import BANDS  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(task):
    if not 0 <= task < 10:
        raise ValueError('person task must be 0–9')
    expected = f'P{task + 1:02d}'
    target = ROOT / f'runs/s301_native_inputs/{expected}.npz'
    if target.exists():
        raise FileExistsError(target)
    gate_path = Path(os.environ['S301_PARITY_REPORT']).resolve(strict=True)
    gate = json.loads(gate_path.read_text())
    if gate.get('held_accuracy_computed') is not False or len(gate.get('records', {})) != 12:
        raise ValueError('registered full CPU/GPU parity report is incomplete')
    if gate['library_sha256'] != digest(Path(os.environ['NATIVE_CORE_LIB'])):
        raise ValueError('full parity did not validate this native library')
    coeff = np.ascontiguousarray([butter(4, list(b), btype='band', fs=120, output='sos') for b in BANDS])
    zi = np.ascontiguousarray([sosfilt_zi(s) for s in coeff])
    low = np.ascontiguousarray(butter(4, 1, btype='low', fs=120, output='sos'))
    lowzi = np.ascontiguousarray(sosfilt_zi(low))
    coefficients = (coeff, zi, low, lowzi)
    if hashlib.sha256(b''.join(x.tobytes() for x in coefficients)).hexdigest() != gate['coefficients_sha256']:
        raise ValueError('filter coefficients differ from the parity gate')
    d = protocol.load()
    fold = [(tr, te) for held, tr, te in protocol.folds(d, 'all_loso') if held == expected]
    if len(fold) != 1:
        raise ValueError('expected one held-person split')
    tr, te = fold[0]
    rows = np.flatnonzero(te)
    if set(d['subject'][tr]) & set(d['subject'][te]):
        raise ValueError('person split overlaps')
    lib = Native()
    cov = np.empty((len(rows), 51, 41, 41), np.float64)
    slow = np.empty((len(rows), 328), np.float32)
    features = np.empty((len(rows), 44239), np.float64)
    start = time.monotonic()
    for i, row in enumerate(rows):
        cov[i], slow[i] = lib.preprocess(np.ascontiguousarray(d['X'][row], dtype=np.float32), coefficients)
    aggregate = lib.train_reference(1)
    try:
        for i in range(len(rows)):
            lib.add(aggregate, cov[i], slow[i])
        prior = lib.prior(aggregate)
    finally:
        lib.train.nt_destroy(aggregate)
    sessions = d['stream'][rows]
    for session in dict.fromkeys(sessions):
        indices = np.flatnonzero(sessions == session)
        ref = lib.train_reference(1)
        try:
            for i in indices:
                lib.add(ref, cov[i], slow[i])
            for i in indices:
                features[i] = lib.feature(ref, cov[i], slow[i])
        finally:
            lib.train.nt_destroy(ref)
    if not np.isfinite(features).all() or not np.isfinite(prior).all():
        raise ValueError('nonfinite native preparation')
    metadata = {'person': expected, 'rows_sha256': hashlib.sha256(rows.tobytes()).hexdigest(),
                'parity_report_sha256': digest(gate_path), 'native_core_sha256': digest(Path(os.environ['NATIVE_CORE_LIB'])),
                'native_train_sha256': digest(Path(os.environ['NATIVE_TRAIN_LIB'])),
                'source_sha256': digest(ROOT / 'producer.sha256'), 'seconds': time.monotonic() - start,
                'session_counts': {str(s): int(np.count_nonzero(sessions == s)) for s in dict.fromkeys(sessions)},
                'held_accuracy_computed': False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        np.savez(f, rows=rows, cov=cov, slow=slow, features=features, prior=prior,
                 metadata=json.dumps(metadata, sort_keys=True))
    print(f'{expected}: native training features and covariances prepared, no score; {metadata["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: prepare.py PERSON_INDEX')
    main(int(sys.argv[1]))
