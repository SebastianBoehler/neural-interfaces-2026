"""No-score full 62-window CPU/GPU parity for the native matched core."""
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import butter, sosfilt_zi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_preprocess import BANDS  # noqa: E402

PD, PF = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float)
ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
REFERENCE = Path(os.environ['S301_REFERENCE']).resolve(strict=True)
LIBRARY = Path(os.environ['NATIVE_CORE_LIB']).resolve(strict=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ptr(x):
    return x.ctypes.data_as(PF if x.dtype == np.float32 else PD)


def native():
    lib = ctypes.CDLL(str(LIBRARY))
    lib.nc_preprocess.argtypes = [PF, PD, PD, PD, PD, PF, PF]
    lib.nc_covariances.argtypes = [PF, PD]
    lib.nc_create.argtypes, lib.nc_create.restype = [PD, ctypes.c_int], ctypes.c_void_p
    lib.nc_step.argtypes = [ctypes.c_void_p, PD, PF, PD]
    lib.nc_reset.argtypes = [ctypes.c_void_p]
    lib.nc_destroy.argtypes = [ctypes.c_void_p]
    lib.nc_head_predict.argtypes = [PD, ctypes.c_int64, ctypes.c_int64, PD, PD, PD, PD, PD, PD]
    lib.nc_error.restype = ctypes.c_char_p
    lib.nc_head_error.restype = ctypes.c_char_p
    for name in ('nc_preprocess', 'nc_covariances', 'nc_step', 'nc_reset', 'nc_head_predict'):
        getattr(lib, name).restype = ctypes.c_int
    return lib


def run():
    target = ROOT / 'runs/s301_parity/report.json'
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    expected = {f'{p}_{o}.npz' for p in ('P01', 'P10') for o in ('within0', 'full0')}
    if set(manifest['files']) != expected or manifest['held_accuracy_computed'] is not False:
        raise ValueError('incomplete no-score reference matrix')
    if (manifest['source_sha256'] != '2e9833fe9adfe9385ed80996b00a1cc4f890a4ff4903bb7d96fe9b422299f227' or
            manifest['data_sha256'] != '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327' or
            manifest['permutation_sha256'] != '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'):
        raise ValueError('reference data, permutation or source differs')
    if manifest['scipy'] != scipy.__version__:
        raise ValueError('reference and native coefficient versions differ')
    for name, digest in manifest['files'].items():
        if sha(REFERENCE / name) != digest:
            raise ValueError(f'changed reference file {name}')
    filters = np.ascontiguousarray([butter(4, list(b), btype='band', fs=120, output='sos') for b in BANDS])
    zi = np.ascontiguousarray([sosfilt_zi(s) for s in filters])
    low = np.ascontiguousarray(butter(4, 1, btype='low', fs=120, output='sos'))
    lowzi = np.ascontiguousarray(sosfilt_zi(low))
    coefficient_sha = hashlib.sha256(b''.join(a.tobytes() for a in (filters, zi, low, lowzi))).hexdigest()
    lib = native()
    records = {}
    for name in sorted(expected):
        with np.load(REFERENCE / name, allow_pickle=False) as z:
            x = {k: np.ascontiguousarray(z[k]) for k in z.files}
        if len(x['raw']) != 62 or x['expected_features'].shape != (62, 44239):
            raise ValueError('incorrect full-window reference shape')
        cov = np.empty((62, 51, 41, 41), np.float64)
        slow = np.empty((62, 328), np.float32)
        band = np.empty((17, 41, 480), np.float32)
        prep = time.monotonic()
        for i in range(62):
            if lib.nc_preprocess(ptr(x['raw'][i]), ptr(filters), ptr(zi), ptr(low), ptr(lowzi), ptr(band), ptr(slow[i])):
                raise RuntimeError('native preprocessing failed')
            if lib.nc_covariances(ptr(band), ptr(cov[i])):
                raise RuntimeError('native covariance failed')
        prep_seconds = time.monotonic() - prep
        for device in (0, 1):
            t0 = time.monotonic()
            h = lib.nc_create(ptr(x['prior']), device)
            if not h:
                raise RuntimeError(lib.nc_error().decode())
            features = np.empty((62, 44239), np.float64)
            try:
                for i in range(62):
                    if lib.nc_step(h, ptr(cov[i]), ptr(slow[i]), ptr(features[i])):
                        raise RuntimeError(lib.nc_error().decode())
                assert lib.nc_reset(h) == 0
                replay = np.empty((61, 44239), np.float64)
                for i in range(61):
                    if lib.nc_step(h, ptr(cov[i]), ptr(slow[i]), ptr(replay[i])):
                        raise RuntimeError(lib.nc_error().decode())
                np.testing.assert_array_equal(replay, features[:61])
            finally:
                lib.nc_destroy(h)
            diff = features - x['expected_features']
            relative = float(np.linalg.norm(diff) / np.linalg.norm(x['expected_features']))
            maximum = float(np.max(np.abs(diff)))
            logits, probs = np.empty((62, 3), np.float64), np.empty((62, 3), np.float64)
            if lib.nc_head_predict(ptr(features), 62, 44239, ptr(x['mean']), ptr(x['scale']),
                                   ptr(x['coef']), ptr(x['intercept']), ptr(logits), ptr(probs)):
                raise RuntimeError(lib.nc_head_error().decode())
            probdiff = float(np.max(np.abs(probs - x['expected_probabilities'])))
            sorted_ref = np.sort(x['expected_probabilities'], axis=1)
            nontied = sorted_ref[:, -1] > sorted_ref[:, -2]
            changed = int(np.count_nonzero((probs.argmax(1) != x['expected_probabilities'].argmax(1)) & nontied))
            if relative > 1e-6 or maximum > 1e-4 or probdiff > 1e-5 or changed:
                raise ValueError(f'{name}/{device}: parity failed {relative=} {maximum=} {probdiff=} {changed=}')
            records[f'{name}|device{device}'] = {'feature_rel_l2': relative, 'feature_max_abs': maximum,
                'probability_max_abs': probdiff, 'non_tied_label_changes': changed,
                'seconds': time.monotonic() - t0}
        records[f'{name}|preprocess_seconds'] = prep_seconds
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f:
        json.dump({'records': records, 'reference_manifest_sha256': sha(REFERENCE / 'manifest.json'),
                   'library_sha256': sha(LIBRARY), 'scipy_version': scipy.__version__,
                   'coefficients_sha256': coefficient_sha, 'held_accuracy_computed': False}, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'full reference parity passed: {target}; no score', flush=True)


if __name__ == '__main__':
    run()
