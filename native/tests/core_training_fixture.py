"""Prepared source-only REAL-core oracle; recomputes features, never refits a head."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from convert_package import save_tensors

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
SOURCE = ROOT / 'ni26_js321_20261117_returning2/source'
FIT = ROOT / 'ni26_js321_20261117_returning_view_fit1'
PACKAGE = ROOT / 'ni26_native353_qualify_v1_20261117/package'
PINS = {
    PACKAGE / 'native.pt': '4bf45aee1fa6457e75e504d5acf747a6c6a4e21c99a3266c0c5a514c6b5ab31c',
    SOURCE / 'experiments/features.py': '360f335d5a9224aad67af9725907cda07ce799fb574a62ae44bb7eb6cbe3dabd',
    FIT / 'source.sha256': '1196bab53ccb843123e34e411c9b9e0e682161d3448ba5ff73acb06880100297',
    SOURCE / 'data/proteus_windows.npz': '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327',
    FIT / 'source_view_P04_s23.npz': '8f52ab55d4dea7eca9b43b610fad2c645c1f3029372570d05ab71b2187123675',
    SOURCE / 'experiments/ts_core.py': '0dd250ac222c7a1678c43a46e31ad7219e6d36369bd2b914293f8c31202d6e1c',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('new immutable output directory required')
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError(f'original source/reference changed: {path}')
    proof = json.loads((FIT / 'fit.json').read_text())
    if proof['source_rows'] != 2278 or proof['query_labels_used'] or proof['accuracy_computed']:
        raise ValueError('original source-only proof differs')
    started = time.monotonic()
    torch.set_num_threads(1)
    sys.path.insert(0, str(SOURCE / 'experiments'))
    import ts_core as T
    with np.load(FIT / 'source_view_P04_s23.npz') as reference:
        ref = {key: reference[key].copy() for key in reference.files}
    rows = ref['idx_tr']
    row_sha = hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()
    if row_sha != 'ca4a869ff981142b692b2a4e2ecb44f075ea2494e398280f103c9a9a36bcea2b':
        raise ValueError('source row order differs')
    with np.load(SOURCE / 'data/proteus_windows.npz', allow_pickle=False) as data:
        subject, session = data['subject'], data['session'].astype(str)
        held = (subject == 'Dreyer2026Proteus/P04') & (session != '1')
        if not np.array_equal(rows, np.flatnonzero(~held)):
            raise ValueError('complete-session source exclusion differs')
        raw = data['X'][rows].copy()
        channels = data['ch_names'].copy()
    if raw.shape != (2278, 41, 480) or raw.dtype != np.float32:
        raise ValueError('canonical model-facing source tensor differs')
    streams = subject + '|' + session
    with threadpool_limits(limits=1):
        features = T.Feats(dict(X=raw, y=ref['y_tr'], stream=streams[rows], ch_names=channels),
                           T.parse('car_ov17_fs2_t0.1_le_k2'))
    # This is the original feature implementation only. No Core.fit, scaler.fit or PCA fit.
    standardized = (features.F - ref['ts_mu']) / ref['ts_sd']
    basis = np.concatenate((ref['tsem_V'], ref['route_V_hi']), axis=1).astype(np.float64)
    with threadpool_limits(limits=6):
        pc = ((standardized - ref['tsem_m']) @ basis).astype(np.float32)
    if not np.array_equal(pc, ref['pc_tr']):
        raise ValueError('recomputed original features do not reproduce saved source PCA coordinates')
    logits = standardized @ ref['ts_W'].T + ref['ts_b']
    logits -= logits.max(1, keepdims=True)
    probabilities = np.exp(logits); probabilities /= probabilities.sum(1, keepdims=True)
    native = torch.load(PACKAGE / 'native.pt', weights_only=True)
    recipe_keys = ('sos', 'slow_sos', 'bands', 'car', 'nsub', 'shrink', 'k', 'slow_cutoff', 'slow_step')
    recipe = {key: native['ts_' + key].numpy() for key in recipe_keys}
    recipe['hilbert'] = np.int64(0)
    for key in ('bands', 'car', 'nsub', 'shrink', 'k', 'slow_cutoff', 'slow_step'):
        if not np.array_equal(recipe[key], ref['ts_' + key]):
            raise ValueError('fixed coefficient recipe differs from original source fit')
    x = np.full((len(subject), 41, 480), np.nan, np.float32); x[rows] = raw
    y = np.full(len(subject), -1, np.int64); y[rows] = ref['y_tr']
    fit_mask = ~held
    _, person = np.unique(subject, return_inverse=True)
    _, stream = np.unique(streams, return_inverse=True)
    args.out.mkdir()
    save_tensors(args.out / 'source.pt', dict(x=x, y=y, person=person.astype(np.int64),
        stream=stream.astype(np.int64), fit_mask=fit_mask, held_person=np.empty(0, np.int64)))
    save_tensors(args.out / 'recipe.pt', recipe)
    save_tensors(args.out / 'reference.pt', dict(features=features.F, probabilities=probabilities,
        rows=rows, log_glob=ref['ts_log_glob'], mu=ref['ts_mu'], sd=ref['ts_sd']))
    record = dict(status='SOURCE_ONLY_ORACLE_NO_REFIT', fold='P04_s23', source_rows=2278,
        excluded_rows=377, source_rows_sha256=row_sha, source_labels_from='existing source-only fit y_tr',
        query_labels_opened=False, query_EEG_used=False, accuracy_computed=False,
        native_training_run=False, original_feature_seconds=time.monotonic()-started,
        coefficient_archive_sha256=sha(PACKAGE / 'native.pt'), producer_sha256=sha(__file__),
        pinned_inputs={str(path): expected for path, expected in PINS.items()},
        outputs={path.name: sha(path) for path in args.out.glob('*.pt')})
    (args.out / 'proof.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
