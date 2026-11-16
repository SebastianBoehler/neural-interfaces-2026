"""Training-only raw baseline cache; official NeuroAI filter methods own preprocessing."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import mne
import numpy as np
from neuralset.extractors.neuro import EegExtractor

ROOT = Path(os.environ['NI26_ROOT'])
RAW = ROOT / 'data/neural_compet/Dreyer2026Proteus/download/nm000290'
CACHE = Path(os.environ['NI26_CACHE'])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def processed(path, names):
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    raw.pick(names).reorder_channels(names)
    if raw.info['sfreq'] != 500:
        raise ValueError(f'unexpected PROTEUS native rate: {raw.info["sfreq"]}')
    # Reuse exact owner methods: notch harmonics, annotation handling and FIR rules.
    extractor = EegExtractor()
    raw = extractor._notch_filter(raw, [50., 60.], 1)
    raw = extractor._band_filter(raw, (.1, 75.), str(path))
    raw.resample(120., n_jobs=1, verbose=False)
    return raw.get_data().astype(np.float64) * 1e6


def parity(d):
    subject = next(p for p in set(d['subject']) if p.split('/')[-1] == 'P02')
    rows = np.flatnonzero(d['subject'] == subject)
    row = rows[0]
    task, run, session = str(d['task'][row]), int(d['run'][row]), str(d['session'][row])
    if not session.startswith('ses-'):
        session = f'ses-{session}'
    paths = sorted((RAW / 'sub-P02' / session / 'eeg').glob(f'*task-{task}_run-{run}_eeg.edf'))
    if len(paths) != 1:
        raise ValueError(f'expected one official source EDF for {task}/{session}/{run}: {paths}')
    values = processed(paths[0], d['ch_names'].astype(str).tolist())
    onset = float(d['onset'][row]); start = int(round(onset * 120))
    errors = {}
    for offset in (-120, -1, 0, 1, 120):
        z = values[:, start + offset:start + offset + 480].astype('float32')
        if z.shape == (41, 480):
            errors[str(offset)] = float(np.max(np.abs(z - d['X'][row])))
    result = dict(source='P02', official_row=int(row), onset=onset, raw_file=str(paths[0]),
                  raw_sha256=sha(paths[0]), max_abs_error_by_sample_offset=errors)
    if errors.get('0', float('inf')) > 2e-4:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / 'failed_parity.json').write_text(json.dumps(result, indent=2))
        raise ValueError(f'raw/official zero-offset parity failed: {result}')
    result['status'] = 'passed'
    (CACHE / 'parity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def export_person(person, d):
    if person not in {f'P{i:02d}' for i in range(1, 11)}:
        raise ValueError('invalid public person')
    target = CACHE / f'{person}.npz'
    if target.exists():
        raise FileExistsError(target)
    names = d['ch_names'].astype(str).tolist()
    files = sorted((RAW / f'sub-{person}').glob('ses-*/eeg/*task-Baseline*_eeg.edf'))
    if not files:
        raise FileNotFoundError(f'no baseline EDFs for {person}')
    xs, records, manifests = [], [], []
    for path in files:
        values = processed(path, names)
        # Drop first/last 4s; retained baselines have no command-window overlap.
        starts = np.arange(480, values.shape[1] - 959, 480, dtype=int)
        if not len(starts):
            raise ValueError(f'baseline too short: {path}')
        z = np.stack([values[:, s:s + 480] for s in starts]).astype('float32')
        if not np.isfinite(z).all():
            raise ValueError(f'nonfinite baseline: {path}')
        xs.append(z); records.extend([str(path.relative_to(RAW))] * len(z))
        manifests.append(dict(path=str(path), sha256=sha(path), n=len(z), sample_starts=starts.tolist()))
    with target.open('xb') as out:
        np.savez(out, X=np.concatenate(xs), record=np.asarray(records), person=person,
                 metadata=json.dumps(dict(files=manifests, channel_names=names,
                     owner='neuralset.extractors.neuro.EegExtractor methods', units='uV',
                     scaling='none', official_export=sha(ROOT / 'data/proteus_windows.npz'))))
    print(json.dumps(dict(person=person, n=sum(map(len, xs)), sha256=sha(target))), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('parity', 'export'))
    parser.add_argument('--person'); args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / 'data/proteus_windows.npz') as source:
        d = {k: source[k] for k in source.files}
    if args.mode == 'parity':
        parity(d)
    else:
        if not (CACHE / 'parity.json').is_file():
            raise ValueError('passed full-record raw/official parity required before export')
        export_person(args.person, d)


if __name__ == '__main__':
    main()
