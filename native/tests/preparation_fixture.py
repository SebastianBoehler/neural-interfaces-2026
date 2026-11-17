"""Independent original source G/scale/order, with no labels or neural calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('metadata', 'dataset', 'features', 'control', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {str(path): sha(path) for path in (args.metadata, args.dataset, args.features, args.control)}
    spec = importlib.util.spec_from_file_location('original_source_preparation', args.features)
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)
    with np.load(args.metadata, allow_pickle=False) as metadata:
        fit, order, row_session = [metadata[key].copy() for key in ('source_fit', 'source_order', 'row_session')]
    control = json.loads(args.control.read_text())
    digest = lambda value: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    if len(fit) != 1898 or digest(fit) != control['source_rows_sha256'] or digest(order) != control['trace']['source_order_sha256']:
        raise ValueError('original source6 identities/order proof differs')
    with np.load(args.dataset, allow_pickle=False) as dataset:
        subject, session = dataset['subject'], dataset['session'].astype(str)
        raw = dataset['X'][fit].copy()  # Only selected source EEG enters any computation.
    roster = {str(value).split('/')[-1] for value in subject[fit]}
    if roster != {'P04', 'P05', 'P06', 'P07', 'P08', 'P10'}:
        raise ValueError('source6 is the explicit original inclusion roster')
    covariance = np.concatenate([features.covariances(raw[i:i + 512]) for i in range(0, len(raw), 512)])
    prior = covariance.mean(0)
    scale = float(np.sqrt(np.trace(prior) / 41))
    if scale != control['trace']['scale']:
        raise ValueError('source-only original scale does not reproduce original control')
    positions = np.searchsorted(fit, order)
    if not np.array_equal(fit[positions], order):
        raise ValueError('source order includes an excluded row')
    expected = (raw[positions].astype(np.float64) / scale).astype(np.float32)
    _, person = np.unique(subject, return_inverse=True)
    _, streams = np.unique(subject + '|' + session, return_inverse=True)
    first = list(dict.fromkeys(streams[order]))
    mapped = np.asarray([first.index(value) for value in streams[order]], np.int64)
    if not np.array_equal(mapped, row_session):
        raise ValueError('metadata session grouping differs from original sampler')
    mask = np.zeros(len(subject), bool)
    mask[fit] = True
    x = np.full((len(subject), 41, 480), np.nan, np.float32)
    x[fit] = raw
    held = np.setdiff1d(np.unique(person), np.unique(person[fit])).astype(np.int64)
    arrays = dict(x=x, stream=streams.astype(np.int64), person=person.astype(np.int64), fit_mask=mask,
                  held_person=held, expected_rows=order, expected_session=row_session,
                  expected_G=prior, expected_scale=np.asarray(scale), expected_x=expected)
    fields = {key: torch.from_numpy(np.ascontiguousarray(value) if np.asarray(value).ndim else np.asarray(value))
              for key, value in arrays.items()}
    fields['native_format'] = torch.tensor(1, dtype=torch.int64)
    torch.save(fields, args.output)
    if {path: sha(path) for path in inputs} != inputs:
        raise ValueError('source preparation inputs changed')
    args.output.with_suffix('.json').write_text(json.dumps(dict(source_rows=1898, inputs=inputs,
        decoded_dataset_keys=['X', 'subject', 'session'], labels_loaded=False, forward_calls=0,
        optimizer_steps=0, excluded_EEG_poisoned=True, scale=scale, fixture_sha256=sha(args.output)), indent=2))


if __name__ == '__main__':
    main()
