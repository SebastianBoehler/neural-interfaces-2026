"""One-time frozen artifact migration; never imported by native train/inference."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.signal import butter
import torch

PINNED_ZIP = '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed'


def tensor(value):
    value = np.asarray(value)
    if value.dtype.kind in 'US':
        rows = [str(v).encode('utf8') for v in value.reshape(-1)]
        out = np.zeros((len(rows), max([len(v) for v in rows] + [0]) + 1), np.uint8)
        for i, row in enumerate(rows):
            out[i, :len(row)] = list(row)
        return torch.from_numpy(out)
    if value.dtype.kind not in 'bifc':
        raise ValueError(f'unsupported native field dtype: {value.dtype}')
    return torch.from_numpy(np.array(value, copy=True, order='C'))


def save_tensors(path, fields):
    values = {key: tensor(value) for key, value in fields.items()}
    values['native_format'] = torch.tensor(1, dtype=torch.int64)
    torch.save(values, path)


def export_package(package, output):
    if hashlib.sha256(package.read_bytes()).hexdigest() != PINNED_ZIP:
        raise ValueError('this migration is qualified only for frozen v6.6 cap250')
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(package) as archive:
        if any(Path(name).name != name for name in archive.namelist()):
            raise ValueError('unexpected nested package member')
        archive.extractall(output)
    with np.load(output / 'weights.npz', allow_pickle=False) as f:
        values = {key: f[key] for key in f.files}
    for prefix in ('ts_', 'ts2_'):
        values[prefix + 'sos'] = np.stack([
            butter(4, band, btype='band', fs=120, output='sos') for band in values[prefix+'bands']])
        values[prefix + 'slow_sos'] = butter(4, float(values[prefix+'slow_cutoff']),
                                             btype='low', fs=120, output='sos')
    values['em_event_sos'] = butter(4, 8, btype='low', fs=120, output='sos')
    values['own_sos'] = np.stack([butter(4, band, btype='band', fs=120, output='sos')
                                  for band in values['own_bands']])
    values['own_slow_sos'] = butter(2, 4, btype='low', fs=120, output='sos')
    for prefix in ('v6_cf_', 'v6_cp_'):
        values[prefix+'sos'] = np.stack([
            butter(4, band, btype='band', fs=float(values[prefix+'fs']), output='sos')
            for band in values[prefix+'bands']])
    save_tensors(output/'native.pt', values)
    return values


def source_fixture(package_dir, dataset, output, count, project):
    # Labels are not loaded: this is a component-equivalence check on source EEG.
    with np.load(dataset, allow_pickle=False) as f:
        rows = np.flatnonzero((f['subject'] == 'Dreyer2026Proteus/P01') & (f['session'] == '1'))[:count]
        if len(rows) != count:
            raise ValueError('not enough rows in exact P01/session1 source fixture')
        x = np.asarray(f['X'][rows], np.float32)
    official = project/'2026-competition/tracks/bci_decoding'
    sys.path.insert(0, str(official))
    spec = importlib.util.spec_from_file_location('frozen_native353_reference', package_dir/'submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(package_dir/'weights.npz', allow_pickle=False) as f:
        z = {key: f[key] for key in f.files}
    first = module.TangentCore(module.with_prefix(z, 'ts_'))
    second = module.TangentCore(module.with_prefix(z, 'ts2_'))
    tail = module.Tail(z)
    p1, p2, features, emitted = [], [], [], []
    # The two actual core outputs supply nonuniform source mixtures without a label.
    for window in x:
        p, features_row = first.step(window)
        q = second.step(window)[0]
        mix = .5*(p+q)
        u = (features_row-z['tsem_m']) @ z['tsem_V']
        p1.append(p);p2.append(q);features.append(features_row)
        emitted.append(tail.step(window, u, mix))
    save_tensors(output, dict(x=x,rows=rows,p1=np.stack(p1),p2=np.stack(p2),
        features=np.stack(features),mix=.5*(np.stack(p1)+np.stack(p2)),
        tail=np.stack(emitted),reapply=tail.reapply(count-1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--source-count', type=int, default=96)
    args = parser.parse_args()
    torch.set_num_threads(1)
    export_package(args.package, args.output)
    if args.dataset:
        if args.project_root is None:
            raise ValueError('--project-root is required for the original reference import')
        source_fixture(args.output, args.dataset, args.output/'source_fixture.pt', args.source_count, args.project_root)


if __name__ == '__main__':
    main()
