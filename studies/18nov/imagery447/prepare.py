"""Recover four original Chang continuous recordings with independent file boundaries."""
import hashlib
import json
from pathlib import Path
import sys

import mne
import numpy as np
from scipy.io import loadmat

PEOPLE = (211, 216, 221, 222, 226, 227)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(repo, person):
    sys.path.insert(0, str(repo / 'scripts'))
    from export_external import preprocess, to_proteus
    root = repo / 'data/moabb_raw/MNE-chang2025-data'
    names = np.load(repo / 'data/proteus_windows.npz')['ch_names'].tolist()
    inventory = {}
    for paradigm in ('MI', 'ME'):
        for visit in (1, 2):
            key = f'S{person}_{paradigm}{visit}'
            path = root / f'S{person}' / (key + '.set')
            hdr = loadmat(path, simplify_cells=True)['EEG']
            fdt = path.with_suffix('.fdt')
            assert hdr['data'] == fdt.name and hdr['srate'] == 1000 and hdr['trials'] == 1
            channels = [ch['labels'] for ch in hdr['chanlocs']]
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose='ERROR')
            assert raw.ch_names == channels and len(channels) == 59
            expected = np.memmap(fdt, '<f4', 'r', shape=(hdr['pnts'], 59))
            assert np.allclose(raw.get_data(start=1000, stop=1010)*1e6, expected[1000:1010].T)
            cue = np.array([ev['latency'] - 1 for ev in hdr['event']])
            label = np.array([int(ev['type']) for ev in hdr['event']])
            assert np.array_equal(raw.annotations.description, label.astype(str))
            assert np.allclose(raw.annotations.onset, cue / 1000)
            assert len(cue) == 30 and np.array_equal(np.bincount(label), [0,10,10,10])
            assert np.all(np.diff(cue) >= 4000)
            voltage = preprocess(raw).get_data() * 1e6
            starts = np.rint(cue * 0.12).astype(int)
            windows = np.stack([voltage[:, a:a+480] for a in starts])
            x, missing = to_proteus(windows, channels, names)
            assert missing == [c for c in names if c in {'CPz','P1','P2'}] and len(missing) == 3
            for i, c in enumerate(names):
                if c not in missing:
                    assert np.array_equal(x[:, i], windows[:, channels.index(c)].astype(np.float32))
            assert np.isfinite(x).all() and np.min(np.std(x, axis=2)) > 1e-5
            np.savez(key + '.npz', X=x, names=names, cue=cue, label=label)
            inventory[key] = dict(set=digest(path), fdt=digest(fdt), rows=len(cue),
                                  missing=missing, windows=digest(Path(key + '.npz')))
    Path(f'source_{person}.json').write_text(json.dumps(inventory))


if __name__ == '__main__':
    prepare(Path(sys.argv[1]), int(sys.argv[2]))
