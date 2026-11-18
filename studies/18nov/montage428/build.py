"""Compare source-independent interpolation on the Scherer raw recordings."""
import json
import sys
from pathlib import Path

import mne
import numpy as np

BASE = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
sys.path.insert(0, str(BASE / 'scripts'))
import export_external as ext

DATA = BASE / 'data/scherer2015/windows.npz'


def maps(sources, targets):
    # Map an identity basis to obtain the very same MNE spline weights.
    spline, _ = ext.to_proteus(np.eye(len(sources))[None], sources, targets)
    montage = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    positions = np.array([montage[c] for c in sources])
    local = np.zeros((len(targets), len(sources)))
    for i, name in enumerate(targets):
        if name in sources:
            local[i, sources.index(name)] = 1.
            continue
        d2 = ((positions - montage[name]) ** 2).sum(1)
        idx = np.argsort(d2)[:3]
        weights = 1 / d2[idx]
        local[i, idx] = weights / weights.sum()
    np.testing.assert_allclose(local.sum(1), 1.)
    return spline[0], local


def build(index):
    with np.load(DATA) as z:
        people, days, onsets, y, targets = (z[k] for k in ('subject','session','onset','y','ch_names'))
        person = 'Scherer2015/' + ext.SCHERER_IDS[index]
        frozen = z['X'][people == person]
    sources = ext.SCHERER_CH
    kept = [i for i, c in enumerate(targets) if c in sources]
    assert len(kept) == 19 and len(sources) == 30
    _, local = maps(sources, list(targets))
    held = [sources.index(targets[i]) for i in kept]
    held_weights = np.zeros((2, len(held), len(sources)))
    for i, ch in enumerate(held):
        others = [j for j in range(len(sources)) if j != ch]
        a, b = maps([sources[j] for j in others], [sources[ch]])
        held_weights[:,i,others] = np.vstack([a, b])
    squares = np.zeros((2,len(held)))
    blocks, errors, n = [], [], 0
    for p, day, w, ch, labels, cues in ext.scherer(targets, [index+1], False):
        rows = np.flatnonzero((people == p) & (days == day))
        assert p == person and ch == sources
        np.testing.assert_array_equal(onsets[rows], cues)
        np.testing.assert_array_equal(y[rows], labels)
        original, _ = ext.to_proteus(w, sources, targets)
        ref = frozen[np.searchsorted(np.flatnonzero(people == person), rows)]
        err = np.max(np.abs(original - ref))
        assert err <= 1e-4, err
        alternative = (local @ w).astype('float32')
        np.testing.assert_array_equal(alternative[:,kept], ref[:,kept])
        assert np.max(np.abs(alternative - ref)) > 1.
        blocks.append(alternative)
        errors.append(float(err))
        for j, weights in enumerate(held_weights):
            squares[j] += ((weights @ w - w[:,held])**2).sum((0,2))
        n += len(w)*w.shape[-1]
    output = np.concatenate(blocks)
    assert output.shape == frozen.shape and np.isfinite(output).all()
    np.save(f'p{index}.npy',output)
    Path(f'p{index}.json').write_text(json.dumps(dict(max_error_uv=max(errors),
        held=[sources[k] for k in held], held_rms_uv=np.sqrt(squares/n).tolist())))


def assemble():
    with np.load(DATA) as z:
        meta = {k:z[k] for k in z.files if k != 'X'}
        data = z['X']
    for i, sid in enumerate(ext.SCHERER_IDS):
        data[meta['subject']=='Scherer2015/'+sid] = np.load(f'p{i}.npy')
    np.savez('local.npz', X=data, **meta)


if __name__ == '__main__':
    assemble() if sys.argv[1]=='assemble' else build(int(sys.argv[1]))
