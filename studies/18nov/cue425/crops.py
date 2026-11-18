"""Recreate the registered Scherer crop then append verified consecutive samples."""
from pathlib import Path
import hashlib
import json
import sys

import mne
import numpy as np
from scipy.io import loadmat

BASE = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
sys.path.insert(0, str(BASE/'scripts'))
import export_external as ext


def build(index):
    sid = ext.SCHERER_IDS[index]
    rawfile = BASE/f'data/scherer2015/raw/{sid}.mat'
    with np.load(BASE/'data/scherer2015/windows.npz') as z:
        people, days, gold, onsets, targets = (z[k] for k in ('subject','session','y','onset','ch_names'))
        reference = z['X'][people == f'Scherer2015/{sid}']
    idx_person = np.flatnonzero(people == f'Scherer2015/{sid}')
    blocks = []
    max_error = 0.
    for sess in loadmat(rawfile, simplify_cells=True)['data']:
        rows = idx_person[days[idx_person] == str(sess['session'])]
        labels = np.asarray(sess['y']).astype(int)
        sel = np.isin(labels, list(ext.SCHERER_MAP))
        cues = np.asarray(sess['trial'])[sel]/float(sess['fs']) + ext.SCHERER_CUE
        np.testing.assert_array_equal(cues,onsets[rows])
        assert [ext.SCHERER_MAP[y] for y in labels[sel]] == gold[rows].tolist()
        info = mne.create_info(ext.SCHERER_CH, float(sess['fs']), 'eeg')
        raw = ext.preprocess(mne.io.RawArray(sess['X'].T * 1e-6, info, verbose=False))
        signal = raw.get_data() * 1e6
        starts = np.round((cues+1.)*ext.SFREQ).astype(int)
        segment = np.stack([signal[:,start:start+540] for start in starts])
        mapped, _ = ext.to_proteus(segment, ext.SCHERER_CH, targets)
        frozen = reference[np.searchsorted(idx_person,rows)]
        err = float(np.max(np.abs(mapped[:,:,:480] - frozen)))
        max_error = max(max_error,err)
        assert err <= 1e-4, f'{sid} session {sess["session"]}, baseline error {err} µV'
        continuous = np.concatenate([frozen, mapped[:,:,480:]], axis=2)
        blocks.append((rows,continuous))
    assert np.array_equal(np.concatenate([r for r,_ in blocks]),idx_person)
    np.save(f'{sid}.npy', np.concatenate([v[:,:,480:] for _,v in blocks]))
    Path(f'{sid}.json').write_text(json.dumps(dict(rows=idx_person.tolist(), max_error_uv=max_error,
        source_mat=hashlib.sha256(rawfile.read_bytes()).hexdigest())))


def assemble():
    path = BASE/'data/scherer2015/windows.npz'
    with np.load(path) as z:
        meta = {k:z[k] for k in z.files if k!='X'}
        x = z['X']
    tail = np.empty((*x.shape[:2],60), np.float32)
    seen = np.zeros(len(x),bool)
    for sid in ext.SCHERER_IDS:
        rows = json.loads(Path(f'{sid}.json').read_text())['rows']
        assert (meta['subject'][rows] == f'Scherer2015/{sid}').all() and not seen[rows].any()
        tail[rows] = np.load(f'{sid}.npy')
        seen[rows] = True
    assert seen.all() and np.isfinite(tail).all()
    for offset in (30,60):
        shifted = np.concatenate([x[:,:,offset:],tail[:,:,:offset]],axis=2)
        assert np.array_equal(shifted[:,:,:480-offset],x[:,:,offset:])
        np.savez(f'shift{offset}.npz',X=shifted,**meta)


if __name__=='__main__':
    assemble() if sys.argv[1]=='assemble' else build(int(sys.argv[1]))
