"""Build disjoint windows from versioned Wang segments with real filter context."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np
from scipy import signal

BASE = Path('/mnt/lustre/work/gehler/gfh527')
SOURCE = BASE / 'ni26_wang399_v2_20261118'
sys.path.insert(0, str(SOURCE))
from fetch import get, TAG, STORE, TASKS


def table(key):
    return list(csv.DictReader(io.StringIO(get(TAG + key)[0].decode()), delimiter='\t'))


def filter_crop(wave, margin):
    trim = (8-margin)*500
    segment = wave[:, trim:wave.shape[1]-trim]
    sos = signal.butter(6, (.5, 45), fs=500, btype='bandpass', output='sos')
    filtered = signal.sosfiltfilt(sos, segment)
    sampled = signal.resample_poly(filtered, 6, 25, axis=1)
    return sampled[:, margin*120:-margin*120].reshape(41, 6, 480).transpose(1, 0, 2)


def main(person, smoke):
    subject = f'sub-{person:02d}'
    proof = json.loads((SOURCE / f'{subject}.json').read_text())
    with np.load(BASE / 'neural-interfaces-2026/data/proteus_windows.npz') as z:
        target = z['ch_names'].tolist()
    with np.load(SOURCE / f'{subject}.npz') as z:
        retained, canonical, ranks = z['x'], z['names'].tolist(), z['order']
    waves, evidence = [], []
    for visit in ((1,) if smoke else (1, 2, 3)):
        blocks = []
        for ti, task in enumerate(TASKS):
            r, = [p for p in proof if (p['task'], p['session'], p['seconds']) == (task, visit, 150)]
            stem = f'{subject}/ses-session{visit}/eeg/{subject}_ses-session{visit}_task-{task}_'
            channels = table(stem+'channels.tsv')
            ch = [c['name'] for c in channels]
            assert len(set(ch)) == len(ch) and set(ch) == set(canonical)
            assert all(c['units'] in ('uV', 'µV') for c in channels)
            assert set(target)-set(ch) == {'FCz'}
            for e in table(stem+'events.tsv'):
                if e['type'] != 'SyncStatus':
                    assert float(e['onset']) >= 174 or float(e['onset'])+float(e['duration']) < 134
            n = len(ch)
            a, b = 134*500*n*4, 174*500*n*4-1
            cache = Path(f"{r['md5']}_{a}_{b}.raw")
            if cache.exists():
                payload = cache.read_bytes()
                assert hashlib.sha256(payload).hexdigest() == cache.with_suffix('.sha256').read_text()
            else:
                url = STORE+stem+'eeg.eeg?versionId='+quote(r['version'], safe='')
                payload, status, headers = get(url, {'Range':f'bytes={a}-{b}'})
                assert status == 206 and headers['etag'].strip('"') == r['md5']
                assert headers['content-range'] == f'bytes {a}-{b}/{300*500*n*4}'
                cache.write_bytes(payload)
                cache.with_suffix('.sha256').write_text(hashlib.sha256(payload).hexdigest())
            assert len(payload) == b-a+1
            wave = np.frombuffer(payload, '<f4').reshape(20000, n).T
            center = wave[:, 8000:10000]
            assert hashlib.sha256(center.T.tobytes()).hexdigest() == r['range_sha256']
            assert np.array_equal(center[[ch.index(c) for c in canonical]], retained[visit-1, ti, 1])
            raw = np.zeros((41, 20000))
            for j, c in enumerate(target):
                if c != 'FCz':
                    raw[j] = wave[ch.index(c)]
            x, shorter = [filter_crop(raw, m) for m in (8, 4)]
            assert np.isfinite(x).all() and x.shape == (6, 41, 480)
            blocks.append(x.astype(np.float32))
            evidence.append(dict(visit=visit, task=task, squared_diff=float(np.square(x-shorter).sum()),
                                 squared_signal=float(np.square(x).sum()), sha256=hashlib.sha256(payload).hexdigest()))
        waves.append(np.concatenate(blocks))
    ratio = np.sqrt(sum(e['squared_diff'] for e in evidence)/sum(e['squared_signal'] for e in evidence))
    name = 'smoke' if smoke else subject
    np.savez_compressed(f'{name}.npz', x=np.stack(waves), names=target, ranks=ranks)
    Path(f'{name}.json').write_text(json.dumps(dict(relative_rms=ratio, ranges=evidence)))


if __name__ == '__main__':
    main(int(sys.argv[1]), len(sys.argv)>2)
