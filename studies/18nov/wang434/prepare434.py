"""Verify Wang release ranges and prepare the four pre-registered fresh participants."""
import configparser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from prepare import filter_crop, table
from fetch import current_file, get, TAG, TASKS

SEQUENCES = {8: [5, 4, 3], 9: [5, 4, 3], 22: [3, 5, 4], 23: [3, 5, 4]}
BASE = Path('/mnt/lustre/work/gehler/gfh527')


def block(stem, target):
    raw, size, md5, version = current_file(stem+'eeg.eeg')
    header = get(current_file(stem+'eeg.vhdr')[0])[0].decode('utf-8-sig')
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read_string(header[header.index('[Common Infos]'):])
    info = cfg['Common Infos']
    assert float(info['SamplingInterval']) == 2000 and info['DataOrientation'] == 'MULTIPLEXED'
    assert cfg['Binary Infos']['BinaryFormat'] == 'IEEE_FLOAT_32'
    meta = json.loads(get(TAG+stem+'eeg.json')[0])
    assert (meta['EEGReference'],meta['RecordingDuration'],meta['SamplingFrequency']) == ('FCz',300,500)
    channels = table(stem+'channels.tsv')
    names = [c['name'] for c in channels]
    n = len(names)
    assert size == n*500*300*4 and n == int(info['NumberOfChannels'])
    assert len(set(names)) == n and set(target)-set(names) == {'FCz'}
    for j, c in enumerate(channels,1):
        fields = cfg['Channel Infos'][f'Ch{j}'].split(',')
        name, _, scale = fields[:3]
        assert (name == c['name'] or (name,c['name']) == ('Cpz','CPz')) and float(scale) == 1
        assert c['units'] in ('uV','µV')
        assert len(fields) == 3 or fields[3] in ('uV','µV')
    for e in table(stem+'events.tsv'):
        if e['type'] != 'SyncStatus':
            assert float(e['onset']) >= 174 or float(e['onset'])+float(e['duration']) < 134
    a, b = 134*500*n*4, 174*500*n*4-1
    cache = Path(f'{md5}_{a}_{b}.raw')
    if cache.exists():
        payload = cache.read_bytes()
        assert sha256(payload).hexdigest() == cache.with_suffix('.sha256').read_text()
    else:
        payload, status, h = get(raw, {'Range':f'bytes={a}-{b}'})
        assert status == 206 and h['etag'].strip('"') == md5 and h['content-range'] == f'bytes {a}-{b}/{size}'
        assert len(payload) == b-a+1
        cache.write_bytes(payload)
        cache.with_suffix('.sha256').write_text(sha256(payload).hexdigest())
    wave = np.frombuffer(payload, '<f4').reshape(20000,n).T
    raw = np.zeros((41,20000))
    for j, name in enumerate(target):
        if name != 'FCz':
            raw[j] = wave[names.index(name)]
    x, short = [filter_crop(raw, m) for m in (8,4)]
    assert np.isfinite(x).all()
    x32 = x.astype(np.float32)
    car = (x32 - x32.mean(axis=1,keepdims=True,dtype=np.float64)).astype(np.float32)
    ref = target.index('FCz')
    error = np.abs(car-car[:,ref:ref+1]-x32).max()
    assert error < 5e-5 and car.std(-1).min() > 1e-5
    return car, dict(stem=stem, md5=md5, version=version, sha256=sha256(payload).hexdigest(), bytes=len(payload),
                     squared_diff=float(np.square(x-short).sum()), squared_signal=float(np.square(x).sum()),
                     reference_error=float(error))


def main(person, smoke):
    assert json.loads(get(TAG+'dataset_description.json')[0])['License'] == 'CC0'
    roster = table('participants.tsv')
    subject = f'sub-{person:02d}'
    row, = [r for r in roster if r['participant_id'] == subject]
    ranks = [[int(row[f'{t}_order_session{v}']) for t in ('Subtraction','Memory','Music')] for v in (1,2,3)]
    assert [r[0] for r in ranks] == SEQUENCES[person]
    with np.load(BASE/'neural-interfaces-2026/data/proteus_windows.npz') as z:
        target = z['ch_names'].tolist()
    waves, proofs = [], []
    for v in ((1,) if smoke else (1,2,3)):
        tasks = []
        for task in TASKS:
            stem = f'{subject}/ses-session{v}/eeg/{subject}_ses-session{v}_task-{task}_'
            wave, proof = block(stem, target)
            tasks.append(wave)
            proofs.append(proof)
        waves.append(np.concatenate(tasks))
    rms = np.sqrt(sum(p['squared_diff'] for p in proofs)/sum(p['squared_signal'] for p in proofs))
    name = 'smoke' if smoke else subject
    Path(name+'.json').write_text(json.dumps(dict(relative_rms=rms,ranges=proofs)))
    assert rms < .01
    np.savez_compressed(name+'.npz', x=np.stack(waves), names=target, ranks=ranks)


if __name__ == '__main__':
    main(int(sys.argv[1]), len(sys.argv)>2)
