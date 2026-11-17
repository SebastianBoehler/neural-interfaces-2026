"""Replay one matched-work intervention against frozen and native endpoints."""
import inspect
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
INPUT = ROOT / 'ni26_native_schedule369_dev_v1_20261117'
OLD = ROOT / 'ni26_cp_schedule365_v1_20261117'
sys.path.insert(0, str(OLD))
from run import digest, owner

PINS = {
    INPUT/'inputs.npz': '6763c661e0d0c8b8a6eda90e0f3013fc5959fcd41250fa3e3bed2dbcae01ee78',
    INPUT/'outputs.npz': '0aeef0009deceaef16c0c519c8e787c0331bfc432a96eba28670ac4300d96d0e',
    OLD/'run.py': 'd7f550ec4c7189f12e5ddd349fafac7d3101eb91f3c0db7a9255eb9547575e45',
    OLD/'schedule.py': '7d46fac351fa9efcf9f7b3bd4036e4f7dc9c1fb83aa36c7fc8d8b169985db993',
}
SCHEDULES = ((80,100,120,140,160,180), (80,110,120,130,160,180))


def check(ok, message):
    if not ok:
        raise ValueError(message)


def prepare():
    for path, expected in PINS.items():
        check(digest(path) == expected, f'changed input {path}')
    assets, _ = owner()
    parent, cfg = assets.get_stage()
    check((cfg['v6_cp_t0'], cfg['v6_cp_k'], cfg['v6_cp_w0']) == (20,20,90), 'schedule')
    source = textwrap.dedent(inspect.getsource(parent.step))
    old = 'if i >= self.t0 and (i - self.t0) % self.k == 0:'
    check(source.count(old) == 1, 'refit predicate')
    scope = dict(parent.step.__globals__)
    exec(compile(source.replace(old, 'if i in self.schedule:'), '<refit375>', 'exec'), scope)
    class Decision(parent):
        step = scope['step']
        def refit(self, t):
            self.did.append(t)
            return super().refit(t)
        def __del__(self):
            pass
    return assets, Decision, cfg


def replay(cls, cfg, rows, feat, raw, filtered, schedule):
    model = cls(cfg)
    model.schedule, model.did, model.feats = schedule, [], feat.__getitem__
    out = np.stack([model.step(r, raw[r], filtered[r], False) for r in rows])
    check(tuple(model.did) == tuple(i for i in schedule if i < len(rows)), 'fit times')
    check(not model.dead and not model.log['err'] and model.fused == max(0,len(rows)-90), 'stage activity')
    return out


def run(smoke, shard):
    assets, cls, cfg = prepare()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        people = np.array([p.split('/')[-1] for p in z['subject']])
        if smoke:
            stage = cls(cfg)
            np.save('features.npy', np.stack([stage.feats(x) for x in z['X'].astype(np.float64)]))
    feat = np.load('features.npy', mmap_mode='r')
    check(feat.shape == (2655,8,41,41), 'covariance schema')
    if not smoke:
        proof = json.loads(Path('smoke.json').read_text())
        check(proof['features'] == digest('features.npy') and proof['source'] == digest('source.sha256'), 'smoke identity')
    pieces = {}
    with np.load(INPUT/'inputs.npz') as bank, np.load(INPUT/'outputs.npz') as native:
        orders = bank['orders'].tolist()
        for order in orders[shard%2::2]:
            raw, filt, expected, idx, bounds = [bank[order+'|'+k] for k in ('raw','filtered','expected','index','bounds')]
            for s,(a,b) in enumerate(zip(bounds[:-1],bounds[1:],strict=True)):
                rows = idx[a:b]
                if people[rows[0]] != sorted(set(people))[shard//2]:
                    continue
                check(len(set(people[rows])) == 1 and len(rows)>180, 'session ownership/count')
                stock, delay = [replay(cls,cfg,rows,feat,raw,filt,sched) for sched in SCHEDULES]
                check(np.array_equal(stock,expected[rows]), 'frozen zero mismatch')
                check(np.array_equal(stock.argmax(1),native[order+'|stock'][rows].argmax(1)), 'native zero mismatch')
                check(np.array_equal(stock[:100],delay[:100]), 'premature action')
                drift = .5*np.abs(raw[rows[60:80]].mean(0)-raw[rows[80:100]].mean(0)).sum()
                key = f'{order}|{s}|'
                pieces.update({key+'rows':rows,key+'fixed':stock.argmax(1),key+'delayed':delay.argmax(1),key+'drift':drift})
                if smoke:
                    short = replay(cls,cfg,rows[:131],feat,raw,filt,SCHEDULES[1])
                    check(np.array_equal(short,delay[:131]), 'prefix mismatch')
                    Path('smoke.json').write_text(json.dumps(dict(features=digest('features.npy'),source=digest('source.sha256'))))
                    return
    check(bool(pieces), 'empty shard')
    np.savez_compressed(f'shard_{shard}.npz', **pieces)


if __name__ == '__main__':
    run(sys.argv[1] == 'smoke', int(sys.argv[2]))
