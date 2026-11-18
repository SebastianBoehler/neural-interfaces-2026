"""Compare two query cohorts under the same frozen five-person training panel."""
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

BASE = Path('/mnt/lustre/work/gehler/gfh527')
OLD = BASE/'ni26_burst407_v1_20261118'
MOTOR = BASE/'ni26_burst407_motor_v1_20261118'
sys.path.insert(0, str(OLD))
import experiment407 as old

parent = old.parent
check, digest = parent.require, parent.digest
SOURCES = ('P04','P05','P06','P07','P08','P10')
DEV = tuple(old.source.HELD)
ROOT = Path(__file__).resolve().parent
check(digest(OLD/'source.sha256') == 'ac37d612169655d9c518188640810edf52a75850da17b899747352092200cb1f', 'source manifest')
for item in (OLD/'source.sha256').read_text().splitlines():
    h, name = item.split()
    check(digest(OLD/name) == h, name)


def fit(panel, smoke):
    assets, _, _ = old.source.setup()
    data = assets.CHAIN/'data/proteus_windows.npz'
    with np.load(data) as z:
        people = np.array([p.rsplit('/',1)[-1] for p in z['subject']])
        stream = np.char.add(np.char.add(z['subject'], '|'), z['session'])
        iface = z['interface']
    test_people = DEV + (SOURCES[panel],)
    target = np.flatnonzero(np.isin(people, test_people))
    train = np.flatnonzero(~np.isin(people, test_people))
    check(set(people[train]) == set(SOURCES)-{SOURCES[panel]}, 'panel leakage')
    check(digest(OLD/'fields.npz') == 'a53cba78199b91298f8420e4952d4786267262238e814f04afc00a2371ac51bb', 'field ancestry')
    with np.load(OLD/'fields.npz') as z:
        fields = [z[k] for k in z.files]
    labels = old.source.read_training_labels(data, train, len(people)).astype(int)
    if smoke:
        target = target[:5]
    result = old.predict(fields, train, target, labels, stream, iface)
    check(all(v.shape == (len(target),3) and np.isfinite(v).all() for v in result.values()), 'output')
    if smoke:
        return
    np.savez_compressed('members.npz', index=target, **result)
    card = dict(source=digest('source.sha256'), members=digest('members.npz'), train=sorted(set(people[train])),
                held=test_people, query_labels_opened=False)
    Path('source.json').write_text(json.dumps(card))


def members():
    c = json.loads(Path('source.json').read_text())
    check(c['source']==digest('source.sha256') and c['members']==digest('members.npz')
          and not c['query_labels_opened'], 'fit lineage')
    return c


def actor(smoke, index):
    check(digest(MOTOR/'motor407.py')=='7f21d26facc5aea6bad6bfa8a8967f1303ba03a8aee015e0390d5f856883aaca', 'motor code')
    sys.path.insert(0, str(MOTOR))
    import motor407
    # Compile the same insertion as §407b, expanding only its list of scored people.
    code = motor407.ACTOR.read_text()
    start = code.index('                mass=raw[target,1:].sum(axis=1)')
    end = code.index('            filtered=package.gate', start)
    code = code[:start]+'                new[target]=motor_update(raw[target],model[a])\n'+code[end:]
    a = ModuleType('panel_actor')
    exec(compile(code,str(motor407.ACTOR),'exec'), a.__dict__)
    a.ARMS, a.SRC, a.members = old.ARMS, Path.cwd(), members
    a.motor_update, a.HELD = motor407.motor_update, tuple(members()['held'])
    a.main(smoke,index)


if __name__ == '__main__':
    mode, panel, index = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.chdir(ROOT/str(panel))
    if mode in ('smoke','fit'):
        fit(panel, mode=='smoke')
    else:
        check(mode in ('actor','order'), 'mode')
        actor(mode=='actor', index)
