"""Source-only kernel readouts and the pinned §401 upstream posterior experiment."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, cdist, squareform

from features import PAIRS, features

BASE = Path('/mnt/lustre/work/gehler/gfh527')
READOUT = BASE/'ni26_priv401_source_v1_20261118'
ACTOR = BASE/'ni26_priv401_full_v1_20261118'
ARMS = ('P','I','X','D')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


for parent, filename, expected in (
    (ACTOR,'full401.py','46f6576d7a165e1086810d6ed89cff051fe35f7a3292a55d4481a25dd65996bb'),
    (ACTOR,'score401.py','d85d87a2508c63be823e49bbf54db462260e70ae4456358bec786314cd7304af'),
    (READOUT,'source.py','bae5fa90cb0f91834284a095d813530e09e3608823d6116c87b9100259d25164')):
    require(digest(parent/filename)==expected, filename)
sys.path[:0] = [str(READOUT), str(ACTOR)]
import source
import full401


def source_run(smoke):
    assets, _, _ = source.setup()
    path = assets.CHAIN/'data/proteus_windows.npz'
    with np.load(path) as z:
        x = z['X']
        names = z['ch_names'].tolist()
        people = np.array([p.split('/')[-1] for p in z['subject']])
        target = np.flatnonzero(np.isin(people, source.HELD))
        train = np.flatnonzero(~np.isin(people, source.HELD))
        stream = np.char.add(np.char.add(z['subject'],'|'), z['session'])[train]
        interface = z['interface'][train]
    require(len(set(people[train]))==6 and len(set(people[target]))==4, 'people split')
    pairs = np.array([(names.index(a),names.index(b)) for a,b in PAIRS])
    probe = features(x[train[0]],pairs,True)
    rev = features(x[train[0]],pairs[:,::-1],True)
    for k in (1,2,3):
        np.testing.assert_allclose(rev[k], -probe[k], atol=1e-12)
    require(np.max(abs(probe[2]-probe[3]))>1e-3, 'inactive phase destruction')
    if smoke:
        for a,b in zip(probe,features(x[train[0]], pairs)):
            np.testing.assert_array_equal(a,b)
        return
    fields = [np.stack(v) for v in zip(*(features(row, pairs) for row in x))]
    y = source.read_training_labels(path,train,len(x)).astype(int)
    w = source.design_weights(stream,interface,y)
    result = {}
    for arm, keys in zip(ARMS, ((0,), (0,1), (0,1,2), (0,1,3))):
        matrix = np.column_stack([fields[k] for k in keys])
        scaler = StandardScaler().fit(matrix[train],sample_weight=w)
        a, b = scaler.transform(matrix[train]), scaler.transform(matrix[target])
        distances = pdist(a,'sqeuclidean')
        width = np.median(distances)
        require(width>0, 'kernel width')
        k = np.exp(-squareform(distances)/width)
        q = np.exp(-cdist(b,a,'sqeuclidean')/width)
        result[arm] = source.predict(k,q,w,np.eye(3)[y])
    require(all(np.isfinite(v).all() for v in result.values()), 'nonfinite readout')
    require(all(np.max(abs(result['X']-result[k]))>.001 for k in ('P','I','D')), 'inactive treatment')
    np.savez_compressed('members.npz',index=target,**result)
    card = dict(source=digest('source.sha256'),members=digest('members.npz'),
                train_people=sorted(set(people[train])),held=source.HELD,
                train_rows=digest_rows(train),query_labels_opened=False)
    Path('source.json').write_text(json.dumps(card,indent=2))


def digest_rows(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def members():
    card = json.loads(Path('source.json').read_text())
    require(card['source']==digest('source.sha256') and card['members']==digest('members.npz')
            and not card['query_labels_opened'], 'member ancestry')
    return card


def main():
    mode = sys.argv[1]
    if mode in ('smoke','fit'):
        source_run(mode=='smoke')
        return
    # Reuse the unchanged, digest-checked actor and official scorer as a pair.
    full401.ARMS, full401.SRC, full401.members = ARMS, Path.cwd(), members
    if mode == 'score':
        import score401
        score401.main()
    else:
        require(mode in ('actor-smoke','order'),'unknown mode')
        full401.main(mode=='actor-smoke', int(sys.argv[2]))


if __name__ == '__main__':
    main()
