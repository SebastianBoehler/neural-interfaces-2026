"""Bind three paired source fits and the restored inference route before held exposure."""
import hashlib
import json
from pathlib import Path
import sys

PRIMARY=Path('/mnt/lustre/work/gehler/gfh527/ni26_joint382_v1_20261118')
SOURCE='67829e9fe023605f5426d1bb035f9b51f00e8fc026c9b6824fdae3035b92c5a9'
BUNDLE='06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
ROWS='882b5b97ec811d697540f6152be0d02eb49448e9e4c88c3d6c28fcb1433b2f1f'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock(root,seed):
    assert seed in (382,1382) and sha(PRIMARY/'source.sha256')==SOURCE
    records={}
    files={}
    for a in 'FJK':
        p=root/f'{a}_{seed}.json'
        w=p.with_suffix('.pt')
        d=json.loads(p.read_text())
        assert d['seed']==seed and d['steps']==120 and d['arm']==a and d['source_sha']==SOURCE
        assert d['rows']==ROWS and d['scope']==['P04','P05','P06','P07','P08','P10']
        assert d['held_labels_opened'] is False and d['checkpoint_sha']==sha(w)
        assert d.get('bundle_sha')==(BUNDLE if seed==1382 else None)
        assert set(d['groups'])==({'norm'} if a=='F' else {'body','head','norm'})
        for g in d['groups']:
            assert d['first_grad'][g]>0 and d['before'][g]!=d['after'][g]
        records[a]=d
        files.update({p.name:sha(p),w.name:d['checkpoint_sha']})
    assert len({x['batches'] for x in records.values()})==1
    assert records['F']['before']['norm']==records['J']['before']['norm']
    assert all(records['J']['before'][g]==records['K']['before'][g] for g in ('body','head'))
    return dict(root=str(root.resolve()),seed=seed,files=files)


if __name__=='__main__':
    root,seed=Path(sys.argv[1]),int(sys.argv[2])
    p=Path('fit_lock.json')
    with p.open('x') as f:
        json.dump(lock(root,seed),f,indent=2)
    print(sha(p))
