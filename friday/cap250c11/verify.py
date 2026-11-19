"""Check paired worker decisions without loading target labels."""
import json
from pathlib import Path
import sys

from data import sha

ARMS=('stock','geometry','rotation')


def load(index,smoke):
    spec={}
    receipts={}
    for arm in ARMS:
        path=Path(f'{arm}_{index}{"_smoke" if smoke else ""}.json')
        h=sha(path)
        assert h==path.with_suffix('.sha256').read_text().strip()
        data=json.loads(path.read_text())
        assert data['source']==sha(Path('source.sha256'))
        spec[arm]=data['sessions']
        receipts[path.name]=h
    assert all(x.keys()==spec['stock'].keys() for x in spec.values())
    for name in spec['stock']:
        participants=[spec[a][name] for a in ARMS]
        first=participants[0]
        for v in participants:
            assert all(v[key]==first[key] for key in ('rows','inner','activity'))
            assert v['labels'][:20]==first['labels'][:20]
            assert len(v['events'])==len(first['events'])
    return spec,receipts


def events(stock,real,control):
    assert all(real[k]==control[k]==stock[k] for k in ('age','raw','weight_hash'))
    assert not stock['edited']
    assert real['edited']==real['proposed']==control['proposed']
    assert len(real['edited'])==len(control['edited'])
    raw=real['raw']
    assert sorted((raw[i],y) for i,y in real['edited'])==sorted((raw[i],y) for i,y in control['edited'])
    return real['edited']!=control['edited']


def smoke():
    affected,moved=0,0
    for idx in (2,5):
        by,receipts=load(idx,True)
        for session in by['stock']:
            triplet=[by[arm][session]['events'] for arm in ARMS]
            for s,g,c in zip(*triplet):
                moved+=events(s,g,c)
                affected+=len(g['edited'])
        print('smoke',idx,receipts,flush=True)
    assert affected>=5 and moved>0
    print('real hard-target edits',affected,'refits with moved controls',moved,flush=True)


if __name__=='__main__':
    assert sys.argv[1:]==['smoke']
    smoke()
