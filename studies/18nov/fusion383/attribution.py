"""Attribute fixed fusion corrections and losses using whole official metric denominators."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_fusion383_v1_20261118')
sys.path.insert(0,str(ROOT))
from experiment import interval
from fusion_inputs import COHORT,FITS,CHAIN,official,pin,restore,sha

SUMMARY='7e80f60accf8e61d01c21704e0ba06cf4e00c8adfdce8fd9ba39332a64a96f3d'
SOURCE='9da18921a7e9d27807c21a511a7c27798c0f7f04f433833ce17419ac462c3522'


def checked(seed):
    pin(ROOT/'source.sha256',SOURCE)
    for line in (ROOT/'source.sha256').read_text().splitlines():
        digest,name=line.split(maxsplit=1)
        pin(ROOT/name,digest)
    pin(ROOT/'result.json',SUMMARY)
    saved=json.loads((ROOT/'result.json').read_text())['seeds'][str(seed)]
    path=ROOT/f'{seed}_result.json'
    pin(path,saved['result_sha'])
    result=json.loads(path.read_text())
    before=ROOT/f'{seed}_before_labels.json'
    pin(before,result['before'])
    doc=json.loads(before.read_text())
    path=ROOT/f'{seed}.npz'
    pin(path,doc['predictions'])
    return path,result['data']['JS|B2_pool']['contrasts']


def weights(gold,stream,iface):
    cells=np.char.add(np.char.add(stream,'|'),iface)
    unique=np.unique(cells)
    assert len(unique)==2
    w=np.zeros(len(gold))
    for cell in unique:
        for c in (0,1,2):
            k=(cells==cell)&(gold==c)
            assert k.any()
            w[k]=100/(len(unique)*3*k.sum())
    assert np.isclose(w.sum(),100)
    return w


def main(smoke):
    sc,_,meta,ages=restore()
    if not smoke:
        assert json.loads(Path('smoke.json').read_text())['source']==sha(Path('source.sha256'))
        with np.load(CHAIN/'data/proteus_windows.npz') as d:
            y=d['y']
        metric=official()
    out={}
    for seed in FITS:
        path,prior=checked(seed)
        modes=('J_real_5','J_donor_5')
        calc={mode:{(age,kind):np.empty((4,8)) for age in ('early','late','all')
                    for kind in ('corrected','destroyed')} for mode in modes}
        raw={mode:{k:0 for k in ('corrected','destroyed','other_change')} for mode in modes}
        with np.load(path) as z:
            for i,p in enumerate(COHORT):
                mask=meta['person']==p
                if not smoke:
                    gold=y[mask]
                    w=weights(gold,meta['stream'][mask],meta['interface'][mask])
                for j,o in enumerate(sc.ORDERS[:1] if smoke else sc.ORDERS):
                    key='|'.join((p,'JS',o,'B2_pool'))+'|'
                    zero=z[key+'zero']
                    early=ages[o][mask]<90
                    assert early.sum()==90 and (~early).any()
                    for mode in modes:
                        alt=z[key+mode]
                        assert len(zero)==len(alt)==mask.sum()
                        if smoke:
                            continue
                        plus=(alt==gold)&(zero!=gold)
                        minus=(alt!=gold)&(zero==gold)
                        other=(alt!=zero)&~(plus|minus)
                        for k,b in [('corrected',plus),('destroyed',minus),('other_change',other)]:
                            raw[mode][k]+=int(b.sum())
                        for age,ok in [('early',early),('late',~early),('all',np.ones_like(early))]:
                            for kind,b in [('corrected',plus),('destroyed',minus)]:
                                calc[mode][age,kind][i,j]=w[ok&b].sum()
                        selector=np.where(plus,alt,zero)
                        s0=metric(gold,zero,meta,mask)['balanced_accuracy_cell_mean']
                        oracle=metric(gold,selector,meta,mask)['balanced_accuracy_cell_mean']
                        assert abs(100*(oracle-s0)-w[plus].sum())<1e-9
        if not smoke:
            metrics={}
            for mode in modes:
                v=calc[mode]
                for kind in ('corrected','destroyed'):
                    assert np.allclose(v['all',kind],v['early',kind]+v['late',kind])
                net=interval(v['all','corrected']-v['all','destroyed'])['pp']
                assert abs(net-prior[mode+'-zero']['pp'])<1e-9
                metrics[mode]={'|'.join(k):interval(a) for k,a in v.items()}
            out[seed]=dict(counts=raw,metrics=metrics)
    dest=Path('smoke.json' if smoke else 'result.json')
    dest.write_text(json.dumps(dict(source=sha(Path('source.sha256')),results=out),indent=2))


if __name__=='__main__':
    main(sys.argv[1]=='smoke')
