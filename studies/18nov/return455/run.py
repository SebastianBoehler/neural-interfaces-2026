"""Replay the stock personalizer against genuine returning recognition stops."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import score379 as ref
from fusion_inputs import official
from assets import ORIGINAL, get_stage, returning_guard

OWNER = ref.WORK/'ni26_aux431_20261118'
PIN = '5b6573ee9abe13a281916b6419b791215975dc2dc9122938d71f9dabcc269c9e'


def data():
    xfile = OWNER/'data/proteus_windows.npz'
    assert ref.sha(xfile) == ref.DATA_SHA
    with np.load(xfile) as z:
        meta = {k:z[k] for k in ('subject','session','interface')}
        x = z['X']
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'),meta['session'].astype(str))
    return meta,x


def run(order, smoke):
    assert ref.sha(Path('aux431_result.json')) == PIN
    receipt = json.loads(Path('aux431_result.json').read_text())
    meta,x = data()
    cp_type, config = get_stage()
    evidence = returning_guard(ref.ORDERS, config['v6_cp_w0'])
    os.environ['PROTO'] = 'seen_session'
    sys.path.insert(0, str(ORIGINAL/'studies/14nov/rundetect'))
    import package as detector
    assert detector.PROTO == 'seen_session'
    archived = Path('inputs')/(order+'.npz')
    assert ref.sha(archived) == receipt['output_sha256'][order]
    with np.load(archived) as z:
        rows, raw, route = z['rows'], z['three'], json.loads(str(z['route']))
    assert route == receipt['recognition'][order]
    assert np.array_equal(rows, detector.ROWS)
    path = OWNER/'runs/shuffle_perms.npz'
    assert ref.sha(path) == ref.ORDER_SHA
    with np.load(path) as z:
        perm = z[f'{order[:-1]}|{order[-1]}|seen_session']
    assert np.array_equal(np.sort(perm), np.sort(rows))
    raw_bank = np.full((len(x),3),1/3)
    raw_bank[rows] = raw
    cache = detector.J.hyp_cache(order,'wprior',detector.X,detector.LAM,sessions=detector.SESS,proto=detector.PROTO)
    filt = detector.gate(raw_bank, detector.J.filt_joint(raw_bank,cache,'urn'),order,detector.latch(order))
    output = np.full((len(x),3), np.nan)
    details = {}
    for stream,stop in route.items():
        delivery = perm[meta['stream'][perm] == stream]
        session_ids=np.unique(detector.C.SES[delivery])
        assert len(delivery)>0 and len(session_ids)==1
        assert np.array_equal(delivery,detector.C.order_rows(order,session_ids[0],detector.PROTO))
        cp = cp_type(config)
        # Native p/r/stack only modify class evidence when CP emits its readout.
        def never_read(*_):
            raise AssertionError('native readout active on a returning row')
        cp.predict = never_read
        for i,r in enumerate(delivery):
            output[r] = cp.step(x[r],raw_bank[r],filt[r],i >= stop)
        assert cp.stopped==stop and cp.fused==0 and cp.log['err']==0 and not cp.dead
        assert np.array_equal(output[delivery],filt[delivery])
        details[stream] = dict(rows=len(delivery),stop=stop,refits=cp.refits,active=cp.fused)
    assert np.isfinite(output[rows]).all()
    if not smoke:
        dest = Path(order+'.npz')
        np.savez_compressed(dest,rows=rows,stock=filt[rows].argmax(1),native=output[rows].argmax(1))
        dest.with_suffix('.json').write_text(json.dumps(dict(output=ref.sha(dest),source=ref.sha(Path('source.sha256')),
            sessions=details,conditional_runtime_stops={k:v for k,v in evidence.items() if k.endswith('|'+order)})))


def score():
    meta,_ = data()
    table = np.zeros((2,8))
    recorded = {}
    for i,o in enumerate(ref.ORDERS):
        path = Path(o+'.npz')
        info = json.loads(path.with_suffix('.json').read_text())
        assert ref.sha(path)==info['output'] and ref.sha(Path('source.sha256'))==info['source']
        with np.load(path) as z:
            assert np.array_equal(z['stock'],z['native'])
            recorded[o] = (z['rows'],z['native'],info)
    with np.load(OWNER/'data/proteus_windows.npz') as z:
        y = z['y']
    metric = official()
    for i,o in enumerate(ref.ORDERS):
        rows,labels,_ = recorded[o]
        for j,p in enumerate(('P04','P10')):
            mask = (meta['subject']=='Dreyer2026Proteus/'+p)&(meta['session'].astype(str)!='1')
            idx = np.flatnonzero(mask)
            inverse=np.full(len(y),-1,int)
            inverse[rows]=np.arange(len(rows))
            pos=inverse[idx]
            assert np.all(pos>=0)
            assert np.array_equal(rows[pos],idx)
            m = metric(y[mask],labels[pos],meta,mask)
            assert m['n_cells']==4
            table[j,i]=100*m['balanced_accuracy_cell_mean']
    persons=(table[:,:4].mean(1)+table[:,4:6].mean(1)+table[:,6:].mean(1))/3
    refscore=json.loads(Path('aux431_result.json').read_text())['three_minus_old']['official']['new_by_person']
    assert np.allclose(persons,refscore,rtol=0,atol=1e-9)
    total=sum(len(x[0]) for x in recorded.values())
    assert total==6096
    Path('result.json').write_text(json.dumps(dict(source=ref.sha(Path('source.sha256')),decisions=total,
        orders=ref.ORDERS,person_order=table.tolist(),people=persons.tolist(),mix=float([.2,.8]@persons),
        changed=0,records={o:info for o,(_,_,info) in recorded.items()}),indent=2))


if __name__=='__main__':
    if sys.argv[1]=='score':
        score()
    else:
        run(sys.argv[1],len(sys.argv)>2)
