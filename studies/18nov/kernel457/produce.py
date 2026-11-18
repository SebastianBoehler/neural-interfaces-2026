"""Blind returning replay of the held-person spectral kernel correction."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import score379 as ref
from fit import features, SpectralMemory
from assets import ORIGINAL

PRIOR = ref.WORK/'ni26_return455_v3_20261118'
MEMORY = ref.WORK/'ni26_memory454_20261118'
DATA = ref.WORK/'ni26_aux431_20261118/data/proteus_windows.npz'
ORDERS = ref.WORK/'ni26_aux431_20261118/runs/shuffle_perms.npz'
SEEDS = (454, 1454)
PEOPLE = ('P04', 'P10')


def load_json(path, digest):
    assert ref.sha(path) == digest, path
    return json.loads(path.read_text())


def stock(order, perm):
    receipt = load_json(PRIOR/'aux431_result.json','5b6573ee9abe13a281916b6419b791215975dc2dc9122938d71f9dabcc269c9e')
    exact = load_json(PRIOR/'result.json','77b4579e92a5f26036da14253c4e328c9df5b71f2930bd8e656bec5250a49b9a')
    os.environ['PROTO'] = 'seen_session'
    sys.path.insert(0, str(ORIGINAL/'studies/14nov/rundetect'))
    import package as k
    assert k.PROTO == 'seen_session' and np.array_equal(k.ROWS,np.sort(perm))
    path = PRIOR/'inputs'/(order+'.npz')
    assert ref.sha(path) == receipt['output_sha256'][order]
    with np.load(path) as f:
        rows = f['rows']
        assert np.array_equal(rows,k.ROWS)
        raw = np.full((k.C.N,3),1/3)
        raw[rows] = f['three']
    hc = k.J.hyp_cache(order,'wprior',k.X,k.LAM,sessions=k.SESS,proto=k.PROTO)
    filt = k.gate(raw,k.J.filt_joint(raw,hc,'urn'),order,k.latch(order))
    path = PRIOR/(order+'.npz')
    assert ref.sha(path) == exact['records'][order]['output']
    with np.load(path) as f:
        assert np.array_equal(f['rows'],rows) and np.array_equal(f['stock'],filt[rows].argmax(1))
    assert np.allclose(filt[rows].sum(1),1) and (filt[rows]>=0).all()
    return rows,filt,receipt['recognition'][order]


@torch.inference_mode()
def run(order, smoke):
    assert ref.sha(DATA)==ref.DATA_SHA and ref.sha(ORDERS)==ref.ORDER_SHA
    with np.load(DATA) as f:
        names = f['subject']
        streams = np.char.add(np.char.add(names,'|'),f['session'])
        x = torch.cat([features(torch.from_numpy(p).float()) for p in np.array_split(f['X'],20)])
    with np.load(ORDERS) as f:
        seen = f[f'{order[:-1]}|{order[-1]}|seen_session']
        old = f[f'{order[:-1]}|{order[-1]}|all_loso']
    rows,prob,routes = stock(order,seen)
    outputs = {'rows':rows,'stock':prob[rows].argmax(1),'recognized':np.zeros(len(rows),bool)}
    parity = []
    for seed in SEEDS:
        for arm in ('memory','protected','donor'):
            outputs[f'{seed}_{arm}'] = np.full(len(rows),-1,np.int8)
        for person in PEOPLE:
            stem = f'kernel_{seed}_{person}'
            checkpoint = Path('weights')/(stem+'.pt')
            record = json.loads((MEMORY/(stem+'.json')).read_text())
            assert (record['seed'],record['kind'],record['held'])==(seed,'kernel',person)
            assert record['orders']==ref.ORDER_SHA and record['data']==ref.DATA_SHA
            assert ref.sha(checkpoint)==record['checkpoint'] and record['source']==ref.sha(MEMORY/'actor_source.sha256')
            assert set(record['train_people'])==set(ref.PEOPLE)-{person}
            model = SpectralMemory('kernel').eval()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
            held = np.flatnonzero(names=='Dreyer2026Proteus/'+person)
            src = np.flatnonzero(names!='Dreyer2026Proteus/'+person)
            encoded = model.encoder(((x-x[src].mean(0))/x[src].std(0).clamp_min(.05)).clamp(-8,8))
            writes = model.writes(encoded)
            sessions = [src[streams[src]==s] for s in np.unique(streams[src])]
            centroid = torch.stack([writes[idx].mean(0) for idx in sessions]).mean(0)
            bank = MEMORY/(stem+'.npz')
            assert ref.sha(bank)==record['output']
            with np.load(bank) as f:
                assert np.array_equal(f['index'],held)
                reference = f['memory'][ref.ORDERS.index(order)]
            for stream,stop in routes.items():
                if stream.split('|')[0] != 'Dreyer2026Proteus/'+person:
                    continue
                old_rows = old[streams[old]==stream]
                check = model.sequence(encoded[old_rows],writes[old_rows])[0].numpy()
                delta = np.max(abs(reference[np.searchsorted(held,old_rows)]-check))
                assert delta<.0002,delta
                parity.append(float(delta))
                seq = seen[streams[seen]==stream]
                pos = np.searchsorted(rows,seq)
                assert np.array_equal(rows[pos],seq)
                source = model.sequence(encoded[seq],centroid.expand(len(seq),-1))[0]
                donor = writes[sessions[0][np.arange(len(seq))%len(sessions[0])]]
                recognized = np.arange(len(seq))>=stop
                outputs['recognized'][pos] = recognized
                with np.errstate(divide='ignore'):
                    logp = np.log(prob[seq])
                for arm,w in [('memory',writes[seq]),('donor',donor)]:
                    logits = model.sequence(encoded[seq],w)[0]
                    diff = (logits-source).double().numpy()
                    correction = .1*np.clip(diff-diff.mean(1,keepdims=True),-1,1)
                    assert np.all(correction[0]==0)
                    pred = (logp+correction).argmax(1)
                    outputs[f'{seed}_{arm}'][pos] = pred
                    if arm=='memory':
                        outputs[f'{seed}_protected'][pos] = np.where(recognized,outputs['stock'][pos],pred)
        assert all((outputs[f'{seed}_{arm}']>=0).all() for arm in ('memory','protected','donor'))
    print('PASS',order,'rows',len(rows),'logit parity',max(parity),flush=True)
    if not smoke:
        path=Path(order+'.npz')
        np.savez_compressed(path,**outputs)
        path.with_suffix('.json').write_text(json.dumps({'blind':ref.sha(path),'source':ref.sha(Path('source.sha256')),
            'max_logit_difference':max(parity)}))


if __name__=='__main__':
    run(sys.argv[1],sys.argv[2:]==['smoke'])
