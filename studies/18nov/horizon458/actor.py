"""Evaluate fixed exponentially decaying histories with each session isolated."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.signal import lfilter
from fit import features, SpectralMemory
from stocks456 import inputs, log_probability, MEMORY, ref

DECAY = 2**(-1/24)
PRIOR=ref.WORK/'ni26_native458_20261118'
PREDICTIONS='6b8100bca1897c85dc509bbeea12981d03714b484af59ba782f5e0141ef4912c'


def history(writes, decay):
    after=lfilter([1.],[1.,-decay],writes,axis=0)
    return np.concatenate((np.zeros_like(after[:1]),after[:-1]))


def read(query, prefix):
    table=prefix[:,:36].reshape(-1,12,3)
    return (query[:,:,None]*table).sum(1)/(20+(query*prefix[:,36:]).sum(1,keepdims=True))


def corrections(query, writes, donor, centroid, decay):
    src=read(query,history(np.broadcast_to(centroid,writes.shape),decay))
    out={}
    for name,values in [('memory',writes),('donor',donor)]:
        delta=read(query,history(values,decay))-src
        assert not delta[0].any()
        out[name]=.1*np.clip(delta-delta.mean(axis=1,keepdims=True),-1,1)
    return out


def feature_bank(person):
    data=ref.BASE/'data/proteus_windows.npz'
    assert ref.sha(data)==ref.DATA_SHA
    with np.load(data) as z:
        names=np.array([p.split('/')[-1] for p in z['subject']])
        streams=np.char.add(np.char.add(names,'|'),z['session'])
        x=torch.cat([features(torch.from_numpy(part).float()) for part in np.array_split(z['X'],24)])
    src=np.flatnonzero(names!=person)
    held=np.flatnonzero(names==person)
    standardized=((x-x[src].mean(0))/x[src].std(0).clamp_min(.05)).clamp(-8,8)
    sessions=[src[streams[src]==s] for s in np.unique(streams[src])]
    actors={}
    for seed in (454,1454):
        stem=f'kernel_{seed}_{person}'
        path=Path('weights')/(stem+'.pt')
        receipt=json.loads((MEMORY/(stem+'.json')).read_text())
        assert ref.sha(path)==receipt['checkpoint'] and receipt['source']==ref.sha(MEMORY/'actor_source.sha256')
        assert set(receipt['train_people'])==set(ref.PEOPLE)-{person}
        assert receipt['data']==ref.DATA_SHA and receipt['orders']==ref.ORDER_SHA
        model=SpectralMemory('kernel').eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        encoded=model.encoder(standardized)
        writes=model.writes(encoded)
        centroid=torch.stack([writes[rows].mean(0) for rows in sessions]).mean(0)
        query=(torch.nn.functional.elu(model.query(encoded))+1)/12
        actors[seed]=(query.numpy(),writes.numpy(),writes[sessions[0]].numpy(),centroid.numpy())
    return streams,held,actors


@torch.inference_mode()
def run(person,smoke):
    meta,ages,_,stock=inputs()
    streams,held,actors=feature_bank(person)
    np.testing.assert_array_equal(held,np.flatnonzero(meta['person']==person))
    path=ref.WORK/'ni26_aux431_20261118/runs/shuffle_perms.npz'
    assert ref.sha(path)==ref.ORDER_SHA and ref.sha(PRIOR/'pair_predictions.npz')==PREDICTIONS
    output={}
    with np.load(path) as perms,np.load(PRIOR/'pair_predictions.npz') as prior:
        for o in ref.ORDERS[:1] if smoke else ref.ORDERS:
            perm=perms[f'{o[:-1]}|{o[-1]}|all_loso']
            deltas={}
            for seed,(q,w,d,c) in actors.items():
                buffers={arm:np.full((len(held),3),np.nan) for arm in ('uniform','recent','mixed','mixed_donor')}
                for s in np.unique(streams[held]):
                    rows=perm[streams[perm]==s]
                    at=np.searchsorted(held,rows)
                    np.testing.assert_array_equal(ages[o][rows],np.arange(len(rows)))
                    args=(q[rows],w[rows],d[np.arange(len(rows))%len(d)],c)
                    long=corrections(*args,1.)
                    repeat=corrections(*args,1.)
                    recent=corrections(*args,DECAY)
                    buffers['uniform'][at]=(long['memory']+repeat['memory'])/2
                    buffers['recent'][at]=recent['memory']
                    buffers['mixed'][at]=(long['memory']+recent['memory'])/2
                    buffers['mixed_donor'][at]=(long['donor']+recent['donor'])/2
                    assert (abs(recent['memory']-long['memory'])>1e-5).any()
                assert all(np.isfinite(v).all() for v in buffers.values())
                deltas[seed]=buffers
            logp=log_probability(stock[o][held],ages[o][held])
            prefix=person+'|'+o+'|'
            output[prefix+'stock']=logp.argmax(1)
            for seed,arm in deltas.items():
                output[prefix+str(seed)]=(logp+arm['uniform']).argmax(1)
                np.testing.assert_array_equal(output[prefix+str(seed)],prior[prefix+str(seed)])
            for arm in buffers:
                diff=(deltas[454][arm]+deltas[1454][arm])/2
                output[prefix+arm]=(logp+diff).argmax(1)
            for arm,old in [('stock','stock'),('uniform','memory')]:
                np.testing.assert_array_equal(output[prefix+arm],prior[prefix+old])
    if not smoke:
        path=Path(person+'.npz')
        np.savez_compressed(path,**output)
        Path(person+'.json').write_text(json.dumps(dict(bank=ref.sha(path),source=ref.sha(Path('source.sha256')))))
    print('PASS',person,flush=True)


if __name__=='__main__':
    run(sys.argv[1],sys.argv[2:]==['smoke'])
