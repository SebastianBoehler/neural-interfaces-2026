"""Fit Wang alone and commit all continuous target scores before evaluating labels."""
from itertools import combinations,product
import json
from pathlib import Path
import sys
import numpy as np
from common import BANDS, C4, DATA
from hashlib import sha256


def features(x,sf):
    # Both the source and target cover exactly four seconds, hence identical frequency bins.
    end=x.shape[-1]
    assert end==4*sf
    tapered=(x-x.mean(-1,keepdims=True))*np.hanning(end)
    spectra=np.fft.rfft(tapered,axis=-1)
    ia,ib=np.array(list(combinations(range(x.shape[1]),2))).T
    freq=np.fft.rfftfreq(end,d=1/sf)
    band=[]
    for lo,hi in BANDS:
        select=(freq>=lo)&(freq<hi)
        z=spectra[:,:,select]
        band.append(np.abs(z[:,ia]-z[:,ib]).__pow__(2).sum(-1))
    power=np.stack(band,-1)
    out=np.log((power+1e-12)/(power.sum(-1,keepdims=True)+6e-12)).reshape(len(x),-1)
    assert np.isfinite(out).all()
    return out


def direction(q,signs=None):
    # q = person x matched observations x channels/bands
    var=np.var(q,axis=(0,1));reg=.5*np.median(var)
    assert reg>0
    person=q.mean(1)
    mean=person.mean(0)[None,:] if signs is None else signs@person/len(person)
    w=mean/(var+reg)
    return w/np.linalg.norm(w,axis=1,keepdims=True)


def cases(x,order):
    effect=x[:,:,0]-.5*(x[:,:,1]+x[:,:,2])
    foil=x[:,:,1]-x[:,:,2]
    time=x[:,:,:,-1]-x[:,:,:,0]
    # order = person, session, task number; gather latest/earliest cognitive block.
    ids=np.indices(order.shape[:2])
    rank=x[ids[0],ids[1],order.argmax(2)]-x[ids[0],ids[1],order.argmin(2)]
    return effect,foil,time,rank


def source_auc(x,order):
    out=[[],[]]
    for i in range(len(x)):
        others=np.arange(len(x))!=i
        effect=cases(x[others],order[others])[0]
        for k,ds in enumerate((effect,effect[:,:1])):
            w=direction(ds.reshape(others.sum(),-1,x.shape[-1]))[0]
            subset=x[i] if k==0 else x[i,2:3]
            vals=subset@w
            ds=vals[:,0,:,None]-vals[:,1:].reshape(len(subset),1,-1)
            out[k].append(float(100*((ds>0)+.5*(ds==0)).mean()))
    return out


def main(corpus):
    with np.load(DATA[corpus]) as z:
        tnames=z['ch_names'].tolist()
        tnative=set(tnames)-set(z['interpolated'] if 'interpolated' in z.files else [])
        tx=z['X']
    dat=[];order=[];names=None
    for person in range(2,8):
        with np.load(f'sub-{person:02}.npz') as z:
            ch=z['names'].tolist()
            if names is None:names=[n for n in tnames if n in tnative and n in ch]
            assert all(n in ch for n in names)
            dat.append(z['x'][:,:,:,[ch.index(n) for n in names],:])
            order.append(z['order'])
    assert len(names)==(40 if corpus=='P' else 18)
    dat=np.stack(dat);order=np.stack(order)
    out={};info={}
    for tag,montage in [('4',C4),('wide',names)]:
        indices=[names.index(n) for n in montage]
        waves=dat[:,:,:,:,indices,:].reshape(-1,len(indices),2000)
        f=np.concatenate([features(waves[i:i+4],500) for i in range(0,len(waves),4)])
        f=f.reshape(*dat.shape[:4],-1)
        assert f.shape[:4]==(6,3,3,3)
        txm=tx[:,[tnames.index(n) for n in montage]]
        tf=np.concatenate([features(txm[i:i+8],120) for i in range(0,len(tx),8)])
        assert np.allclose(tf[[5,1]],features(txm[[5,1]],120))
        effects=cases(f,order)
        signs=np.array(list(product((-1.,1.),repeat=6)))
        flatten=lambda v:v.reshape(6,-1,f.shape[-1])
        weights=direction(flatten(effects[0]),signs)
        assert np.allclose(weights[-1],direction(flatten(effects[0]))[0])
        out[tag+'null']=-tf@weights.T
        for name,q in zip(('arith','foil','drift','order'),effects):
            out[tag+name]=-tf@direction(flatten(q))[0]
        for i in range(3):
            out[tag+'session'+str(i+1)]=-tf@direction(flatten(effects[0][:,i]))[0]
        info[tag]=dict(source_auc=source_auc(f,order),channels=montage)
    np.savez('scores'+corpus+'.npz',**out)
    info['bank_sha256']=sha256(Path('scores'+corpus+'.npz').read_bytes()).hexdigest()
    Path('proof'+corpus+'.json').write_text(json.dumps(info))
if __name__=='__main__':main(sys.argv[1])
