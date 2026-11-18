"""Insert a source-only cognitive-pair posterior before count filtering and CP."""
import json
import sys
from pathlib import Path
import numpy as np
BASE=Path('/mnt/lustre/work/gehler/gfh527')
SRC=BASE/'ni26_priv401_source_v1_20261118'
sys.path.insert(0,str(SRC))
from source import ARMS,HELD,INPUT,SCHEDULE,check,digest,setup


def members():
    check(digest(SRC/'source.sha256')=='587b2cae97ec18e463c17676ccc79ae4ec7178b60f91977087e68beef9c8de64','fit source')
    for line in (SRC/'source.sha256').read_text().splitlines():
        h,f=line.split(maxsplit=1)
        check(digest(SRC/f)==h,'fit lineage')
    card=json.loads((SRC/'source.json').read_text())
    check(card['members']==digest(SRC/'members.npz') and not card['query_labels_opened'] and
          card['source']==digest(SRC/'source.sha256'),'fit sealed')
    return card


def cp_run(cls,cfg,rr,features,raw,filt,stop=None):
    class CP(cls):
        def __del__(self):
            pass
    net=CP(cfg)
    net.feats=features.__getitem__
    out=np.stack([net.step(r,raw[r],filt[r],i==stop) for i,r in enumerate(rr)])
    h=len(rr) if stop is None else stop
    check(not net.dead and not net.log['err'] and net.refits==len(range(20,h,20)) and net.fused==max(0,h-90),'CP activation')
    return out


def main(smoke,index):
    card=members()
    assets,cls,cfg=setup()
    package=assets.get_count_filter()
    features=np.load(SCHEDULE/'features.npy',mmap_mode='r')
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        people=np.array([p.rsplit('/',1)[-1] for p in z['subject']])
    with np.load(SRC/'members.npz') as z:
        target=z['index']
        check(np.array_equal(target,np.flatnonzero(np.isin(people,HELD))),'held people')
        model={a:z[a] for a in ARMS}
    out={}
    with np.load(INPUT/'inputs.npz') as z,np.load(INPUT/'outputs.npz') as native:
        o=z['orders'][index]
        raw=z[o+'|raw']
        filt0=z[o+'|filtered']
        gold=z[o+'|expected']
        rr_all=np.split(z[o+'|index'],z[o+'|bounds'][1:-1])
        sessions=[package.C.order_rows(o,s,package.PROTO) for s in package.SESS]
        check(len(sessions)==len(rr_all) and all(np.array_equal(a,b) for a,b in zip(sessions,rr_all)),'independent pinned delivery')
        latch=package.latch(o)
        cache=package.J.hyp_cache(o,'wprior',package.X,package.LAM,sessions=package.SESS,proto=package.PROTO)
        for a in ('Z',)+ARMS:
            new=raw.copy()
            if a!='Z':
                mass=raw[target,1:].sum(axis=1)
                member=model[a][:,1:]
                q=member[:,0]/member.sum(1)
                new[target,1]=.90*raw[target,1]+.10*mass*q
                new[target,2]=mass-new[target,1]
                check(np.array_equal(new[:,0],raw[:,0]) and np.max(abs(new.sum(1)-raw.sum(1)))<1e-14
                      and np.max(abs(new-raw))>.01,'inactive pair or MI mass change')
            filtered=package.gate(new,package.J.filt_joint(new,cache,'urn'),o,latch)
            if a=='Z':
                check(np.array_equal(filtered,filt0),'count-filter zero')
            pred=np.full(len(raw),-1,dtype='int8')
            for rr in sessions:
                if people[rr[0]] not in HELD:
                    continue
                p=cp_run(cls,cfg,rr,features,new,filtered)
                pred[rr]=p.argmax(1)
                if a=='Z':
                    check(np.array_equal(p,gold[rr]) and np.array_equal(pred[rr],native[o+'|stock'][rr].argmax(1)),'CP/native zero')
                if smoke and a=='X':
                    short=cp_run(cls,cfg,rr[:131],features,new,filtered)
                    stop=cp_run(cls,cfg,rr[:131],features,new,filtered,63)
                    check(np.array_equal(short,p[:131]) and np.array_equal(stop,filtered[rr[:131]]),'CP prefix/stop')
            check((pred[target]>=0).all(),'missing rows')
            out[a]=pred[target]
    n='smoke' if smoke else 'order_'+str(index)
    np.savez_compressed(n+'.npz',index=target,**out)
    Path(n+'.json').write_text(json.dumps(dict(source=digest('source.sha256'),members=card['members'],order=str(o),outputs=digest(n+'.npz'))))


if __name__=='__main__':
    main(sys.argv[1]=='smoke',int(sys.argv[2]))
