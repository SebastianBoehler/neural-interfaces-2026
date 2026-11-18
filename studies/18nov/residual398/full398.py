"""Replay cap250 with all frozen residual and original k20 members at the CP selector."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/lustre/work/gehler/gfh527')
sys.path.insert(0,str(ROOT/'ni26_rank394b_v1_20261118'))
from rank394 import BAND,INPUT,SCHEDULE,SEEDS,PEOPLE,check,digest,setup,run
ARMS=('Z','N','A6','RT','RP','RU','CT','D')


def main(smoke,idx):
    s,p=SEEDS[idx//4],PEOPLE[idx%4]
    card=json.loads(Path(p+'.json').read_text())
    check(card['held']==p and not card['held_labels_opened'] and card['fits']==digest('fits.json')
          and card['outputs']==digest(p+'.npz') and card['source']==digest('source.sha256'),'actor identity')
    assets,cp,cfg=setup()
    feat=np.load(SCHEDULE/'features.npy',mmap_mode='r')
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        people=np.array([v.rsplit('/',1)[-1] for v in z['subject']])
    target=np.flatnonzero(people==p)
    pack={}
    seals=json.loads((BAND/'before_labels.json').read_text())
    with np.load(INPUT/'inputs.npz') as ctl,np.load(INPUT/'outputs.npz') as native,np.load(p+'.npz') as bank:
        check(np.array_equal(target,bank['index']),'global indices')
        for oi,o in enumerate(ctl['orders'][:1] if smoke else ctl['orders']):
            name=f'core_{2*sorted(set(people)).index(p)+oi%2}'
            check(digest(BAND/(name+'.npz'))==seals['new'][name],'paired near/stock archive')
            sessions=np.split(ctl[o+'|index'],ctl[o+'|bounds'][1:-1])
            matches=[(j,rr) for j,rr in enumerate(sessions) if np.array_equal(np.sort(rr),target)]
            check(len(matches)==1,'held ownership')
            sid,rr=matches[0]
            pos=np.searchsorted(target,rr)
            raw,filt,gold=[ctl[o+'|'+k][rr] for k in ('raw','filtered','expected')]
            check(np.array_equal(native[o+'|stock'][rr].argmax(1),gold.argmax(1)),'native labels')
            counts={}
            with np.load(BAND/(name+'.npz')) as old:
                pre=f'{o}|{sid}|'
                check(np.array_equal(old[pre+'rows'],rr),'pinned permutation')
                for a in ARMS:
                    key='A6|'+o if a in ('A6','Z','N') else f'{s}|'+('RT' if a=='D' else a)+'|'+o
                    wave=bank[key][pos]
                    pred,counts[a]=run(cp,cfg,feat[rr],raw,filt,wave,a,s)
                    check(np.array_equal(pred[:90],gold[:90]),'early change')
                    if a in ('N','Z'):
                        check(np.array_equal(pred.argmax(1),old[pre+('Z' if a=='Z' else 'N'+str(s+4))]),'near/stock replay')
                    if a=='Z':
                        check(np.array_equal(pred,gold),'original scores')
                    if smoke and a in ('RT','D'):
                        q,_=run(cp,cfg,feat[rr[:131]],raw[:131],filt[:131],wave[:131],a,s)
                        stop,_=run(cp,cfg,feat[rr[:131]],raw[:131],filt[:131],wave[:131],a,s,63)
                        check(np.array_equal(q,pred[:131]) and np.array_equal(stop,filt[:131]),'prefix/stop')
                    pack[o+'|'+a]=pred.argmax(1)[np.argsort(pos)].astype('int8')
            check(all(t==counts['N'] for k,t in counts.items() if k!='Z'),'dose')
    n='smoke' if smoke else f'full_{s}_{p}'
    np.savez_compressed(n+'.npz',index=target,**pack)
    Path(n+'.json').write_text(json.dumps(dict(source=digest('source.sha256'),outputs=digest(n+'.npz'))))


if __name__=='__main__':
    main(sys.argv[1]=='smoke',int(sys.argv[2]))
