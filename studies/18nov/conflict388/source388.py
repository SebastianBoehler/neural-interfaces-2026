"""Measure source-person CE on disjoint episode schedules, with source people retained."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path.home()/'ni26/388'
sys.path.insert(0,str(ROOT))
from train388 import BUNDLE,BUNDLE_SHA,PEOPLE,sha

TRAIN_SHA='4bf417f9b5dcf1ba94215b7247c80664a6217b16e0e85b72ccbe76d2b066e8ca'


@torch.no_grad()
def main(smoke):
    assert 'H100' in torch.cuda.get_device_name() and sha(ROOT/'source.sha256')==TRAIN_SHA
    assert sha(BUNDLE/'bundle.json')==BUNDLE_SHA
    for p,h in json.loads((BUNDLE/'bundle.json').read_text())['files'].items():
        assert sha(BUNDLE/p)==h
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    net=torch.jit.load(str(BUNDLE/'learned.ts'),map_location='cuda').eval()
    with np.load(BUNDLE/'source.npz') as d:
        ss=d['streams']
        x,y,w=[torch.from_numpy(d[k]).cuda() for k in ('x','y','weights')]
    rng=np.random.default_rng(38888)
    episodes={}
    for p in PEOPLE:
        sources=np.unique(ss[np.char.startswith(ss,'Dreyer2026Proteus/'+p+'|')])
        assert len(sources) in (1,3)
        episodes[p]=[rng.choice(np.flatnonzero(ss==sources[i%len(sources)]),48,replace=False) for i in range(12)]
    out={}
    for name in (('init',) if smoke else ('init','N_382','H_382','X_382','N_1382','H_1382','X_1382')):
        if name!='init':
            r=json.loads((ROOT/(name+'.json')).read_text())
            path=ROOT/(name+'.pt')
            assert r['weights']==sha(path) and r['source']==TRAIN_SHA
            net.load_state_dict(torch.load(path,weights_only=True,map_location='cuda'))
        out[name]={}
        for p,rows in episodes.items():
            losses=[]
            for b in rows:
                logit=net(x[b])
                losses.append((F.cross_entropy(logit,y[b],reduction='none')*w[b]).mean().item())
            assert np.isfinite(losses).all()
            out[name][p]=float(np.mean(losses))
    hardest=max(out['init'],key=out['init'].get)
    if not smoke:
        sr=json.loads(Path('smoke.json').read_text())
        assert out['init']==sr['losses']['init'] and hardest==sr['hardest']
    path=Path('smoke.json' if smoke else 'result.json')
    path.write_text(json.dumps(dict(hardest=hardest,losses=out,source=sha(Path('source.sha256'))),indent=2))


if __name__=='__main__':
    main(sys.argv[1]=='smoke')
