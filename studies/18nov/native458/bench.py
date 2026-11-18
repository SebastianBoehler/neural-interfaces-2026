"""Cold session resets and warm one-window CUDA latency on authenticated real traces."""
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
import score379 as ref
from fit import features, SpectralMemory

BASE = ref.WORK
MEMORY = BASE/'ni26_memory454_20261118'
WEIGHTS = BASE/'ni26_kernel457_20261118/weights'
SEEDS = (454,1454)


def assets(person):
    data = ref.BASE/'data/proteus_windows.npz'
    orders = BASE/'ni26_aux431_20261118/runs/shuffle_perms.npz'
    assert ref.sha(data)==ref.DATA_SHA and ref.sha(orders)==ref.ORDER_SHA
    with np.load(data) as z:
        names = np.array([p.split('/')[-1] for p in z['subject']])
        streams = np.char.add(np.char.add(names,'|'),z['session'])
        raw = torch.from_numpy(z['X']).float()
    with np.load(orders) as z:
        perm = z['within|0|all_loso']
    held = np.flatnonzero(names==person)
    src = np.flatnonzero(names!=person)
    session = np.unique(streams[held])[0]
    rows = perm[streams[perm]==session]
    x = torch.cat([features(t.cuda()) for t in raw.split(128)])
    mean,scale = x[src].mean(0),x[src].std(0).clamp_min(.05)
    x = ((x-mean)/scale).clamp(-8,8)
    checkpoints,centroids,refs = [],[],[]
    for seed in SEEDS:
        stem = f'kernel_{seed}_{person}'
        path = WEIGHTS/(stem+'.pt')
        meta = json.loads((MEMORY/(stem+'.json')).read_text())
        assert ref.sha(path)==meta['checkpoint'] and set(meta['train_people'])==set(ref.PEOPLE)-{person}
        assert meta['orders']==ref.ORDER_SHA and meta['data']==ref.DATA_SHA
        model = SpectralMemory('kernel').cuda().eval()
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
        checkpoints.append(model.state_dict())
        writes = model.writes(model.encoder(x))
        centroids.append(torch.stack([writes[src[streams[src]==s]].mean(0) for s in np.unique(streams[src])]).mean(0))
        path = MEMORY/(stem+'.npz')
        assert ref.sha(path)==meta['output']
        with np.load(path) as z:
            np.testing.assert_array_equal(z['index'],held)
            refs.append(np.stack([z[a][0,np.searchsorted(held,rows)] for a in ('memory','source')],axis=1))
    prefixes=('encoder.0','encoder.1','head','query','key','value')
    weights=[torch.stack([c[p+'.'+v] for c in checkpoints]) for p in prefixes for v in ('weight','bias')]
    return raw[rows].pin_memory(),weights,torch.stack(centroids),mean,scale,np.stack(refs,axis=1),session


@torch.inference_mode()
def run(person,smoke):
    assert 'A100' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.ops.load_library(str(Path('decoder.so').resolve()))
    raw,weights,centroid,mean,scale,reference,session=assets(person)
    if smoke:
        raw,reference=raw[:12],reference[:12]
    frame=torch.empty_like(raw[0],device='cuda')
    hann=torch.hann_window(480,device='cuda')
    state=torch.zeros((2,2,48),device='cuda')
    parts=[[w[i:i+1] for w in weights] for i in range(2)]
    frame.copy_(raw[0])
    def classify(serial=False):
        x=torch.ops.ni458.feature(frame,hann,mean,scale)
        if serial:
            logits=torch.cat([torch.ops.ni458.step(x,parts[i],state[i:i+1],centroid[i:i+1]) for i in range(2)])
        else:
            logits=torch.ops.ni458.step(x,weights,state,centroid)
        diff=logits[:,0]-logits[:,1]
        correction=.1*(diff-diff.mean(1,keepdim=True)).clamp(-1,1)
        return logits,correction
    for _ in range(3):
        classify(True)
        classify()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fixed=classify()
    def execute(mode):
        if mode=='graph':
            graph.replay()
            return fixed
        return classify(mode=='serial')
    stages=('serial','batch','graph')
    validation={}
    target=None
    for mode in stages:
        state.zero_()
        traces,outputs=[],[]
        for row in raw:
            frame.copy_(row,non_blocking=True)
            logits,corr=execute(mode)
            traces.append(logits.cpu().numpy().copy())
            outputs.append(corr.cpu().numpy().copy())
        outputs=np.stack(outputs)
        error=float(np.max(abs(np.stack(traces)-reference)))
        assert error < .0002,(mode,error)
        assert not outputs[0].any() and outputs[1:].any()
        if target is None:
            target=outputs
        np.testing.assert_allclose(target,outputs,atol=.0001,rtol=0)
        validation[mode]=error
    timings={s:[] for s in stages}
    for rep in range(1 if smoke else 3):
        for mode in stages[rep:]+stages[:rep]:
            state.zero_()
            torch.cuda.synchronize()
            start=time.perf_counter()
            for row in raw:
                frame.copy_(row,non_blocking=True)
                execute(mode)[1].cpu()
            elapsed=time.perf_counter()-start
            timings[mode].append(elapsed/len(raw)*1000)
    ratio={s:statistics.median([a/b for a,b in zip(v,timings['serial'])]) for s,v in timings.items()}
    result=dict(session=session,n=len(raw),times_ms=timings,ratio=ratio,max_error=validation,
                device=torch.cuda.get_device_name(),source=ref.sha(Path('source.sha256')),library=ref.sha(Path('decoder.so')))
    print(json.dumps(result),flush=True)
    Path(('smoke' if smoke else person)+'.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    run(sys.argv[1],sys.argv[2:]==['smoke'])
