"""Paired source fits of the conditional and constant reliability gates."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from gate import Gate,endpoints,features,mix

BUNDLE=Path.home()/'ni26/382/bundle'
sys.path.insert(0,str(BUNDLE))
from fit_bundle import sha
from kernel import schedule,group_digest,sha_array


def main(seed,smoke):
    assert 'H100' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert sha(BUNDLE/'bundle.json')=='06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
    meta=json.loads((BUNDLE/'bundle.json').read_text())
    for name,h in meta['files'].items():
        assert sha(BUNDLE/name)==h
    with np.load(BUNDLE/'source.npz') as z:
        x,y,w=[torch.from_numpy(z[k]).cuda() for k in ('x','y','weights')]
        streams=z['streams']
    persons=np.array([s.split('/')[1].split('|')[0] for s in streams])
    assert sorted(set(persons))==meta['people'] and meta['source_only']
    net=torch.jit.load(str(BUNDLE/'native.ts'),map_location='cuda').eval()
    for param in net.parameters():
        param.requires_grad_(False)
    fixed=group_digest(net.parameters())
    eps=float(list(net.spatial_conv.children())[1].code_with_constants[1].const_mapping['c1'])
    assert eps==1e-5
    probe=schedule(streams,382,1)[0]
    k,p=endpoints(net,x[probe],eps)
    with np.load(BUNDLE/'probes.npz') as z:
        ref=torch.from_numpy(z['native']).cuda()
        torch.testing.assert_close(ref,k,atol=.005,rtol=.0001)
        assert torch.equal(ref.argmax(1),k.argmax(1))
    assert torch.equal(mix(k,p,1.),k) and torch.equal(mix(k,p,0.),p) and (k-p).abs().max()>.001
    gates=[Gate(c).cuda() for c in [True,False]]
    opts=[torch.optim.Adam(g.parameters(),lr=.03) for g in gates]
    rng=np.random.default_rng(seed)
    batches=[]
    for i in range(6 if smoke else 120):
        if i%6==0:
            order=rng.permutation(meta['people'])
        stream=rng.choice(np.unique(streams[persons==order[i%6]]))
        b=rng.choice(np.flatnonzero(streams==stream),48,replace=False)
        batches.append(b)
        k,p=endpoints(net,x[b],eps)
        f=features(k,p)
        for gate,opt in zip(gates,opts):
            opt.zero_grad()
            loss=(F.cross_entropy(mix(k,p,gate(f)),y[b],reduction='none')*w[b]).mean()
            if gate.slopes is not None:
                loss=loss+.0001*gate.slopes.square().sum()
            loss.backward()
            assert all(torch.isfinite(t.grad).all() and t.grad.abs().max()>0 for t in gate.parameters())
            opt.step()
    assert group_digest(net.parameters())==fixed
    assert gates[0](f).std()>.00001
    rec=dict(seed=seed,steps=len(batches),fixed=fixed,source=sha(Path('source.sha256')),
             rows=meta['source_rows_sha256'],episodes=sha_array(np.stack(batches)),
             gates=[{n:v.detach().cpu().tolist() for n,v in g.named_parameters()} for g in gates])
    Path(f'{"smoke" if smoke else "fit"}_{seed}.json').write_text(json.dumps(rec,indent=2))
    print(rec,flush=True)


if __name__=='__main__':
    main(int(sys.argv[1]),sys.argv[2:]==['smoke'])
