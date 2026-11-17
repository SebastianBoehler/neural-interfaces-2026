"""§376 immutable outer-fold producer. Does not read outer labels."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from torch.nn import functional as F

from nested import digest, proxy_weights, require
ROOT = Path(__file__).resolve().parent
PARENT = Path('/mnt/lustre/work/gehler/gfh527/ni26_moments332_v2_20261117')
sys.path.insert(0,str(PARENT))
import source332 as src


@torch.no_grad()
def representations(net, eeg, scale):
    zs, logits = [], []
    for batch in np.array_split(np.arange(len(eeg)), (len(eeg)+31)//32):
        x = torch.from_numpy(eeg[batch]/scale).float()
        z = net.representation(x)
        raw = net.task(torch.cat((z, torch.zeros((len(z),287))),dim=1))
        if not zs:
            require(torch.allclose(raw,net(x),atol=1e-6), 'active native logits differ')
        zs.append(z)
        logits.append(raw)
    return torch.cat(zs), torch.cat(logits)


def train(z, logits, y, weight, steps):
    head = torch.nn.Linear(z.shape[1],3)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        require(torch.equal(logits+head(z).tanh(), logits), 'initial logits differ')
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.001)
    y, w = torch.from_numpy(y).long(), torch.from_numpy(weight).float()
    for step in range(steps):
        loss = (F.cross_entropy(logits+head(z).tanh(),y,reduction='none')*w).mean()
        optimizer.zero_grad()
        loss.backward()
        require(torch.isfinite(loss) and torch.isfinite(head.weight.grad).all()
                and torch.max(head.weight.grad.abs())>0, 'inactive residual fit')
        optimizer.step()
    head.eval()
    return head


def run(held, smoke):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    src.check()
    require(src.sha(PARENT/'source.sha256') == '6c0f967ca5e8a7240b676de483bdc575eab2918c526fee4b832dc107d925abab',
            'parent snapshot differs')
    for line in (PARENT/'source.sha256').read_text().splitlines():
        h,name = line.split(maxsplit=1)
        require(src.sha(PARENT/name) == h, 'changed parent code')
    folder = ROOT/('smoke' if smoke else held)
    folder.mkdir()
    values, traces = {}, []
    for seed in ((0,) if smoke else (0,1)):
        d,fit,test,_,perms,nets,original,record = src.load(held,seed,'cpu')
        people = np.array([str(s).split('/')[-1] for s in d['subject']])
        require(np.all(people[test] == held) and not set(fit)&set(test), 'held fold collision')
        if seed == 0:
            y = src.source_labels.read_training_labels(src.data323.DATA,fit,len(people))
            weights, provenance = proxy_weights(d['X'][fit],y,d['stream'][fit],d['interface'][fit],
                                               people[fit],fit,held)
        net,scale = nets['P']
        z, logits = representations(net,d['X'][fit],scale)
        mu, sd = z.mean(0), z.std(0,unbiased=False).clamp_min(.01)
        z = (z-mu)/sd
        if smoke:
            q, raw = z[:8], logits[:8]
        else:
            q, raw = representations(net,d['X'][test],scale)
            reverse, _ = representations(net,d['X'][test[:8][::-1]],scale)
            require(torch.allclose(reverse.flip(0),q[:8],rtol=2e-5,atol=2e-5), 'order/prefix guard')
            q = (q-mu)/sd
        deltas = {}
        for arm,w in weights.items():
            head = train(z,logits,y,w,1 if smoke else 24)
            with torch.no_grad():
                deltas[arm] = head(q).tanh().double().numpy()
        require(len({digest(v) for v in deltas.values()}) == 4, 'inactive residual contrasts')
        if smoke:
            (folder/'success.json').write_text(json.dumps({'source_only':True,'smoke_pass':True,
                'id':os.environ['SLURM_JOB_ID'],'source_sha256':src.sha(ROOT/'source.sha256')}))
            print('source-only one-step smoke passed',flush=True)
            return
        err = 0.
        for order in src.ORDERS:
            frozen = original['PF|'+order]
            require(np.array_equal(np.sort(perms[order]),np.arange(len(people)))
                    and np.array_equal(frozen + 0*deltas['E'],frozen), 'order/zero identity')
            mismatch = abs(frozen - raw.double().softmax(1).numpy()).max()
            err = max(err, float(mismatch))
            require(mismatch < 1e-4 and np.array_equal(frozen.argmax(1),raw.argmax(1).numpy()),
                    f'fixed native CPU/GPU identity differs {mismatch}')
            values.setdefault('RF|'+order,[]).append(frozen)
            for arm,delta in deltas.items():
                values.setdefault(arm+'|'+order,[]).append(softmax(np.log(frozen)+delta,axis=1))
        traces.append(dict(seed=seed, fixed_max_prob_error=err, checkpoint_sha=record['checkpoints']['P'],
                           frozen_archive=record['original_fold']['output_sha256'],
                           source_feature_sha=digest(z.numpy())))
    result = {k:np.mean(v,axis=0) for k,v in values.items()}
    path = folder/'fold.npz'
    np.savez_compressed(path,index=test,**result)
    (folder/'proof.json').write_text(json.dumps(dict(held=held,seeds=[0,1],train_people=sorted(set(people[fit])),
        labels_opened=False,archive_sha256=src.sha(path),source_sha256=src.sha(ROOT/'source.sha256'),
        teacher=provenance,train_rows=digest(fit),test_rows=digest(test),traces=traces,
        elapsed_s=time.monotonic()-started),indent=2))
    print(f'{held}: 40 complete held outputs; outer labels unopened',flush=True)


if __name__ == '__main__':
    run(sys.argv[1],sys.argv[2:] == ['smoke'])
