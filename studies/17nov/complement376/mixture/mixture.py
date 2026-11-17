"""Compare probability and log-probability pooling during source-only member training."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from torch.nn import functional as F

BASE = Path('/mnt/lustre/work/gehler/gfh527')
PARENT = BASE/'ni26_complement376_v2_20261117'
FIDELITY = BASE/'ni26_complement376b_v1_20261117'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(PARENT))
from produce import digest, representations, require, src
from nested import balanced

BANK = 'd110fc231ca91b84e90cdaa680e3710cece7e7439f47e1d517f1fd8dcd978ba8'
PINS = {PARENT:'31482d1ac91efbdb54fdaff81fd5c8f62c44ad7f575ab8896d88e77e1beab843',
        FIDELITY:'62a58e927b900120296cab98e106bfabfabb71809759b80f64f14152c3db4bee'}


def loss_terms(logits, proxy, target, arm):
    if arm == 'A':
        q = logits.softmax(1)
        return -(.75*proxy + .25*q)[torch.arange(len(target)),target].log()
    if arm == 'G':
        return F.cross_entropy(.75*proxy.log()+.25*logits,target,reduction='none')
    raise ValueError('unknown mixture objective')


def action_guard(logits, proxy, y):
    z = logits[:16].detach().double().requires_grad_(True)
    p, target = proxy[:16].double(), y[:16]
    row = torch.arange(len(z))
    def grad(posterior, arm):
        return torch.autograd.grad(loss_terms(z,posterior,target,arm).sum(),z)[0]
    q = z.detach().softmax(1)[row,target]
    w = .25*q/(.75*p[row,target]+.25*q)
    equivalent = torch.autograd.grad((F.cross_entropy(z,target,reduction='none')*w).sum(),z)[0]
    changed = p.clone()
    wrong1, wrong2 = (target+1)%3, (target+2)%3
    changed[row,wrong1],changed[row,wrong2] = p[row,wrong2],p[row,wrong1]
    a, g = grad(p,'A'), grad(p,'G')
    err = (a-equivalent).abs().max().item()
    direction = (g-grad(changed,'G')).abs().max().item()
    require(err<1e-12 and torch.equal(a,grad(changed,'A')) and direction>1e-6,'analytic action guard')
    return dict(arithmetic_gradient_error=err, wrong_class_swap_geometric_gradient_delta=direction)


def train(z, logits, proxy, y, weight, arm):
    head = torch.nn.Linear(z.shape[1],3)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        require(torch.equal(logits+head(z).tanh(),logits),'zero head differs')
    optimizer = torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.001)
    for _ in range(24):
        value = (weight*loss_terms(logits+head(z).tanh(),proxy,y,arm)).mean()
        optimizer.zero_grad()
        value.backward()
        require(torch.isfinite(value) and torch.isfinite(head.weight.grad).all()
                and head.weight.grad.abs().max()>0,'mixture update inactive')
        optimizer.step()
    return head.eval()


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for path,expected in PINS.items():
        require(src.sha(path/'source.sha256')==expected,'source version changed')
        for line in (path/'source.sha256').read_text().splitlines():
            h,n = line.split(maxsplit=1)
            require(src.sha(path/n)==h,'source changed')
    require(src.sha(FIDELITY/'bank.npz') == BANK,'nested bank changed')
    held = f'P{1+int(os.environ["SLURM_ARRAY_TASK_ID"]):02d}'
    prev = PARENT/held
    proof = json.loads((prev/'proof.json').read_text())
    require(proof['archive_sha256']==src.sha(prev/'fold.npz') and proof['source_sha256']==PINS[PARENT]
            and not proof['labels_opened'],'§376 fold changed')
    path = ROOT/held
    path.mkdir()
    values, trace = {}, []
    for seed in (0,1):
        d, fit, test, _, _, nets, original, record = src.load(held,seed,'cpu')
        require(digest(fit)==proof['train_rows'] and digest(test)==proof['test_rows'],'changed fold rows')
        if not seed:
            y = src.source_labels.read_training_labels(src.data323.DATA,fit,len(d['X']))
            group = np.array(['|'.join((s,i,str(c))) for s,i,c in zip(d['stream'][fit],d['interface'][fit],y)])
            weight = balanced(group)
            require(digest(weight)==proof['teacher']['weights']['O'],'ordinary weight differs')
            with np.load(FIDELITY/'bank.npz',allow_pickle=False) as f:
                px = f[held+'_inner']
            require(digest(px)==proof['teacher']['source_only_probabilities_sha'],'nested teacher differs')
            proxy, target, w = torch.from_numpy(px).float(),torch.from_numpy(y).long(),torch.from_numpy(weight).float()
        model,scale = nets['P']
        require(record['checkpoints']['P']==proof['traces'][seed]['checkpoint_sha'],'changed body checkpoint')
        z, logits = representations(model,d['X'][fit],scale)
        mu, sd = z.mean(0), z.std(0,unbiased=False).clamp_min(.01)
        z = (z-mu)/sd
        # Different CPU identifications can lead to sub-ulp feature differences.
        guard = action_guard(logits,proxy,target)
        heads = {arm:train(z,logits,proxy,target,w,arm) for arm in ('A','G')}
        q, raw = representations(model,d['X'][test],scale)
        q = (q-mu)/sd
        with torch.no_grad():
            delta = {a:h(q).tanh().double().numpy() for a,h in heads.items()}
        require(np.max(abs(delta['A']-delta['G']))>1e-5,'two objectives inactive')
        for order in src.ORDERS:
            frozen = original['PF|'+order]
            require(np.array_equal(frozen+0*delta['A'],frozen) and
                    np.array_equal(frozen.argmax(1),raw.argmax(1).numpy()) and
                    np.max(abs(frozen-raw.double().softmax(1).numpy()))<1e-4,'frozen endpoint drift')
            for arm in heads:
                values.setdefault(arm+'|'+order,[]).append(softmax(np.log(frozen)+delta[arm],axis=1))
            values.setdefault('RF|'+order,[]).append(frozen)
        trace.append(dict(seed=seed,source_only_smoke=guard,source_feature_bytes_match=
                          digest(z.numpy())==proof['traces'][seed]['source_feature_sha']))
    output = {k:np.mean(v,axis=0) for k,v in values.items()}
    with np.load(prev/'fold.npz') as old:
        for order in src.ORDERS:
            require(np.array_equal(output['RF|'+order],old['RF|'+order]),'RF bitwise emission mismatch')
            for a in ('O','E','C','S'):
                output[a+'|'+order] = old[a+'|'+order]
    archive = path/'fold.npz'
    np.savez_compressed(archive,index=test,**output)
    with (path/'proof.json').open('x') as f:
        json.dump(dict(held=held,train_rows=digest(fit),test_rows=digest(test),seeds=[0,1],labels_opened=False,
                       train_people=proof['train_people'],teacher=proof['teacher'],traces=trace,archive_sha256=src.sha(archive),
                       parent_sha=src.sha(prev/'proof.json'),parent_archive_sha=src.sha(prev/'fold.npz'),
                       nested_bank_sha=BANK,source_sha256=src.sha(ROOT/'source.sha256'),
                       elapsed_s=time.monotonic()-started),f,indent=2)


if __name__ == '__main__':
    main()
