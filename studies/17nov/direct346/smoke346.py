"""Real source-only import, direct-feature, causal-state and cost qualification."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from common346 import source,restored,permutations,bank,sha,digest
from features346 import direct,shuffled,fit_head,predict


def main():
    root=Path(os.environ['DIRECT_ROOT']);start=time.monotonic();torch.set_num_threads(2)
    d,fit,held,people,y=source();model,card=restored(d,fit,held)
    rows=np.concatenate([fit[people[fit]==person][:24] for person in sorted(set(people[fit]))])
    positions=np.searchsorted(fit,rows);perms=permutations(len(people));tick=time.monotonic()
    latent,_,guard=bank(model,d,rows,perms['within0']);extra=direct(d['X'][rows]);seconds=time.monotonic()-tick
    one=direct(d['X'][rows[:1]])
    if not np.array_equal(one,extra[:1]):raise ValueError('direct per-window batch independence failed')
    with np.load(__import__('common346').BASE/'data/proteus_windows.npz') as z:run=z['run']
    deranged,order=shuffled(rows,d,run,extra)
    traces={};head_start=time.monotonic()
    for name,x in {'B':latent,'C':np.c_[latent,extra],'D':np.c_[latent,deranged]}.items():
        head=fit_head(x,y[positions]);traces[name]=head[2];predict(head,x)
        if name!='B' and np.max(np.abs(head[1].coef_[:,1152:]))<=1e-10:raise ValueError('extra readout inactive')
    head_seconds=time.monotonic()-head_start
    # Full allocation uses six cores: measured two-core source work is conservative.
    estimate=1.5*(seconds*(len(fit)+8*len(held))/len(rows)+head_seconds*(len(fit)/len(rows))**2+60)
    elapsed=time.monotonic()-start;projected_coreseconds=2*(elapsed+15)+6*(estimate+15)
    report=dict(passed=True,budget_passed=bool(estimate<600 and projected_coreseconds<=3600),
        source_sha256=sha(root/'source.sha256'),checkpoint_sha256=card['checkpoint_sha256'],rows_sha256=digest(rows),
        source_people=sorted(set(people[rows])),held_transformed=False,held_labels_opened=False,
        native_guard=guard,direct_sha256=digest(extra),derangement_sha256=digest(order),traces=traces,
        feature_seconds=seconds,three_heads_seconds=head_seconds,projected_full_seconds=estimate,
        projected_total_CPUcoreseconds=projected_coreseconds,seconds=elapsed)
    (root/'smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['budget_passed']:raise ValueError('measured one-CPUh cost guard failed')


if __name__=='__main__':main()
