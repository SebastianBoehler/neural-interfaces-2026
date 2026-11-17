"""Official-cell/class-balanced probability quality; no model or row selection."""
import numpy as np


def scheme(values):return float(np.mean([np.mean([v for o,v in values.items() if o.startswith(s)]) for s in ('within','full','runs')]))


def probability_quality(d,rows,y,p):
    cells=np.char.add(np.char.add(d['stream'][rows],'|'),d['interface'][rows]);nll=[];brier=[];recall=[]
    for name in np.unique(cells):
        choose=cells==name
        if set(y[choose])!={0,1,2}:raise ValueError('heldcell missingclass')
        nll.append(np.mean([-np.log(np.clip(p[choose&(y==k),k],1e-12,1)).mean() for k in range(3)]))
        brier.append(np.mean([np.square(p[choose&(y==k)]-np.eye(3)[k]).sum(-1).mean() for k in range(3)]))
        recall.append([float(np.mean(p[choose&(y==k)].argmax(1)==k)) for k in range(3)])
    return dict(NLL=float(np.mean(nll)),Brier=float(np.mean(brier)),class_recalls=np.mean(recall,axis=0).tolist())
