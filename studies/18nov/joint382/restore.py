"""Restore training and inference from the genuine original A6 and strict §381 proofs."""
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch

PRIOR=Path('/mnt/lustre/work/gehler/gfh527/ni26_onebn381_actor_v1_20261118')
spec=importlib.util.spec_from_file_location('prior381',PRIOR/'evaluate.py')
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
prior.joint.check_manifest(PRIOR,'1c6cd0e0a399cc60979801982437d5614b1e7635ec7f8899a80106401bbd5aaf')
joint=prior.joint


def init(net,arm):
    assert arm in ('F','J','K') and len(net.spatial_conv)==3
    model=copy.deepcopy(net).eval()
    if arm in ('F','J'):
        layer=prior.LearnedBN(model.spatial_conv[1]).eval()
        layer.load_state_dict(torch.load(prior.TRAIN/'D.pt',map_location='cpu',weights_only=True))
        model.spatial_conv[1]=layer
    else:
        model.spatial_conv[1].k=20.
    return model


def source_arrays(d,rows,runtime):
    y=joint.labels(rows,len(d['X']))
    streams=d['stream'][rows]
    weights=np.zeros(len(rows),np.float32)
    for s in np.unique(streams):
        session=streams==s
        unique=np.unique(d['interface'][rows][session])
        assert len(unique)==2
        for iface in unique:
            for task in range(3):
                subset=session&(d['interface'][rows]==iface)&(y==task)
                assert subset.any()
                weights[subset]=session.sum()/(6*subset.sum())
    return dict(x=runtime._aligned(rows,True),y=y.astype(np.int64),weights=weights,streams=streams)
