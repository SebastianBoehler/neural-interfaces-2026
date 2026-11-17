"""Pinned native336 recipe with exactly composed H/W/B interventions."""
import inspect
import sys
from pathlib import Path
import numpy as np
ROOT336=Path('/mnt/lustre/work/gehler/gfh527/ni26_reference_bridge336_v1_20261117')
sys.path.insert(0,str(ROOT336))
import recipe336 as old
if old.sha(ROOT336/'recipe336.py')!='468d632f51e52ffd16f1dff1ae61a2a4b22483a22a7618f25a6d36952d6df3be':raise ValueError('336APIchanged')
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
CELLS=[('source7',a) for a in ('HW','HB','HWB')]+[('source6',a) for a in ('A','H','W','B','HW','HB','HWB')]


def source(scope):
    d,fit,query,people=old.load()
    if scope=='source6':fit=fit[people[fit]!='P09'];query=np.flatnonzero(people=='P09')
    elif scope!='source7':raise ValueError('unknownsourceinstrument')
    y=old.labels(fit,len(d['X']));d['y']=np.full(len(d['X']),-1,np.int64);d['y'][fit]=y
    d.update(w=np.ones(len(d['X'])),nolow=np.zeros(len(d['X']),bool),tag='')
    return d,fit,query,people,y


def observe(model,arm,fit):
    text=inspect.getsource(old.observe)
    replacements={"arm not in ('A','H','U','W','B')":"arm not in ('A','H','W','B','HW','HB','HWB')",
                  "arm=='H'":"'H' in arm","arm=='W'":"'W' in arm","arm=='B'":"'B' in arm"}
    for a,b in replacements.items():
        if a not in text:raise ValueError('compositionanchorchanged:'+a)
        text=text.replace(a,b)
    env=dict(vars(old));exec(compile(text,'338_composed336observer','exec'),env)
    return env['observe'](model,arm,fit)
