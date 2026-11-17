"""Four fixed source6 fits. No held labels are read by producers."""
import functools
import json
import os
import time
import types
from pathlib import Path
import numpy as np
import torch
from common344 import source,native,observe,inference,state_sha,sha,digest,teacher_load
from pretext344 import attach
from concurrent344 import continue_fit,snapshot


def fit_arm(arm,teacher,folder,limit=None,save=True,presteps=64):
    d,fit,held,people,y=source();model=native(d);fine={}
    # The old observer rejects incomplete fits, so short source smoke uses the actual loop directly.
    if limit is None:fine=observe(model,'A',fit)
    record={};continuation=functools.partial(continue_fit,arm=arm,teacher=teacher,fit=fit,held=held,people=people,folder=folder,record=record,limit=limit,save=save)
    if arm in ('B','C','D'):
        pre=attach(model,'covariance',teacher,fit,continuation,steps=presteps)
    else:
        native_train=model._train;pre={}
        def train(self,*args,**kwargs):
            self._prepare();continuation(self,None,native_train,*args,**kwargs)
        model._train=types.MethodType(train,model)
    # Population before-pretext embedding is created without RNG/batch changes inside _train.
    wrapped_train=model._train
    def prefine(self,*args,**kwargs):
        self._prepare()
        if save:snapshot(self,None,folder,'initial',fit,held)
        return wrapped_train(*args,**kwargs)
    model._train=types.MethodType(prefine,model)
    started=time.monotonic();model.fit(fit)
    report=dict(arm=arm,pretext=pre,fine=fine,concurrent=record,seconds=time.monotonic()-started,body_sha256=state_sha(model.net),source_rows_sha256=digest(fit),held_rows_sha256=digest(held),scale=model.scale,held_labels_opened=False,P02P03_inference=False)
    return model,d,fit,held,y,report


def main():
    torch.set_num_threads(6);root=Path(os.environ['RETENTION_ROOT']);teacher,card=teacher_load();arm=os.environ['ARM']
    smoke=json.loads((Path(os.environ['SMOKE_ROOT'])/'smoke.json').read_text())
    if not smoke['passed'] or not smoke['budget_passed']:raise ValueError('source smoke gate')
    folder=root/'output'/arm;folder.mkdir(parents=True,exist_ok=False)
    model,d,fit,held,y,report=fit_arm(arm,teacher,folder)
    if report['fine']['steps']!=1240 or len(report['fine']['batches'])!=40:raise ValueError('incomplete native40')
    model.set_k(20.);predictions,checks=inference(model,d,fit,held,y)
    np.savez(folder/'predictions.npz',index=held,**predictions)
    report.update(artifacts={p.name:sha(p) for p in folder.iterdir() if p.is_file()},inference=checks,source_sha256=sha(root/'source.sha256'),teacher_sha256=card['teacher_sha256'],prediction_sha256=sha(folder/'predictions.npz'))
    (folder/'record.json').write_text(json.dumps(report));print(arm,report['seconds'],flush=True)


if __name__=='__main__':main()
