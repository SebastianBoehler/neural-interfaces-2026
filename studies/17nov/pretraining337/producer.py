"""One small pretext and exact native40 finefit; all banks/guards before P09 labels."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from common import source,native,observe,inference,state_sha,sha,digest,embedding,ARMS
from pretext import attach


def run(arm,teacher):
    d,fit,held,people,y=source();model=native(d);fine=observe(model,'A',fit);pre=attach(model,arm,teacher,fit)
    model.fit(fit);model.set_k(20.)
    if pre['steps']!=64 or fine['steps']!=1240 or len(fine['batches'])!=40:raise ValueError('incomplete pretext/native40')
    prediction,checks=inference(model,d,fit,held,y);features=embedding(model,np.r_[fit,held])
    return d,fit,held,people,y,model,pre,fine,prediction,checks,features


def main():
    root=Path(os.environ['PRETRAIN_ROOT']);torch.set_num_threads(6);start=time.monotonic();arm=ARMS[int(sys.argv[1])]
    smoke=json.loads((Path(os.environ['SMOKE_ROOT'])/'smoke.json').read_text())
    if not smoke['passed'] or not smoke['budget_passed']:raise ValueError('smoke/cost gate required')
    path=Path(os.environ['TEACHER_ROOT']);record=json.loads((path/'teacher.json').read_text())
    if sha(path/'teacher.npz')!=record['teacher_sha256']:raise ValueError('teacher changed')
    with np.load(path/'teacher.npz',allow_pickle=False) as z:teacher={k:z[k] for k in z.files}
    d,fit,held,people,y,model,pre,fine,p,checks,features=run(arm,teacher)
    folder=root/'output'/arm;folder.mkdir(parents=True,exist_ok=False)
    checkpoint=folder/'model.pt';torch.save(dict(state=model.net.state_dict(),G=model.G,scale=model.scale),checkpoint)
    with (folder/'predictions.npz').open('xb') as out:np.savez(out,index=held,**p)
    with (folder/'embeddings.npz').open('xb') as out:np.savez(out,rows=np.r_[fit,held],features=features)
    model.set_k(None)
    with torch.no_grad():probs=torch.softmax(model.net.final_layer(torch.from_numpy(features[:len(fit)])),dim=1).numpy()
    direct=model.predict_proba_stream(fit[:16])
    if not np.allclose(probs[:16],direct,atol=1e-5,rtol=1e-5):raise ValueError('source head-readout identity')
    quality=dict(CE=float(-np.log(np.clip(probs[np.arange(len(y)),y],1e-12,1)).mean()),class_recalls=[float(np.mean(probs[y==c].argmax(1)==c)) for c in range(3)])
    report=dict(arm=arm,source_sha256=sha(root/'source.sha256'),teacher_sha256=record['teacher_sha256'],checkpoint_sha256=sha(checkpoint),
        prediction_sha256=sha(folder/'predictions.npz'),embedding_sha256=sha(folder/'embeddings.npz'),source_rows_sha256=digest(fit),held_rows_sha256=digest(held),
        pretext=pre,fine=fine,source_quality=quality,inference=checks,device=model.dev,threads=torch.get_num_threads(),scale=model.scale,
        held_labels_opened=False,P02P03_inference=False,seconds=time.monotonic()-start,node=os.uname().nodename)
    (folder/'record.json').write_text(json.dumps(report));print(arm,report['seconds'],flush=True)


if __name__=='__main__':main()
