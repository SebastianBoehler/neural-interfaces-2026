"""§320 real source smoke then bounded person-crossfitted source pilot, no outer score."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(os.environ['NI26_ROOT']);sys.path.insert(0,str(ROOT/'experiments'))
import protocol
from encoder import fingerprint
from fit import encoder_fit,ranker_fit,source_tensors
from views import aggregate,bank,branches,calibrate,descriptors,mask_sha


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def batch(net,x,mask):
    return torch.cat([branches(net,b,mask) for b in x.split(64)])


def guards(net,x,mask):
    before=fingerprint(net);logits=branches(net,x,mask)
    with torch.no_grad():ordinary=net(x).double()
    if not torch.allclose(logits[:,0],ordinary,atol=2e-5,rtol=2e-5):
        raise ValueError('identity branch changed unchanged head')
    if not torch.allclose(logits.mean(1),logits[:,0],atol=2e-5,rtol=2e-5):
        raise ValueError('inverted partition bank mean-logit identity failed')
    zeros=branches(net,x,torch.ones_like(mask))
    if not torch.allclose(zeros,ordinary[:,None].expand_as(zeros),atol=2e-5,rtol=2e-5):
        raise ValueError('same-path zero mask changed logits')
    z=torch.zeros(logits.shape[:2]);uniform=aggregate(logits,z)
    if not torch.equal(uniform,aggregate(logits,torch.zeros_like(z))):
        raise ValueError('zero-ranker uniform path differs')
    if torch.allclose(logits[:,1:],logits[:,:1].expand_as(logits[:,1:]),atol=1e-8,rtol=1e-8):
        raise ValueError('mask bank inactive')
    prefix=branches(net,x[:3],mask)
    branches(net,x.flip(0),mask)
    if fingerprint(net)!=before or not torch.allclose(prefix,logits[:3],atol=2e-5,rtol=2e-5):
        raise ValueError('state/reset/prefix guard failed')
    if mask_sha(mask)!=mask_sha(bank(net.dim)):
        raise ValueError('fixed deployment mask identity changed')
    return dict(identity=True,mean_logit_identity=True,zero_mask=True,zero_ranker_uniform=True,
                active_masks=True,prefix_reset=True,population_bn_unchanged=True,mask_sha256=mask_sha(mask))


def main():
    if os.environ['NI26_DEVICE']!='cpu':raise ValueError('registered explicit CPU route required')
    torch.set_num_threads(6);started=time.monotonic();ws=Path.cwd();d=protocol.load()
    people=sorted(set(d['subject'].tolist()))
    outer=next(p for p in people if p.endswith('/P01'));validation=next(p for p in people if p.endswith('/P02'))
    sources=[p for p in people if p not in (outer,validation)]
    if len(sources)!=8:raise ValueError('expected eight source people')
    # Source-only real smoke runs before the full scientific pilot.
    fit=[p for p in sources if p!=sources[0]]
    _,x,y,scale=source_tensors(d,fit);_,qx,qy,_=source_tensors(d,[sources[0]],scale)
    net,meta=encoder_fit(x,y,1);mask=bank(net.dim);check=guards(net,qx[:17],mask)
    smoke_rows=np.concatenate([np.flatnonzero(qy.numpy()==c)[:6] for c in (0,1,2)])
    logits=branches(net,qx[smoke_rows],mask)
    model,_,_,diagnostic=ranker_fit(logits,qy[smoke_rows],np.repeat(sources[0],len(smoke_rows)),steps=2)
    if diagnostic['max_gradient']<=0:raise ValueError('source smoke ranker inactive')
    smoke=dict(status='real_source_smoke_passed',guard=check,source_fit=fit,source_query=sources[0],encoder=meta)
    (ws/'smoke.json').write_text(json.dumps(smoke,indent=2));print('source smoke passed',flush=True)
    del net,x,y,qx,qy,model
    predictions=[];labels=[];owners=[];folds=[]
    for query in sources:
        fit=[p for p in sources if p!=query]
        rows,x,y,scale=source_tensors(d,fit);qr,qx,qy,_=source_tensors(d,[query],scale)
        if query in fit or outer in fit or validation in fit:raise ValueError('ranker source crossfit leakage')
        net,meta=encoder_fit(x,y,10)
        if mask_sha(bank(net.dim))!=check['mask_sha256']:raise ValueError('source/deploy mask identity differs')
        guard=guards(net,qx[:17],mask);predictions.append(batch(net,qx,mask));labels.append(qy)
        owners.extend([query]*len(qy))
        folds.append(dict(query=query,fit_people=fit,encoder=meta,guard=guard,scale=scale,
                          fit_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),query_rows_sha256=hashlib.sha256(qr.tobytes()).hexdigest()))
        print('completed source crossfit',query,meta['seconds'],flush=True)
    oof=torch.cat(predictions);oy=torch.cat(labels);owners=np.array(owners)
    ranker,static,temperatures,rank_meta=ranker_fit(oof,oy,owners)
    _,x,y,scale=source_tensors(d,sources);vr,vx,vy,_=source_tensors(d,[validation],scale)
    net,meta=encoder_fit(x,y,10);guard=guards(net,vx[:17],mask)
    logits=batch(net,vx,mask);features=descriptors(logits)
    with torch.no_grad():scores=ranker(features)
    output={'H':logits[:,0].softmax(-1),'U':aggregate(logits,torch.zeros_like(scores)),
            'S':aggregate(logits,static[None].expand_as(scores)),'R':aggregate(logits,scores)}
    for name in tuple(output):output[name+'C']=calibrate(output[name],temperatures[name])
    for name in ('H','U','S','R'):
        if not torch.equal(output[name].argmax(1),output[name+'C'].argmax(1)):
            raise ValueError('postaggregation temperature changed class predictions')
    weights=scores.softmax(-1)
    # Only now are separately held sourceP02 labels used for pilot accuracy.
    results={name:dict(ba=float(torch.stack([(p.argmax(1)[vy==c]==c).float().mean() for c in (0,1,2)]).mean()),
                      ce=float(-p.log().gather(1,vy[:,None]).mean())) for name,p in output.items()}
    report=dict(status='source_pilot_complete',outer_excluded=outer,validation_person=validation,fit_people=sources,
                fixed_epochs=10,source_manifest=sha(ws/'source.sha256'),data_sha256=sha(ROOT/'data/proteus_windows.npz'),
                smoke=smoke,folds=folds,ranker=rank_meta,source_control=meta,guard=guard,
                results=results,temperatures=temperatures,source_weight_std=float(weights.std(0).max()),
                seconds=time.monotonic()-started,validation_rows_sha256=hashlib.sha256(vr.tobytes()).hexdigest(),
                interpretation='source-only repeatedly-usedP02 pilot;not outer/current-chain gain or paper reproduction')
    report['source_control_valid']=results['H']['ba']>=.6860
    report['instance_reliability_active']=report['source_weight_std']>=1e-4
    report['source_harm_stop']=results['R']['ba']-results['U']['ba']<-.03
    report['expansion_policy']='manual scientific review;no automatic outer matrix'
    torch.save(dict(encoder=net.state_dict(),scale=scale,ranker=ranker.state_dict(),static=static,
                    mask=mask,temperatures=temperatures),ws/'source_models.pt')
    report['checkpoint_sha256']=sha(ws/'source_models.pt')
    (ws/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(results),flush=True)


if __name__=='__main__':main()
