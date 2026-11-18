"""Make fresh matched EA members in certified seen-session folds."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from new_fit import configure_runtime,replay
from probe import ORDERS,digest,fingerprint,read_inputs,tensor_identity,write_json
from returning_inputs import FAMILIES,LAYOUT,PEOPLE,rows

HERE=Path(__file__).resolve().parent
REV='c625ecc736a881836a25f85a741ea9147bf698bf'
PARENT=Path('/mnt/lustre/work/gehler/gfh527/ni26_friday_ea_matrix_20261118')


def fit_member(actor,source,train_fir,deep,torch):
    builder=deep.build
    initial={}
    def build(*args,**kwargs):
        net=builder(*args,**kwargs)
        if train_fir:
            initial.update({k:v.detach().cpu().clone()
                            for k,v in net.spectral_filtering.named_parameters()})
        return net
    deep.build=build
    try:
        actor.fit(source)
    finally:
        deep.build=builder
    if not train_fir:
        assert not initial
        return {}
    bank=actor.net.spectral_filtering
    assert not bank.method_iir and not bank.forward_filter and bank.n_bands==9
    learned=dict(bank.named_parameters())
    numerators={f'fir_list.{i}' for i in range(bank.n_bands)}
    denominators={f'fir_a_list.{i}' for i in range(bank.n_bands)}
    assert initial.keys()==learned.keys()==numerators|denominators
    spectral={}
    for key,val in learned.items():
        first=initial[key]
        after=val.detach().cpu()
        assert val.requires_grad and torch.isfinite(after).all()
        update=float((first-after).abs().max())
        if key in numerators:
            assert val.grad is not None and torch.isfinite(val.grad).all(),key
            gradient=float(val.grad.abs().max())
            assert update>0 and gradient>0,key
        else:
            assert val.grad is None and update==0,key
            gradient=None
        spectral[key]=dict(shape=list(after.shape),dtype=str(after.dtype),
                           max_update=update,max_grad=gradient)
    return spectral


def fit(root,name,held,seed,smoke):
    import braindecode
    import deep
    import features
    import protocol
    import run_cv
    import torch
    from source_labels import read_training_labels

    slot,nseeds,arch,cls=FAMILIES[name]
    assert seed in range(nseeds) and smoke==((held,seed)==('P10',0))
    for mod in (deep,features,protocol,run_cv):
        assert Path(mod.__file__).resolve().parent==PARENT/REV/'experiments',mod.__name__
    assert braindecode.__version__=='1.8.1'
    configure_runtime(torch)
    data,_=read_inputs(root)
    splits,index,deliveries=rows(root,data)
    source,test=splits[held]
    audit=json.loads((HERE/'inputs_audit.json').read_text())
    assert audit['held_labels_opened'] is False and fingerprint(index)==audit['global_rows_sha']
    assert fingerprint(source)==audit['folds'][held]['source_sha']
    assert fingerprint(test)==audit['folds'][held]['held_sha']
    assert {k:fingerprint(v) for k,v in deliveries.items()}==audit['orders']
    assert audit['families'][name]['exact_archived_slot'] is True
    orders=dict(recording=test,**{k:v[np.isin(v,test)] for k,v in deliveries.items()})

    data['y']=np.full(len(data['subject']),-1,dtype=np.int64)
    data['y'][source]=read_training_labels(root/'data/proteus_windows.npz',source,len(data['y']))
    assert np.isin(data['y'][source],(0,1,2)).all() and (data['y'][test]==-1).all()
    data['w']=np.ones(len(data['y']))
    run_cv.SEED[0]=seed
    actor=run_cv.CONFIGS[name](data)
    train_fir=name=='fbcnet_ea_tf'
    assert (actor.arch,actor.align,actor.k,actor.epochs,actor.batch,actor.lr,actor.wd,
            actor.seed,actor.dev,actor.adabn,actor.aug,actor.train_fir,actor.ft,
            actor.ls,actor.lld,actor.lp,actor.crops,actor.net_kw)==(
            arch,True,5.,40,64,1e-3,1e-3,seed,'cuda',None,False,train_fir,0,0.,1.,0,(),{})
    prior=features.covariances(data['X'][source]).mean(0)

    spectral=fit_member(actor,source,train_fir,deep,torch)
    assert type(actor.net) is getattr(braindecode.models,cls) and not actor.net.training
    assert all(torch.isfinite(t).all() for t in actor.net.state_dict().values())
    batches={k:int(b) for k,b in actor.net.named_buffers() if k.endswith('num_batches_tracked')}
    assert batches and all(v==40*math.ceil(len(source)/64) for v in batches.values())
    assert np.array_equal(actor.G,prior) and (data['y'][test]==-1).all()
    frozen=tensor_identity(actor.net)

    values,checks={},{}
    for key,order in orders.items():
        values[key],checks[key]=replay(actor,features,test,source,order,torch)
    delta={key:float(np.abs(values[key]-values['recording']).max()) for key in ORDERS}
    assert tensor_identity(actor.net)==frozen and max(delta.values())>1e-4
    assert tuple(checks['full0'][f'Dreyer2026Proteus/{held}|{d}']['first_row']
                 for d in ('2','3'))==PEOPLE[held][3]
    if smoke:
        repeat=fit_member(actor,source,train_fir,deep,torch)
        assert tensor_identity(actor.net)==frozen and repeat==spectral and np.array_equal(actor.G,prior)
        for key,order in orders.items():
            again,_=replay(actor,features,test,source,order,torch)
            assert np.array_equal(again,values[key]),key

    tag=f'{name}_{held}_seed{seed}'
    checkpoint=HERE/f'{tag}_fresh.pt'
    output=HERE/f'{tag}_orders.npz'
    with checkpoint.open('xb') as handle:
        torch.save(dict(model={k:v.cpu() for k,v in actor.net.state_dict().items()},
                        prior=torch.from_numpy(prior),fold=held,seed=seed,source=REV,state=frozen),handle)
    with output.open('xb') as handle:
        np.savez_compressed(handle,index=test,orders=np.array(LAYOUT),
                            delivery=np.stack(list(orders.values())),proba=np.stack(list(values.values())))
    row=dict(family=name,slot=slot,fold=held,seed=seed,source=REV,held_labels_opened=False,
        historical_reconstruction=False,shared_by_comparators=True,deterministic=True,
        duplicate_fit_bit_exact=smoke,fit_rows_sha=fingerprint(source),held_rows_sha=fingerprint(test),
        source_label_sha=fingerprint(data['y'][source]),source_prior_sha=fingerprint(prior),
        state=frozen,spectral=spectral,batches=batches,checks=checks,delta_from_recording=delta,
        source_manifest=digest(HERE/'source.sha256'),audit_sha=digest(HERE/'inputs_audit.json'),
        checkpoint_sha=digest(checkpoint),predictions_sha=digest(output),
        gpu=torch.cuda.get_device_name(),node=os.environ['SLURMD_NODENAME'])
    write_json(HERE/f'{tag}_receipt.json',row)
    print('PASS',tag,frozen,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('family',choices=FAMILIES)
    parser.add_argument('held',choices=PEOPLE)
    parser.add_argument('seed',type=int)
    parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    fit(a.root,a.family,a.held,a.seed,a.smoke)
