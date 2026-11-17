"""Source-only independent weighted-auxiliary gradient-path check; no updates."""
import torch
import numpy as np
from architecture import TokenLayer
from data354 import original
from train354 import WEIGHTS


def check(data,people,person,folder):
    checkpoint=torch.load(folder/'O/model.pt',map_location='cpu',weights_only=False)
    selected=[];ids=[]
    for index,name in enumerate(sorted(set(people)-{person})[:2]):
        rows=np.flatnonzero(people==name);stream=data['stream'][rows[0]]
        rows=rows[data['stream'][rows]==stream][:32]
        if len(rows)!=32:raise ValueError('source gradient batch too small')
        selected.extend(rows);ids.extend([index]*32)
    selected=np.asarray(selected)
    if np.any(people[selected]==person):raise ValueError('held person in auxiliary gradient probe')
    x=torch.from_numpy((data['X'][selected].astype(np.float64)/checkpoint['scale']).astype(np.float32))
    result={}
    for arm in ('L','G'):
        torch.manual_seed(0);net=original.deep.build('fbcnet',41,480,False)
        if original.session_bn.swap_bn(net,False,20.)!=1:raise ValueError('source probe backbone changed')
        net.load_state_dict(checkpoint['state'],strict=True);net.eval()
        for layer in net.modules():
            if isinstance(layer,original.session_bn.SessionBN):layer.k=None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(354001);net.temporal_layer=TokenLayer(net.temporal_layer,arm)
        adapter=net.temporal_layer.adapter;adapter.train()
        net(x);terms,_=adapter.losses(torch.tensor(ids));parameters=list(adapter.parameters())
        record={}
        for name,value in terms.items():
            gradients=torch.autograd.grad(WEIGHTS[name]*value,parameters,retain_graph=True,allow_unused=True)
            active=[g for g in gradients if g is not None]
            maximum=max(float(g.abs().max()) for g in active)
            if not value.requires_grad or not all(torch.isfinite(g).all() for g in active):raise ValueError('disconnected/nonfinite auxiliary graph')
            if name!='order' or float(value.detach())>0:
                if maximum<=0:raise ValueError('positive auxiliary objective has zero gradient')
            record[name]=dict(loss=float(value.detach()),weighted_gradient_max=maximum,
                              hinge_satisfied=(name=='order' and float(value.detach())==0))
        adapter.cache=None;result[arm]=record
    return dict(held_person=person,source_people=sorted(set(people[selected])),probe=result,
                optimizer_steps=0,query_labels_opened=False,source_rows_sha256=original.digest(selected))
