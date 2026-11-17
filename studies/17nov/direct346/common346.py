"""Immutable strong raw reference and actual causal penultimate features."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch

PARENT=Path('/mnt/lustre/work/gehler/gfh527/ni26_pretrain337_pilot_v3_20261117')
sys.path.insert(0,str(PARENT))
from common import source,native,capture,sha,digest,state_sha,labels,official
from recipe336 import BASE

if sha(PARENT/'source.sha256')!='b9970a7e3ad25489f5a2b20ef570a7a53bf1eac2fa6050e9e2278e36765137a1':raise ValueError('337 dependency manifest changed')
for line in (PARENT/'source.sha256').read_text().splitlines():
    pin,name=line.split(maxsplit=1)
    if sha(PARENT/name)!=pin:raise ValueError('337 dependency changed:'+name)

CONTROL=Path('/mnt/lustre/work/gehler/gfh527/ni26_combinations338_v2_20261117/output/source6/A')
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def restored(d,fit,held):
    card=json.loads((CONTROL/'record.json').read_text())
    if card['checkpoint_sha256']!='eadd5c23c9f7cac239bf78a99993d8d3e47392d24d166e0ede6fce5b0e6ac9f8' or sha(CONTROL/'model.pt')!=card['checkpoint_sha256'] or sha(CONTROL/'predictions.npz')!=card['prediction_sha256']:raise ValueError('strongA checkpoint/prediction changed')
    if card['query_labels_opened'] or card['source_rows_sha256']!=digest(fit) or card['query_rows_sha256']!=digest(held):raise ValueError('strongA source scope changed')
    model=native(d);capture(model,fit)
    saved=torch.load(CONTROL/'model.pt',map_location='cpu',weights_only=False)
    model.net.load_state_dict(saved['state'],strict=True)
    if model.scale!=saved['scale'] or not np.array_equal(model.G,saved['G']):raise ValueError('strongA preprocessing changed')
    model.set_k(20.);model.net.eval()
    if model.align or model.dev!='cpu' or len(model.sbns())!=1:raise ValueError('strongA route changed')
    return model,card


def permutations(n):
    with np.load(BASE/'runs/shuffle_perms.npz') as z:
        result={o:z[f'{o[:-1]}|{o[-1]}|all_loso'].copy() for o in ORDERS}
    if any(not np.array_equal(np.sort(v),np.arange(n)) for v in result.values()):raise ValueError('permutation changed')
    return result


def extract(model,rows):
    features=[]
    def hook(module,args):features.append(args[0].detach().numpy().copy())
    before=state_sha(model.net);handle=model.net.final_layer.register_forward_pre_hook(hook)
    try:probability=model.predict_proba_stream(rows)
    finally:handle.remove()
    if len(features)!=1 or features[0].shape!=(len(rows),1152) or before!=state_sha(model.net):raise ValueError('nativefeature path/state differs')
    with torch.no_grad():direct=torch.softmax(model.net.final_layer(torch.from_numpy(features[0])).double(),1).numpy()
    if np.max(np.abs(direct-probability))>1e-12:raise ValueError('actual nativehead readout identity failed')
    if not np.isfinite(features[0]).all():raise ValueError('nonfinite nativefeature')
    return features[0],probability


def bank(model,d,rows,perm):
    rank=np.empty(len(perm),int);rank[perm]=np.arange(len(perm))
    x=np.empty((len(rows),1152),np.float32);p=np.empty((len(rows),3));errors=[]
    streams=list(dict.fromkeys(d['stream'][rows]))
    for stream in streams:
        local=np.flatnonzero(d['stream'][rows]==stream);local=local[np.argsort(rank[rows[local]],kind='stable')]
        rr=rows[local];xx,pp=extract(model,rr);short=min(7,len(rr));xp,prefix=extract(model,rr[:short])
        error=float(np.max(np.abs(pp[:short]-prefix)));errors.append(error)
        if error>1e-4 or not np.allclose(xx[:short],xp,atol=1e-4,rtol=1e-4):raise ValueError('causal prefix guard failed')
        twin=copy.deepcopy(model);tx,tp=extract(twin,rr)
        if not np.array_equal(xx,tx) or not np.array_equal(pp,tp):raise ValueError('deepcopy/reset guard failed')
        if len(streams)>1:
            other=rows[d['stream'][rows]!=stream][:7];extract(model,other)
        rx,rp=extract(model,rr)
        if not np.array_equal(xx,rx) or not np.array_equal(pp,rp):raise ValueError('interleaved session/reset guard failed')
        x[local]=xx;p[local]=pp
    return x,p,dict(prefix_probability_max=max(errors),state_sha256=state_sha(model.net),rows_sha256=digest(rows),order_sha256=digest(perm))
