"""All four causal banks and snapshot probes guarded before P09 labels."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from common344 import source,teacher_load,sha,digest,labels,official,PARENT
from retention import prepare,metrics
from diagnostics import probability_quality,scheme

ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
CONTROL=Path('/mnt/lustre/work/gehler/gfh527/ni26_combinations338_v2_20261117/output/source6/A')

root=Path(os.environ['RETENTION_ROOT']);torch.set_num_threads(6);started=time.monotonic();d,fit,held,people,yfit=source();teacher,card=teacher_load();records={};banks={};features={}
for arm in 'ABCD':
    folder=root/'output'/arm;r=json.loads((folder/'record.json').read_text());records[arm]=r
    if r['held_labels_opened'] or r['P02P03_inference'] or r['source_sha256']!=sha(root/'source.sha256') or r['teacher_sha256']!=card['teacher_sha256']:raise ValueError('producer source/label boundary')
    if r['source_rows_sha256']!=digest(fit) or r['held_rows_sha256']!=digest(held):raise ValueError('row boundary')
    for name,value in r['artifacts'].items():
        if sha(folder/name)!=value:raise ValueError('artifact changed:'+name)
    if set(r['artifacts'])!={f'{s}.{ext}' for s in ('initial','fine0','fine10','fine40') for ext in ('pt','npz')}|{'predictions.npz'}:raise ValueError('missing actual intermediate snapshot')
    if r['fine']['steps']!=1240 or len(r['fine']['batches'])!=40 or not r['fine']['head_is_parametrized'] or r['fine']['criterion_weights'] is not None:raise ValueError('native recipe/head guard')
    with np.load(folder/'predictions.npz') as z:
        if set(z.files)!={'index'}|set(ORDERS) or not np.array_equal(z['index'],held):raise ValueError('causal bank row/order guard')
        banks[arm]={o:z[o].copy() for o in ORDERS}
    for snapshot in ('initial','fine0','fine10','fine40'):
        with np.load(folder/f'{snapshot}.npz') as z:
            if not np.array_equal(z['rows'],np.r_[fit,held]):raise ValueError('snapshot rows')
            features[arm,snapshot]=z['features'].copy()
for arm in 'BCD':
    for key in ('scale','steps','batches','torch_rng_before','torch_rng_after','augment_rng_before','augment_rng_after'):
        if records[arm]['fine'][key]!=records['A']['fine'][key]:raise ValueError('native trace mismatch:'+key)
    if records[arm]['pretext']['steps']!=64 or not records[arm]['pretext']['fine_rng_restored']:raise ValueError('pretext incomplete')
for snapshot in ('initial','fine0'):
    if any(not np.array_equal(features['B',snapshot],features[a,snapshot]) for a in 'CD'):raise ValueError('C/D pre-fine initial mismatch')
if records['C']['concurrent']['alpha']!=records['D']['concurrent']['alpha'] or len(records['D']['concurrent']['derangement_hashes'])!=40:raise ValueError('C/D lambda/control activity')
# Strong original A and exact337C reconstruction B must retain original final model identity.
oldA=json.loads((CONTROL/'record.json').read_text());oldB=json.loads((PARENT/'output/covariance/record.json').read_text())
for arm,prior in (('A',oldA['trace']),('B',oldB['fine'])):
    if records[arm]['body_sha256']!=prior['final_state_sha256']:raise ValueError('historical exact finalbody mismatch:'+arm)
    with np.load((CONTROL if arm=='A' else PARENT/'output/covariance')/'predictions.npz') as z:
        if any(not np.array_equal(banks[arm][o],z[o]) for o in ORDERS):raise ValueError('historical causal bank mismatch:'+arm)
for values in banks.values():
    if any(p.shape!=(len(held),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6) for p in values.values()):raise ValueError('posterior validity')
target=teacher['covariance'][np.searchsorted(teacher['rows'],np.r_[fit,held])];maps={};checks={}
for key,value in features.items():maps[key],checks[key]=prepare(value,target,len(fit))
(root/'before_labels.json').write_text(json.dumps(dict(all4banks_all16maps_guards_pass=True,held_labels_opened=False,source_sha256=sha(root/'source.sha256'))))
reconstruction={f'{a}/{s}':metrics(maps[a,s],target,len(fit),checks[a,s]['source_target_mean']) for a,s in maps}
y=labels(held,len(people));metric=official();scores={}
for arm,values in banks.items():
    by={o:metric.score(y,p.argmax(1),d,people=='P09')['balanced_accuracy_cell_mean'] for o,p in values.items()};quality={o:probability_quality(d,held,y,p) for o,p in values.items()}
    scores[arm]=dict(accuracy=scheme(by),by_order=by,NLL=scheme({o:v['NLL'] for o,v in quality.items()}),Brier=scheme({o:v['Brier'] for o,v in quality.items()}))
report=dict(scores=scores,reconstruction=reconstruction,records=records,source_sha256=sha(root/'source.sha256'),before_labels_sha256=sha(root/'before_labels.json'),seconds=time.monotonic()-started,scope='INDIVIDUAL-MODEL exposedP09 oneperson/oneseed; noFULL-SYSTEM claim',P02P03_inference=False)
(root/'result.json').write_text(json.dumps(report));print(json.dumps(scores),flush=True)
