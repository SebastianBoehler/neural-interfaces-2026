"""Complete nine-cell guarded INDIVIDUAL-MODEL pilot; existing official metric."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

STUDY=Path('/mnt/lustre/work/gehler/gfh527/ni26_factor354_v1_20261117')
RECOVERY=STUDY.with_name('ni26_factor354_recovery_v1_20261117')
GRADIENT_SMOKE=STUDY.with_name('ni26_factor354_gradient_smoke_v1_20261117')
SOURCE_SHA='3d662db2f44d8d83e623c0240333eea941aad1a54b8ec4da6b93f0092d8e4f3c'
GRADIENT_SMOKE_SHA='70bc58334210956b38cd7ac7ee5b372dfa763f7e2a9aaf6076e6d4ea174219df'
RECOVERED={('P02','L'),('P03','O')}
sys.path.insert(0,str(STUDY))
from data354 import original,PEOPLE,ORDERS
from gradient354 import check as gradient_check
ARMS=('O','L','G')


def average(by):
    return float(np.mean([np.mean([by[o] for o in ORDERS if o.startswith(prefix)]) for prefix in ('within','full','runs')]))


def producer_output(person,arm):
    root=RECOVERY if (person,arm) in RECOVERED else STUDY
    return root/'output'/person/arm


def verify_producers():
    for producer in (STUDY,RECOVERY):
        if original.sha(producer/'source.sha256')!=SOURCE_SHA:raise ValueError('producer manifest changed')
        for line in (producer/'source.sha256').read_text().splitlines():
            pin,name=line.split(maxsplit=1)
            if original.sha(producer/name)!=pin:raise ValueError('producer source changed')
        if not json.loads((producer/'smoke.json').read_text())['original_one_epoch_BITWISE']:
            raise ValueError('baseline smoke missing')
    if original.sha(GRADIENT_SMOKE/'result.json')!=GRADIENT_SMOKE_SHA:
        raise ValueError('gradient smoke proof changed')
    if original.sha(Path(__file__).parent/'gradient354.py')!=original.sha(GRADIENT_SMOKE/'gradient354.py'):
        raise ValueError('executed gradient check differs from the source smoke')


def main():
    root=Path.cwd();verify_producers();data,_,_,people=original.load();source_sha=SOURCE_SHA
    banks={};cards={};proof={};changed={}
    for person in PEOPLE:
        held=np.flatnonzero(people==person);fit=np.flatnonzero(people!=person);initial=[];plans=[];scales=[]
        for arm in ARMS:
            folder=producer_output(person,arm);card=json.loads((folder/'record.json').read_text());trace=card['trace']
            if card['source_sha256']!=source_sha or card['query_labels_opened'] or card['person']!=person or card['arm']!=arm:
                raise ValueError('cell scope/label/source guard')
            if card['source_rows_sha256']!=original.digest(fit) or card['query_rows_sha256']!=original.digest(held):raise ValueError('cell row guard')
            if set(card['source_people'])!=set(people[fit]) or person in card['source_people']:raise ValueError('held source guard')
            if trace['epochs']!=40 or trace['steps']!=40*trace['batches_per_epoch'] or len(trace['plans'])!=40 or trace['visited_source_rows']!=len(fit):
                raise ValueError('incomplete source/step exposure')
            if card['guards']['prefix_max_abs']>1e-4 or card['guards']['n_sbn']!=1:raise ValueError('causal normalization guard')
            if original.sha(folder/'model.pt')!=card['model_sha256'] or original.sha(folder/'predictions.npz')!=card['prediction_sha256']:
                raise ValueError('cell output changed')
            expected={'O':0,'L':47024,'G':47075}[arm]
            if trace['adapter_parameters']!=expected or trace['gradient_max']<=0:raise ValueError('capacity/gradient guard')
            if arm in ('L','G'):
                if min(trace['auxiliary'].values())<=0 or set(trace['auxiliary'])!={'order','style','contrast','reconstruction'}:
                    raise ValueError('auxiliary inactive')
                plans.append(trace['plans'])
            if arm=='G' and (trace['mixture_gradient_max']<=1e-12 or trace['gate_variation_max']<=1e-8):raise ValueError('dynamic mixture inactive')
            with np.load(folder/'predictions.npz') as archive:
                if not np.array_equal(archive['index'],held) or set(archive.files)!={'index',*ORDERS}:raise ValueError('order/coverage guard')
                bank={order:archive[order].copy() for order in ORDERS}
            if any(p.shape!=(len(held),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-10,rtol=0) for p in bank.values()):
                raise ValueError('probability guard')
            initial.append(trace['initial_backbone_sha256']);scales.append(trace['scale']);banks[person,arm]=bank;cards[person,arm]=card
            proof[person+'_'+arm]=dict(record_sha256=original.sha(folder/'record.json'),
                prediction_sha256=card['prediction_sha256'],producer=str(folder.parents[2]))
        if len(set(initial))!=1 or len(set(scales))!=1 or plans[0]!=plans[1]:raise ValueError('matched initial state/scale/sampler guard')
        for arm in ('L','G'):
            changed[person+'_'+arm]=max(float(np.abs(banks[person,arm][o]-banks[person,'O'][o]).max()) for o in ORDERS)
            if changed[person+'_'+arm]<=1e-8:raise ValueError('candidate prediction silent no-op')
    if original.sha(original.BASE/'runs/shuffle_perms.npz')!='6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f':
        raise ValueError('canonical delivery orders changed')
    gradients={person:gradient_check(data,people,person,producer_output(person,'O').parent) for person in PEOPLE}
    before=dict(auxiliary_gradient_guards=gradients,status='ALL_NINE_CELLS_EIGHT_ORDERS_GUARDED_BEFORE_GOLD',source_sha256=source_sha,cells=proof,
        probability_change=changed,query_labels_opened=False,unrun_arms=['A','M'],
        legacy_recipe_tag='original inference helper retains baseline recipe string; actual arm/adapter/source hashes above define the executed model')
    (root/'before_labels.json').write_text(json.dumps(before,indent=2))
    metric=original.official();scores={};by=np.empty((len(PEOPLE),len(ARMS),len(ORDERS)))
    for pi,person in enumerate(PEOPLE):
        held=np.flatnonzero(people==person);labels=original.labels(held,len(people))
        for ai,arm in enumerate(ARMS):
            pp=banks[person,arm];ordered={o:metric.score(labels,pp[o].argmax(1),data,people==person)['balanced_accuracy_cell_mean'] for o in ORDERS}
            scores[person+'_'+arm]=dict(primary=average(ordered),uniform8=float(np.mean(list(ordered.values()))),by_order=ordered)
            by[pi,ai]=[ordered[o] for o in ORDERS]
    rng=np.random.default_rng(354);contrasts={}
    for name,a,b in (('G-L',2,1),('G-O',2,0),('L-O',1,0)):
        delta=100*(by[:,a]-by[:,b]);draw=[]
        for _ in range(5000):
            persons=rng.integers(0,len(PEOPLE),len(PEOPLE));parts=[]
            for p in persons:
                parts.append(np.mean([np.mean(delta[p,rng.choice([i for i,o in enumerate(ORDERS) if o.startswith(s)],sum(o.startswith(s) for o in ORDERS),replace=True)]) for s in ('within','full','runs')]))
            draw.append(np.mean(parts))
        contrasts[name]=dict(primary_pp=float(np.mean([average(dict(zip(ORDERS,row))) for row in delta])),
            nested80=list(map(float,np.quantile(draw,[.1,.9]))),by_person={p:100*(scores[p+'_'+ARMS[a]]['primary']-scores[p+'_'+ARMS[b]]['primary']) for p in PEOPLE})
    result=dict(status='INDIVIDUAL_MODEL_EXPOSED_THREE_PERSON_PILOT',scores=scores,contrasts=contrasts,
        before_labels_sha256=original.sha(root/'before_labels.json'),metric_sha256=original.sha(metric.__file__),
        producer_source_sha256=source_sha,cohort=list(PEOPLE),arms=list(ARMS),seed=0,
        scope='source9 per held person; exposedpilot; no FULL-SYSTEM DEV/sealed/rank/familyclosure',unrun_arms=['A','M'])
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
