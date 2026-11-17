"""Read-only posthoc equal-scheme convention report; preserve immutable uniform-eight primary."""
import json
from pathlib import Path

import numpy as np

from returning_assets import ROOT,sha

PRIMARY=ROOT/'ni26_js321_20261117_returning_pair_cpu_complete1/conditional_results.json'
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
SCHEMES=(np.arange(4),np.arange(4,6),np.arange(6,8))


def main():
    original=sha(PRIMARY);p=json.loads(PRIMARY.read_text())
    if p['status']!='COMPLETE_CONDITIONAL_ENGINEERING_SENSITIVITY_NOT_FULLSYSTEMDEV' or len(p['complete_cells'])!=16 or not p['off_minus_base_exact_zero']:raise ValueError('complete owning primary report required')
    values=np.full((2,8,3),np.nan)
    for row in p['official_metrics']:
        who=('P04','P10').index(row['fold']);order=ORDERS.index(row['order'])
        for j,arm in enumerate(('base','off','js')):values[who,order,j]=row['metrics'][arm]['balanced_accuracy_cell_mean']
    if not np.isfinite(values).all():raise ValueError('partial primary official metrics')
    delta=100*(values[:,:,2]-values[:,:,0]);w=np.array([.2,.8]);uniform=float(w@delta.mean(1))
    if abs(uniform-p['returning_weighted_JS_minus_base_pp'])>1e-10:raise ValueError('immutable primarypoint reconstruction differs')
    scheme_values=np.stack([values[:,s].mean(1) for s in SCHEMES],1);scheme_delta=100*(scheme_values[:,:,2]-scheme_values[:,:,0]);equal=scheme_delta.mean(1)
    rng=np.random.default_rng(3211702);boot=[]
    for _ in range(2000):
        people=rng.integers(0,2,2);draw=[]
        for person in people:draw.append(np.mean([delta[person,rng.choice(s,len(s),replace=True)].mean() for s in SCHEMES]))
        weights=w[people];boot.append(float(weights@draw/weights.sum()))
    result=dict(status='READONLY_POSTHOC_SUPPLEMENTAL_EQUAL_SCHEME_NOT_PRIMARY_NOT_GENERALIZATION',
        primary_report=str(PRIMARY),primary_sha256=original,primary_uniform8_JS_minus_base_pp=uniform,
        primary_uniform8_95pct_descriptor_pp=p['people_order_descriptive_interval_pp'],
        primary_orderweights=dict(within=.5,full=.25,runs=.25),supplemental_orderweights=dict(within=1/3,full=1/3,runs=1/3),
        supplemental_equal_scheme_JS_minus_base_pp=float(w@equal),supplemental_equal_scheme_80pct_descriptor_pp=np.percentile(boot,[10,90]).tolist(),
        supplemental_person_JS_minus_base_pp={name:float(equal[i]) for i,name in enumerate(('P04','P10'))},
        person_order_JS_minus_base_pp={name:{o:float(delta[i,j]) for j,o in enumerate(ORDERS)} for i,name in enumerate(('P04','P10'))},
        person_scheme_JS_minus_base_pp={name:{s:float(scheme_delta[i,j]) for j,s in enumerate(('within','full','runs'))} for i,name in enumerate(('P04','P10'))},
        source_sha256=sha('source.sha256'),scope='samecompleteactualCPU3armrecordedofficialmetrics;oldauxNN/classical/teacherG/OOFexposure;ONLY2people',
        caveats='Posthocconventioncheck;uniform8primaryunchanged;80%supplementdifferentfrom95%primary;NOmodel/evaluator/HPOchange;NOsealed/rank/promotionproof')
    if sha(PRIMARY)!=original:raise ValueError('primary report changed')
    Path('supplement.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
