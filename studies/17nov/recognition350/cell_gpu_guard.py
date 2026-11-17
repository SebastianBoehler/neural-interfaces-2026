"""One whole causal policy pass; owning original metadata/evaluator, no query labels."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from returning_pair_cell import inputs
from returning_pair_guard import ROOT,zero,state
from returning_pair_checks import Frames
from returning_assets import sha
from package_guard import instrument,frame,owning_mix,reference_mix
from record import load_model,ft_clock_restart
from benchmark_utils.streaming import fresh_model

GUARD=ROOT/'ni26_recognition350_20261117_gpu_guard2'
OLD=ROOT/'ni26_js321_20261117_returning_pair_cpu1'


def main(index):
    start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if torch.__version__!='2.14.1+cu130' or torch.cuda.is_initialized():raise ValueError('explicit qualified CPU required')
    guard=json.loads((GUARD/'output/guard.json').read_text())
    if guard['executor']!='cleanSIF_A100_deterministic' or guard['CPU_unknown215_proven'] or guard['status']!='POLICY350_GPU_SOURCE_FUNCTIONAL_PASS_NOT_ACCURACY' or guard['source_sha256']!=sha(GUARD/'source.sha256') or guard['build_sha256']!=sha(GUARD/'build.json'):raise ValueError('complete actual owning source guard absent')
    fold,order,X,channels,canonical,source,delivery,sessions,views,viewfile,cell_keys=inputs(index)
    oldproof=json.loads((OLD/f'cell_{index}/proof.json').read_text())
    if oldproof['index']!=index or oldproof['fold']!=fold or oldproof['order']!=order:raise ValueError('old complete cell identity differs')
    if oldproof['predictions_sha256']!=sha(OLD/f'cell_{index}/predictions.npz'):raise ValueError('complete old comparison bank changed')
    path=GUARD/'bundles'/fold/'2';build=json.loads((GUARD/'build.json').read_text())
    entry=next(v for v in build['artifacts'] if v['fold']==fold and v['mode']==2)
    if any(sha(path/k)!=v for k,v in entry['files'].items()):raise ValueError('policy source assets changed')
    template=instrument(load_model(path,channels,'policy350_whole_'+fold,'cpu'),False);zero(template)
    for r in source:frame(template,X[r])
    template.reset_state();zero(template)
    values=np.full((len(canonical),3),np.nan);labels=np.full(len(canonical),-1,dtype=np.int64);records=[];reference=owning_mix()
    with np.load(path/'weights.npz') as z:weights={k:z[k].copy(order='K') for k in z.files}
    for session in dict.fromkeys(sessions[delivery]):
        rows=delivery[sessions[delivery]==session];m=fresh_model(template);before=zero(m)['immutable_sha256'];ft_clock_restart(m)
        left,right=fresh_model(template),fresh_model(template);ft_clock_restart(left);ft_clock_restart(right);prefix=[]
        for t,r in enumerate(rows[:17]):prefix.append(frame(left,X[r])[1]);frame(right,X[rows[-1-t]])
        del left,right
        frames=Frames({'policy':m},'cpu');error=0.;viewerror=0.
        for t,r in enumerate(rows):
            if time.monotonic()-start>470:raise ValueError('prospective whole CPU8minbudget guard')
            label,p,trace=frames({'policy':m},X[r])['policy'];position=int(np.searchsorted(canonical,r))
            if (t+1)%20==0:print(json.dumps(dict(session=str(session),windows=t+1,seconds=time.monotonic()-start,recognition=m.log6['rec'],FT=m.ft.refits,CP=m.cp.fused)),flush=True)
            if t<17 and not np.array_equal(p,prefix[t]):raise ValueError('source prefix/deepcopy differs')
            expected_js=not m.recB
            if expected_js:expected,_=reference_mix(m,trace,weights,reference)
            else:expected=type(m).step.__globals__['Ensemble6'].mix6(m,trace['mix'],trace['v6']['members'],trace['v6']['p_ft'])
            error=max(error,float(np.abs(expected-trace['v6']['mix6']).max()));viewerror=max(viewerror,float(np.abs(trace['u']-views[position]).max()))
            if error>1e-8 or viewerror>1e-8 or not np.isclose(m.routeB.a,1/17 if expected_js else 1/16,rtol=0,atol=1e-15):raise ValueError('active whole policy recipe/view/denominator differs')
            if m.js_step!=t+1 or m.js_calls!=t+1 or m.js_done!=t+1 or m.js_bank.groups[0][0].n:raise ValueError('current auxiliary callback differs')
            if m.log6['err'] or m.cp.dead or m.cp.log['err'] or m.cf.dead or m.dead or m.ft.over:raise ValueError('runtime stage or FT budget failed')
            if not np.isfinite(p).all():raise ValueError('nonfinite emitted scores')
            values[position]=p;labels[position]=label
        prior=next(v for v in oldproof['records'] if v['session']==str(session))['activation']['base']
        actual=dict(rec=m.log6['rec'],ft_refits=m.ft.refits,cp_fused=m.cp.fused,cp_stopped=m.cp.stopped)
        if any(actual[k]!=prior[k] for k in actual):raise ValueError('original recognition/FT/CP activation differs')
        if state(m)['immutable_sha256']!=before:raise ValueError('original/auxiliary tensors changed')
        if m.policy350_js+m.policy350_base!=len(rows) or m.policy350_js!=(actual['rec'] if actual['rec']>=0 else len(rows)):raise ValueError('actual latch policy inactive')
        records.append(dict(session=str(session),windows=len(rows),activation=actual,js_windows=m.policy350_js,base_recipe_windows=m.policy350_base,recipe_maxabs=error,rawview_maxabs=viewerror,reset_copy_prefix_exact=True))
        Path('/out/cases_partial.json').write_text(json.dumps(dict(status='PARTIAL_SESSIONS_NOT_WHOLE_PASS',records=records,query_labels_loaded=False),indent=2))
    if (labels<0).any() or not np.isfinite(values).all():raise ValueError('incomplete whole cell')
    np.savez('/out/predictions.npz',index=canonical,policy_scores=values,policy_labels=labels)
    Path('/out/proof.json').write_text(json.dumps(dict(status='POLICY350_COMPLETE_WHOLE_CELL_NOT_SCORED',index=index,fold=fold,order=order,rows=len(canonical),official_cell_keys=cell_keys,records=records,seconds=time.monotonic()-start,source_sha256=sha('/probe/source.sha256'),owning_guard_sha256=sha(GUARD/'output/guard.json'),predictions_sha256=sha('/out/predictions.npz'),query_labels_loaded=False,executor='cleanSIF_CPU_deterministic',guard_executor='cleanSIF_A100_deterministic',CPU_unknown215_proven=False),indent=2))


if __name__=='__main__':main(int(sys.argv[1]))
