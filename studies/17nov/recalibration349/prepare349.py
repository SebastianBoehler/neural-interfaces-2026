"""Source-only phase/ONEBN diagnostics and matched final-weight statistic replay."""
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from source349 import load,bn,install,parameter_sha,permutations,sha,digest,state_sha
from moments349 import phase,oracle,recalibrate,quality,describe


def main():
    root=Path(os.environ['RECAL349_ROOT']);start=time.monotonic();torch.set_num_threads(2)
    d,fit,people,y,eligible,late,models,records,qualification=load();perm=permutations(len(people))['within0']
    positions=np.searchsorted(fit,eligible);report={};buffers={};maximum_cost=0.;source_rows=0
    for arm,model in models.items():
        parameters=parameter_sha(model.net);original_state=state_sha(model.net)
        old_mean=bn(model).running_mean.detach().numpy().copy();old_var=bn(model).running_var.detach().numpy().copy()
        buffers[arm+'_old_mean']=old_mean;buffers[arm+'_old_var']=old_var
        original,saved=phase(model,d,fit,d['X'][fit],perm,fixture=True);later,_=phase(model,d,eligible,late,perm)
        measured=quality(original['proba'],y);expected=records[arm]['source_quality']
        if abs(measured['pooledNLL']-expected['pooledNLL'])>1e-6 or measured['class_recalls']!=expected['class_recalls']:raise ValueError('original347source task loss/recalls not reproduced')
        tests=oracle(model,saved['activation']);diagnostics=describe(original,later,positions,y[positions])
        cases=[(arm+'_recal',False)] if arm=='B' else [('C_mixed_recal',True),('C_official_recal',False)]
        variants={}
        # Real source stream costs include native preprocessing, not a synthetic timing fixture.
        rank=np.empty(len(perm),int);rank[perm]=np.arange(len(perm));rr=fit[d['stream'][fit]==d['stream'][fit[0]]];rr=rr[np.argsort(rank[rr])]
        for name,mixed in cases:
            mean,var,proof=recalibrate(model,fit,eligible,original,later,records[arm],mixed)
            buffers[name+'_mean']=mean;buffers[name+'_var']=var;changed=install(model,mean,var)
            adjusted,_=phase(model,d,fit,d['X'][fit],perm)
            # Only network state is copied, never the dataset-bearing AlignedNet object.
            x=torch.from_numpy(model._aligned(rr[:7],True));net=copy.deepcopy(model.net).eval()
            with torch.no_grad():small=torch.softmax(net(x).double(),1).numpy()
            direct=model.predict_proba_stream(rr[:7])
            if not np.array_equal(small,direct):raise ValueError('real source net-only deepcopy identity failed')
            times=[]
            for _ in range(2):
                tick=time.monotonic();model.predict_proba_stream(rr);times.append(time.monotonic()-tick)
            maximum_cost=max(maximum_cost,max(times)/len(rr));source_rows=len(rr)
            variants[name]=dict(replay=proof,changed=changed,source_original_quality=quality(adjusted['proba'],y),source_full_session_seconds=times,
                prior_mean_shift_RMS=float(np.sqrt(np.mean((mean-old_mean)**2/np.maximum(old_var,1e-12)))),
                prior_logvariance_shift_RMS=float(np.sqrt(np.mean(np.log(np.maximum(var,1e-12)/np.maximum(old_var,1e-12))**2))))
            with torch.no_grad():bn(model).running_mean.copy_(torch.from_numpy(old_mean));bn(model).running_var.copy_(torch.from_numpy(old_var))
            if state_sha(model.net)!=original_state or parameter_sha(model.net)!=parameters:raise ValueError('source intervention restore/weight identity failed')
        report[arm]=dict(checkpoint_sha256=records[arm]['checkpoint_sha256'],parameter_sha256=parameters,original_state_sha256=original_state,
            BN_name='spatial_conv.1',BN_channels=288,BN_group=32,BN_momentum=.1,BN_prior=20.,source_oracle=tests,
            original_source_quality=measured,matched_official_quality=quality(original['proba'][positions],y[positions]),
            late_source_quality=quality(later['proba'],y[positions]),phase_geometry=diagnostics,variants=variants)
    path=root/'buffers.npz'
    with path.open('xb') as out:np.savez(out,**buffers)
    query=np.flatnonzero(np.isin(people,['P02','P03','P09']));prediction=1.5*maximum_cost*len(query)*8*5*1.25+120
    elapsed=time.monotonic()-start;coreseconds=2*(elapsed+20)+6*(prediction+20)+240
    output=dict(passed=True,budget_passed=bool(prediction<=360 and coreseconds<=3600),source_sha256=sha(root/'source.sha256'),
        buffer_sha256=sha(path),source_rows_sha256=digest(fit),eligible_rows_sha256=digest(eligible),late_sha256=qualification['late_sha256'],
        source_people=sorted(set(people[fit])),records=report,query_neural_forwards=False,query_labels_opened=False,
        optimizer_steps=0,predicted_query_seconds=prediction,predicted_total_CPUcoreseconds=coreseconds,
        timing_source_session_rows=source_rows,maximum_source_per_window_seconds=maximum_cost,seconds=elapsed)
    (root/'prepare.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:v for k,v in output.items() if k!='records'}),flush=True)
    if not output['budget_passed']:raise ValueError('one-CPUh source-measured counterfactual budget stop')


if __name__=='__main__':main()
