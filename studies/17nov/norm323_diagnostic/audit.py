"""Selected post-hoc attenuation diagnostic, original complete-matrix guards/metrics."""
import json
import os
import time
from pathlib import Path
import numpy as np
from inputs import load,CHAIN,PRODUCER,ARMS,WHOLE,PINS
from metrics import weights,reduce,correct,probability,decomposition


def main():
    started=time.monotonic();root=Path(os.environ['DIAGNOSTIC_ROOT'])
    s,output,seeds,member,pre,y,cell,person,clips=load()
    w,owners=weights(y,cell,person,s.PEOPLE)
    counts=np.array([len(np.unique(cell[person==p])) for p in s.PEOPLE])
    def summary(values):
        d=np.stack(values)
        return {'people_order':s.summarize(d,s.ORDERS,s.PEOPLE),
                'official_equal_cell':s.cell_weighted_summary(d,s.ORDERS,counts)}
    # Independent row-weight expression must reproduce the original score before interpretation.
    for arm in ('control',)+ARMS:
        for order in s.ORDERS:
            for base in s.BASES:
                scores=output[arm,order,base]
                if not np.allclose(100*reduce(correct(scores,y),w,owners),s.person_accuracy(scores,y,cell,person,s.PEOPLE),atol=1e-10,rtol=0):
                    raise ValueError('row weights do not match original scorer')
    whole=json.loads((PRODUCER/'whole_results.json').read_text());scope={}
    for arm in ARMS:
        separate=[np.mean([s.person_accuracy(seeds[arm,k,o],y,cell,person,s.PEOPLE) for k in (0,1)],axis=0) for o in s.ORDERS]
        ensemble=[s.person_accuracy(member[arm,o],y,cell,person,s.PEOPLE) for o in s.ORDERS]
        independent=summary(separate);expected=whole['official_equal_cell_scheme_equal'][arm]
        if abs(independent['official_equal_cell']['mean']-expected)>1e-9:raise ValueError('whole seed score replay mismatch')
        scope[arm]={'mean_separate_seed_metrics':independent,'actual_two_seed_member':summary(ensemble)}
    contrasts={}
    for fixed,causal in (('P0F','P0C'),('S0F','S0C')):
        metrics={};stage_arrays={}
        def add(name,o,value):metrics.setdefault(name,{})[o]=value
        for order in s.ORDERS:
            f,c=member[fixed,order],member[causal,order]
            cf,cc=correct(f,y),correct(c,y)
            corr,dest,delta=decomposition(f,c,y,w,owners)
            add('member_correction_pp',order,corr);add('member_destruction_pp',order,dest);add('member_accuracy_delta_pp',order,delta)
            separate=np.mean([s.person_accuracy(seeds[causal,k,order],y,cell,person,s.PEOPLE)-s.person_accuracy(seeds[fixed,k,order],y,cell,person,s.PEOPLE) for k in (0,1)],axis=0)
            add('separate_seed_accuracy_delta_pp',order,separate)
            for metric,value in probability(c,y,w,owners).items():
                fixed_value=probability(f,y,w,owners)[metric]
                add('member_'+metric+'_delta',order,value-fixed_value)
                add('fixed_member_'+metric,order,fixed_value)
                add('causal_member_'+metric,order,value)
                incumbent=np.mean([probability(pre['control',order,b],y,w,owners)[metric] for b in s.BASES],axis=0)
                add('control_V6Fr_'+metric,order,incumbent)
            for stage,incumbent in (('V6Fr',pre),('finalCP',output)):
                hit=np.mean([correct(incumbent['control',order,b],y) for b in s.BASES],axis=0)
                # hit is a six-base mean, not a pseudo-independent sample.
                for state,weight in (('correct',hit),('wrong',1-hit)):
                    add(f'member_corrections_on_control_{stage}_{state}_pp',order,100*reduce((~cf&cc)*weight,w,owners))
                    add(f'member_destructions_on_control_{stage}_{state}_pp',order,100*reduce((cf&~cc)*weight,w,owners))
                for label,own in (('fixed',cf),('causal',cc)):
                    for name,value in {'both_correct':own*hit,'member_only_correct':own*(1-hit),
                                       'control_only_correct':(~own)*hit,'both_wrong':(~own)*(1-hit)}.items():
                        add(f'{label}_member_control_{stage}_{name}_pp',order,100*reduce(value,w,owners))
                for kind,total in (('corrections',corr),('destructions',dest)):
                    key=f'member_{kind}_on_control_{stage}_'
                    if not np.allclose(metrics[key+'correct_pp'][order]+metrics[key+'wrong_pp'][order],total,atol=1e-10):
                        raise ValueError('overlap partition identity')
            before=[];after=[];correction_effect=[];destruction_effect=[]
            for base in s.BASES:
                pf,pc=pre[fixed,order,base],pre[causal,order,base]
                of,oc=output[fixed,order,base],output[causal,order,base]
                cfcp,dfcp,gf=decomposition(pf,of,y,w,owners)
                cccp,dccp,gc=decomposition(pc,oc,y,w,owners)
                a=s.person_accuracy(pc,y,cell,person,s.PEOPLE)-s.person_accuracy(pf,y,cell,person,s.PEOPLE)
                b=s.person_accuracy(oc,y,cell,person,s.PEOPLE)-s.person_accuracy(of,y,cell,person,s.PEOPLE)
                if not np.allclose(b-a,(cccp-dccp)-(cfcp-dfcp),atol=1e-10,rtol=0):raise ValueError('CP contrast telescoping identity')
                before.append(a);after.append(b);correction_effect.append(cccp-cfcp);destruction_effect.append(dccp-dfcp)
                stage_arrays[f'{order}|{base}']=dict(pre_CP=a.tolist(),final_CP=b.tolist(),
                    fixed_CP_corrections=cfcp.tolist(),fixed_CP_destructions=dfcp.tolist(),
                    causal_CP_corrections=cccp.tolist(),causal_CP_destructions=dccp.tolist())
            add('V6Fr_accuracy_delta_pp',order,np.mean(before,axis=0))
            add('finalCP_accuracy_delta_pp',order,np.mean(after,axis=0))
            add('CP_attenuation_delta_pp',order,np.mean(after,axis=0)-np.mean(before,axis=0))
            add('CP_correction_contrast_pp',order,np.mean(correction_effect,axis=0))
            add('CP_destruction_contrast_pp',order,np.mean(destruction_effect,axis=0))
            for kind in ('margin','NLL','Brier','ECE10'):
                v=np.mean([probability(pre[causal,order,b],y,w,owners)[kind]-probability(pre[fixed,order,b],y,w,owners)[kind] for b in s.BASES],axis=0)
                add('V6Fr_'+kind+'_delta',order,v)
        stats={name:summary([values[o] for o in s.ORDERS]) for name,values in metrics.items()}
        expected=whole['contrasts'][f'{causal}_minus_{fixed}']['official_equal_cell']['mean']
        if abs(stats['separate_seed_accuracy_delta_pp']['official_equal_cell']['mean']-expected)>1e-9:raise ValueError('whole contrast replay mismatch')
        # Exact completed-chain contrast is also validated independently of text rounding.
        expected_chain={('P0F','P0C'):-.07140753572831558,('S0F','S0C'):-.05547168792588057}[fixed,causal]
        if abs(stats['finalCP_accuracy_delta_pp']['official_equal_cell']['mean']-expected_chain)>1e-9:raise ValueError('chain contrast replay mismatch')
        contrasts[f'{causal}_minus_{fixed}']={'metrics':stats,'per_order_base_people_CP':stage_arrays}
    result={'scope':'selected post-hoc complete saved-prediction diagnostic; no causal attribution or new policy',
        'source_manifest_sha256':s.sha256(root/'source.sha256'),'producer_source_sha256':whole['source_manifest_sha256'],
        'consumer_source_sha256':PINS['representation-source.sha256'],'whole_report_sha256':WHOLE,'chain_report_sha256':PINS['results.txt'],
        'labels_after_original_complete_guards':True,'seed_scope':scope,'contrasts':contrasts,
        'member_clip_below_1e6_counts':clips,'CP_probability_diagnostics':False,'total_seconds':time.monotonic()-started}
    with (root/'results.json').open('x') as out:json.dump(result,out,indent=2)
    print('OK: complete guard/seed/ensemble/row-weight/CP telescoping identities; diagnostic complete',flush=True)


if __name__=='__main__':main()
