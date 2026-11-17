"""Recover the six realP01source models required by the actual318B2_pool identity."""
import ast
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from reference import MANIFEST,SOURCE,compare,export,original,sha

FIRST=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_reference1')
STUDY=Path('/mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117')


def main():
    ws=Path.cwd();started=time.monotonic();torch.set_num_threads(1)
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise ValueError('explicit original A100 route required')
    script=STUDY/'studies/14nov/newmember/run_s.py'
    tree=ast.parse(script.read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='BS' for t in n.targets))
    bases=ast.literal_eval(assignment.value)
    if bases!={'B2_s0':[0],'B2_s1':[1],'B2_s2':[2],'B2_s012':[0,1,2],
               'B2_s345':[3,4,5],'B2_pool':[0,1,2,3,4,5]}:
        raise ValueError('actual study seed identity differs')
    module=original();d=module.protocol.load()
    held,train,test=list(module.protocol.folds(d,'all_loso'))[0]
    rows=np.flatnonzero(train);people=sorted(set(d['subject'][train]))
    validation=json.loads((SOURCE/'runs/coral304_streams/validation.json').read_text())
    report=dict(status='recovering',held=held,fit_people=people,actual_study_bases=bases,
                study_runner_sha256=sha(script),source_sha256=sha(ws/'source.sha256'),models=[],metrics_computed=False)
    folder=ws/'held';folder.mkdir(exist_ok=False)
    for seed in range(6):
        archive=SOURCE/f'runs/coral304/conditional30_s{seed}_P01.npz'
        if validation['producer_hashes'].get(archive.name)!=sha(archive):
            raise ValueError('original owning collector archive differs')
        with np.load(archive,allow_pickle=False) as z:
            previous=json.loads(str(z['metadata']));reference={k:z[k].copy() for k in z.files if k!='metadata'}
        if ((previous['arm'],previous['seed'],previous['held'],previous['manifest_sha256'])
                !=('conditional30',seed,'P01',MANIFEST) or previous['fit_people']!=people):
            raise ValueError('historical actual seed/source identity differs')
        if hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest()!=previous['train_rows_sha256']:
            raise ValueError('exact historical training rows differ')
        checkpoint=folder/f'cc30_P01_s{seed}.pt';trace=folder/f'cc30_P01_k2_s{seed}.pt'
        if seed==0:
            first=json.loads((FIRST/'result.json').read_text());r=first['runs'][0]
            if first['status']!='historical_member_and_export_guard_passed' or not r['historical_guard']['passed']:
                raise ValueError('seed0 prior guard missing')
            source_ck=FIRST/'conditional30_P01_seed0_repeat0.pt';source_trace=FIRST/'cc30_P01_s0_k2.pt'
            if sha(source_ck)!=r['checkpoint_sha256'] or sha(source_trace)!=r['export']['trace_sha256']:
                raise ValueError('validated seed0 artifacts changed')
            shutil.copy2(source_ck,checkpoint);shutil.copy2(source_trace,trace)
            record=dict(seed=seed,reused_from=str(FIRST),historical_guard=r['historical_guard'],export=r['export'])
        else:
            model=module.make(d,'conditional30',seed);fit_start=time.monotonic();model.fit(rows);model.set_k(2.)
            if set(model.fit_subjects)!=set(people) or model.repair_stats['gradient_norm']<=0:
                raise ValueError('source/activity guard failed')
            prediction,parameter_sha=module.infer(model,d,test);comparison=compare(prediction,reference)
            torch.save(dict(state={k:v.detach().cpu() for k,v in model.net.state_dict().items()},G=model.G,
                            scale=model.scale,seed=seed,fit_people=people,held='P01',parameter_sha256=parameter_sha),checkpoint)
            record=dict(seed=seed,fit_seconds=time.monotonic()-fit_start,historical_guard=comparison,
                        parameter_sha256=parameter_sha,archive_parameter_sha256=previous['parameter_sha256'])
            if not comparison['passed']:
                report['models'].append(record);report['status']='historical_recovery_failed'
                (ws/'recovery.json').write_text(json.dumps(report,indent=2))
                raise ValueError('strict historical guard failed; no best-repeat selection or release')
            first_rows=np.flatnonzero(test);session=d['stream'][first_rows[0]]
            first_rows=first_rows[d['stream'][first_rows]==session]
            inputs=model._aligned(first_rows,causal=True).astype(np.float32)
            record['export']=export(model,module,trace,inputs)
            del model;torch.cuda.empty_cache()
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if seed==0:
            common=dict(G=ck['G'],scale=ck['scale'])
        elif not np.array_equal(ck['G'],common['G']) or ck['scale']!=common['scale']:
            raise ValueError('source input priors differ across seeds')
        record.update(checkpoint=str(checkpoint),trace=str(trace),checkpoint_sha256=sha(checkpoint),trace_sha256=sha(trace),archive_sha256=sha(archive))
        report['models'].append(record)
        (ws/'recovery-progress.json').write_text(json.dumps(report,indent=2))
    np.savez(ws/'held_input.npz',G=common['G'],scale=common['scale'],align=True,k=2.,shrink=.001)
    report.update(status='six_seed_historical_and_export_guards_passed',seconds=time.monotonic()-started,
                  held_input_sha256=sha(ws/'held_input.npz'))
    (ws/'recovery.json').write_text(json.dumps(report,indent=2))
    print('six actual P01CC30models recovered and exported;no accuracy scored',flush=True)


if __name__=='__main__':main()
