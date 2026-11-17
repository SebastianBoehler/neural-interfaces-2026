"""Samefold/seed CC30 reconstruction and export; no ground-truth scoring or promotion."""
import json
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

from reference import MANIFEST,SOURCE,compare,export,original,sha


def main():
    started=time.monotonic();ws=Path.cwd()
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise ValueError('registered samehardware A100 guard unavailable')
    torch.set_num_threads(1)
    module=original();d=module.protocol.load()
    held,train,test=list(module.protocol.folds(d,'all_loso'))[0]
    if held!='P01':raise ValueError('expected registered P01fold')
    rows=np.flatnonzero(train);held_people=set(d['subject'][test]);fit_people=sorted(set(d['subject'][train]))
    if len(fit_people)!=9 or held_people&set(fit_people):raise ValueError('source boundary failed')
    archive=SOURCE/'runs/coral304/conditional30_s0_P01.npz'
    with np.load(archive,allow_pickle=False) as z:
        reference={k:z[k].copy() for k in z.files if k!='metadata'}
        previous=json.loads(str(z['metadata']))
    validation=json.loads((SOURCE/'runs/coral304_streams/validation.json').read_text())
    if validation['producer_hashes'].get(archive.name)!=sha(archive):
        raise ValueError('original owning assembler did not validate this archive')
    if previous['manifest_sha256']!=MANIFEST or previous['fit_people']!=fit_people:
        raise ValueError('historical recipe/source rows differ')
    if sha(SOURCE/'data/proteus_windows.npz')!=previous['data_sha256']:
        raise ValueError('historical training data changed')
    if hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest()!=previous['train_rows_sha256']:
        raise ValueError('exact historical source row list differs')
    if sha(SOURCE/'runs/shuffle_perms.npz')!=previous['permutation_sha256']:
        raise ValueError('historical order file changed')
    report=dict(source_manifest=sha(ws/'source.sha256'),original_source_manifest=MANIFEST,
                archive_sha256=sha(archive),held=held,seed=0,fit_people=fit_people,metrics_computed=False,
                gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),
                tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32,cudnn=torch.backends.cudnn.allow_tf32),
                historical=previous,runs=[])
    outputs=[]
    # Predeclared diagnostic repeats run only if the first historical guard fails.
    for repeat in range(3):
        model=module.make(d,'conditional30',0);training=time.monotonic();epochs=model.epochs
        model.fit(rows);elapsed=time.monotonic()-training;model.set_k(2.)
        if set(model.fit_subjects)!=set(fit_people) or model.repair_stats['gradient_norm']<=0:
            raise ValueError('source/conditional gradient activity failed')
        prediction,parameter_hash=module.infer(model,d,test)
        result=compare(prediction,reference);outputs.append(prediction)
        checkpoint=ws/f'conditional30_P01_seed0_repeat{repeat}.pt'
        torch.save(dict(state={k:v.detach().cpu() for k,v in model.net.state_dict().items()},G=model.G,
                        scale=model.scale,seed=0,cfg='fb_sbn_rot05',conditional_coral=30.,fit_people=fit_people,
                        held=held,epochs=epochs,parameter_sha256=parameter_hash,reference_guard=result),checkpoint)
        record=dict(repeat=repeat,epochs=epochs,training_seconds=elapsed,parameter_sha256=parameter_hash,
                    checkpoint_sha256=sha(checkpoint),historical_guard=result,penalty=model.repair_stats)
        if repeat==0 and result['passed']:
            first=np.flatnonzero(test)
            session=d['stream'][first[0]];first=first[d['stream'][first]==session]
            inputs=model._aligned(first,causal=True).astype(np.float32)
            try:
                record['export']=export(model,module,ws/'cc30_P01_s0_k2.pt',inputs)
            except Exception as error:
                record['export_error']=repr(error);report['runs'].append(record)
                report['status']='historical_guard_passed_export_failed';report['all10_release_authorized']=False
                (ws/'result.json').write_text(json.dumps(report,indent=2))
                raise
            report['runs'].append(record);report['status']='historical_member_and_export_guard_passed'
            break
        report['runs'].append(record)
        report['status']='historical_guard_failed_diagnostic_repeats_only'
        del model;torch.cuda.empty_cache()
    if len(outputs)>1:
        report['repeat_discrepancies']=[dict(left=i,right=j,comparison=compare(outputs[i],outputs[j]))
                                       for i in range(len(outputs)) for j in range(i+1,len(outputs))]
    report['seconds']=time.monotonic()-started
    report['all10_release_authorized']=False
    (ws/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status=report['status'],runs=report['runs'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
