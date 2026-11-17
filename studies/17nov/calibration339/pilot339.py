"""One nested source-held calibration/cost prototype; NO outer metrics or matrix."""
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from inputs339 import src,inputs,logits,prepare,fingerprint,OUTER,CAL,VIEWS
from temperature339 import ensemble,apply,weights,fit


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    root=Path(os.environ['CAL_ROOT']);torch.set_num_threads(6);start=time.monotonic()
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise ValueError('registeredA100only')
    torch.cuda.reset_peak_memory_stats();d,rows,cal,people,perms=inputs();records={};banks={};transfer={}
    # Actual one-epoch same323trainer timing and gradient probe; no calibration labels.
    m=src.model322; m.SEED=0;x,y,scale,mu,sd,cw,groups=m.setup(d,rows,'cuda',epochs=1)
    probe=m.Factor('population',False,'cuda');before=fingerprint(probe);tick=time.monotonic()
    smoke=m.train(probe,x,y,mu,sd,cw,groups);torch.cuda.synchronize();seconds=time.monotonic()-tick
    if before==fingerprint(probe) or smoke['max_gradient']<=0:raise ValueError('source8trainingno-op')
    projection=1.5*seconds*40*2+120
    (root/'cost.json').write_text(json.dumps(dict(smoke=smoke,seconds=seconds,projection_seconds=projection,limit_seconds=540,device=torch.cuda.get_device_name(),source_sha256=src.sha(root/'source.sha256'),calibration_labels_opened=False)))
    if projection>540:raise ValueError(f'prototypebudgetstop:{projection}')
    del probe,x,y;gc.collect();torch.cuda.empty_cache()
    for seed in (0,1):
        m.SEED=seed;x,y,scale,mu,sd,cw,groups=m.setup(d,rows,'cuda');model=m.Factor('population',False,'cuda')
        initial=fingerprint(model);tick=time.monotonic();training=m.train(model,x,y,mu,sd,cw,groups);torch.cuda.synchronize()
        if training['steps']!=sum(map(len,groups)) or len(groups)!=40:raise ValueError('fixedrecipeincomplete')
        trained=fingerprint(model);folder=root/f'source8_s{seed}';folder.mkdir(exist_ok=False)
        torch.save(dict(state=model.state_dict(),scale=scale,seed=seed,outer=OUTER,cal=CAL),folder/'model.pt')
        bank,checks=logits(model,scale,d,cal,perms,'cuda');banks[seed]=bank
        if fingerprint(model)!=trained:raise ValueError('fittedweightsmutated')
        # Saved SOURCE9 model excludes OUTER but includes CAL; transfer is only numerical feasibility,
        # never held performance or temperature fitting on its in-sample logits.
        original=src.load(OUTER,seed,'cuda');oldnet,oldscale=original[5]['P'];oldbank,oldchecks=logits(oldnet,oldscale,d,cal,perms,'cuda');transfer[seed]=oldbank
        records[str(seed)]=dict(initial=initial,trained=trained,training=training,seconds=time.monotonic()-tick,
            checkpoint_sha256=src.sha(folder/'model.pt'),scale=scale,checks=checks,source9_checks=oldchecks,
            original_source9_checkpoint=original[7]['checkpoints']['P'],
            batches_sha256=hashlib.sha256(b''.join(b.tobytes() for ep in groups for b in ep)).hexdigest(),
            fit_input_sha256=digest(d['X'][rows]),fit_rows_sha256=digest(rows),cal_rows_sha256=digest(cal))
        del model,x,y,oldnet,original;gc.collect();torch.cuda.empty_cache()
    expected={v+'|'+o for v in VIEWS for o in perms}
    if any(set(b)!=expected for b in list(banks.values())+list(transfer.values())):raise ValueError('complete2seed3view8orderbanksfirst')
    if any(not np.isfinite(p).all() for b in list(banks.values())+list(transfer.values()) for p in b.values()):raise ValueError('nonfinitelogits')
    with (root/'source_logits.npz').open('xb') as f:
        np.savez_compressed(f,index=cal,**{f'{kind}s{s}|{k}':v for kind,items in (('held8',banks),('insample9',transfer)) for s,b in items.items() for k,v in b.items()})
    # ONLY NOW open calibration labels. P01 outer labels are NEVER read.
    yc=src.source_labels.read_training_labels(src.data323.DATA,cal,len(d['X']))
    cells=np.char.add(np.char.add(d['stream'][cal],'|'),d['interface'][cal]);w=weights(yc,cells);result={}
    import protocol
    if Path(protocol.__file__).resolve()!=(src.PROD/'experiments/protocol.py').resolve():raise ValueError('wrongowningmetricAPI')
    for view in VIEWS:
        source={o:ensemble([banks[s][view+'|'+o] for s in (0,1)]) for o in perms}
        for o,lp in source.items():
            identity=apply(lp,1.)
            if not np.allclose(lp,identity,atol=1e-12,rtol=0):raise ValueError('identitytemperature')
            own=protocol.score(yc,lp.argmax(1),d,people==CAL)['balanced_accuracy_cell_mean']
            if not np.isclose(own,np.dot(w,lp.argmax(1)==yc),atol=1e-12):raise ValueError('officialcell/classweightparity')
        answer=fit(source,yc,w,tuple(perms));t=answer['T'];changed=0.
        for o in perms:
            lp=ensemble([transfer[s][view+'|'+o] for s in (0,1)]);calibrated=apply(lp,t)
            if not np.array_equal(lp.argmax(1),calibrated.argmax(1)):raise ValueError('transferredtemperaturechangedargmax')
            changed=max(changed,float(abs(np.exp(lp)-np.exp(calibrated)).max()))
        answer['source9_numerical_transfer_max_probability_change']=changed
        answer['active_fit_and_transfer']=answer['max_probability_change']>1e-7 and changed>1e-7
        result[view]=answer
    report=dict(scope='ONEsourceheldcalibrationengineering/objectiveprototype;NOoutermetrics/consumer/matrix',outer=OUTER,calibration=CAL,
        fit_people=sorted(set(people[rows])),source_manifest_sha256=src.sha(root/'source.sha256'),source323=src.SOURCE,
        source332='6c0f967ca5e8a7240b676de483bdc575eab2918c526fee4b832dc107d925abab',records=records,
        temperatures=result,smoke=smoke,smoke_seconds=seconds,conservative_projection_seconds=projection,
        logits_sha256=src.sha(root/'source_logits.npz'),outer_labels_opened=False,calibration_labels_opened_after_complete_banks=True,
        device=torch.cuda.get_device_name(),peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start,
        seed_aggregation='mean two seed probabilities BEFORE one temperature per F/B/V view',
        limitation='NLL fitted/evaluated on same sourceheld calibrationperson; optimization not independent calibrationgeneralization; source8to9shift')
    (root/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
