"""Exact raw-source official-crop qualification before saving any labelled late data."""
import importlib.util
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from metadata347 import plan,source,restored,BASE,sha,digest

OWNER=Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_data1/data.py')
OWNER_SHA='5aa3e0af44a362e10ab151058d456f72fb98ed7d640e47acb1f818b989ce102d'


def main():
    root=Path(os.environ['CROP_ROOT']);start=time.monotonic();torch.set_num_threads(2)
    d,fit,held,people,y=source();groups,eligible,summary=plan(d,fit,people,y)
    if sha(OWNER)!=OWNER_SHA:raise ValueError('immutable preprocessing owner changed')
    spec=importlib.util.spec_from_file_location('raw347owner',OWNER);owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    model,card=restored(d,fit,held);chunks=[];rows=[];records=[];maximum=0.;prefix=0.
    try:
        for i,group in enumerate(groups):
            path=Path(group['path']);tick=time.monotonic();values=owner.processed(path,d['ch_names'].astype(str).tolist())
            # NeuralSet caches float32 VOLTS before its later microvolt scale_factor.
            # The immutable312 helper returns float64 microvolts; recover volts first.
            values=(values/1e6).astype(np.float32)*np.float32(1e6)
            rr=np.asarray([r['row'] for r in group['rows']],np.int64)
            original=np.stack([values[:,r['original_start']:r['original_start']+480] for r in group['rows']]).astype(np.float32)
            if original.shape!=(len(rr),41,480) or not np.isfinite(original).all():raise ValueError('raw source original schema/bounds')
            error=float(np.max(np.abs(original-d['X'][rr])));maximum=max(maximum,error)
            if error>2e-4:raise ValueError('raw/official ALLsource-row numerical parity failed:'+str((group['path'],error)))
            if i==0:
                n=min(7,len(rr));a=torch.from_numpy((original[:n].astype(np.float64)/model.scale).astype(np.float32))
                b=torch.from_numpy(model._aligned(rr[:n],True))
                with torch.no_grad():pa=torch.softmax(model.net(a).double(),1).numpy();pb=torch.softmax(model.net(b).double(),1).numpy()
                prefix=float(np.max(np.abs(pa-pb)))
                if prefix>1e-6 or not np.array_equal(pa.argmax(1),pb.argmax(1)):raise ValueError('actualA6 original raw-source prefix posterior failed')
                (root/'first_file_smoke.json').write_text(json.dumps(dict(passed=True,held_labels_opened=False,
                    source_sha256=sha(root/'source.sha256'),path=str(path),max_input_error=error,prefix_probability_error=prefix,
                    rows_sha256=digest(rr),seconds=time.monotonic()-start)))
            local=[r for r in group['rows'] if r['eligible']]
            late=np.stack([values[:,r['late_start']:r['late_start']+480] for r in local]).astype(np.float32)
            if late.shape!=(len(local),41,480) or not np.isfinite(late).all():raise ValueError('late source crop schema/bounds')
            chunks.append(late);rows.extend(r['row'] for r in local)
            record={k:v for k,v in group.items() if k!='rows'}
            record.update(raw_sha256=sha(path),original_rows_sha256=digest(rr),original_reconstructed_sha256=digest(original),
                late_rows_sha256=digest(np.asarray([r['row'] for r in local],np.int64)),max_input_error=error,seconds=time.monotonic()-tick)
            records.append(record);print('sourcefile',i+1,len(groups),'maxabs',error,flush=True)
        order=np.argsort(rows);rows=np.asarray(rows,np.int64)[order];late=np.concatenate(chunks)[order]
        if not np.array_equal(rows,eligible):raise ValueError('complete eligible-row map differs')
        with (root/'late.npz').open('xb') as out:np.savez(out,index=rows,X=late)
        report=dict(passed=True,source_sha256=sha(root/'source.sha256'),owner_sha256=OWNER_SHA,
            data_sha256=sha(BASE/'data/proteus_windows.npz'),summary=summary,files=records,
            original_source_rows_sha256=digest(fit),late_sha256=sha(root/'late.npz'),late_input_sha256=digest(late),
            original_scale=model.scale,checkpoint_sha256=card['checkpoint_sha256'],max_original_error=maximum,
            first_source_prefix_error=prefix,source_original_RMS=float(np.sqrt(np.mean(d['X'][eligible].astype(np.float64)**2))),
            source_late_RMS=float(np.sqrt(np.mean(late.astype(np.float64)**2))),held_labels_opened=False,
            query_transformed=False,seconds=time.monotonic()-start)
        (root/'qualification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)
    except Exception as error:
        (root/'failed.json').write_text(json.dumps(dict(error=str(error),source_sha256=sha(root/'source.sha256'),
            completed_source_files=records,metadata_summary=summary,held_labels_opened=False,seconds=time.monotonic()-start),indent=2))
        raise


if __name__=='__main__':main()
