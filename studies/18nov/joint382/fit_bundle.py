"""Train the independently seeded §382 replication on a source-only CUDA bundle."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from kernel import schedule,train


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main(bundle,arm,seed,smoke):
    assert arm in ('F','J','K') and torch.cuda.is_available()
    meta=json.loads((bundle/'bundle.json').read_text())
    assert meta['source_only'] and meta['fixed_training_batch_length']==48
    for p,h in meta['files'].items():
        assert sha(bundle/p)==h,p
    assert sha(Path(__file__))==meta['files']['fit_bundle.py']
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    name='native' if arm=='K' else 'learned'
    model=torch.jit.load(str(bundle/(name+'.ts')),map_location='cuda').eval()
    with np.load(bundle/'source.npz',allow_pickle=False) as d:
        assert set(d.files)=={'x','y','weights','streams','source_rows'}
        streams=d['streams']
        x,y,w=[torch.from_numpy(d[k]).cuda() for k in ('x','y','weights')]
    rows=schedule(streams,382,1)[0]
    with np.load(bundle/'probes.npz') as z,torch.no_grad():
        expected=torch.from_numpy(z[name]).cuda()
        actual=model(x[rows])
        probe_error=float((actual-expected).abs().max())
        assert probe_error<.005 and torch.equal(actual.argmax(1),expected.argmax(1)),probe_error
    rec=train(model,x,y,w,streams,arm,seed,3 if smoke else 120)
    key=('smoke_' if smoke else '')+arm+'_'+str(seed)
    rec.update(arm=arm,source_sha=meta['primary_source'],bundle_sha=sha(bundle/'bundle.json'),
               source_only=True,scope=meta['people'],rows=meta['source_rows_sha256'],
               held_labels_opened=False,device=torch.cuda.get_device_name(),probe_max_error=probe_error)
    if not smoke:
        path=Path(key+'.pt')
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
        rec['checkpoint_sha']=sha(path)
    Path(key+'.json').write_text(json.dumps(rec,indent=2))
    print(key,rec,flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),sys.argv[4:]==['smoke'])
