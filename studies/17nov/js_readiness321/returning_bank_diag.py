"""Source-only full-prefix numerical diagnosis; never scores accuracy or rescues a failed guard."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

import returning_bank as bank


def fingerprint(model,state):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        if k.rsplit('.',1)[-1] not in state:h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(tf32):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=bool(tf32)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.set_num_threads(1)
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('A100 required')
    sys.path.insert(0,str(bank.ROOT/'ni26_js321_20261117_returning2/source/experiments'))
    from session_bn import STATE,reset_stream
    with np.load(bank.DATA,allow_pickle=True) as z:x=z['X'];sub=z['subject'];session=z['session'].astype(str)
    stream=sub+'|'+session
    with np.load(bank.ROOT/'neural-interfaces-2026/runs/shuffle_perms.npz') as z:delivery=z['within|0|seen_session']
    rows=delivery[stream[delivery]=='Dreyer2026Proteus/P04|2'];Input=bank.input_class();reports=[]
    for kind in bank.KINDS:
        ckpt,trace,proof=bank.asset(kind,'P04_s23',0);c=torch.load(ckpt,map_location='cpu',weights_only=False)
        raw=Input(dict(align=np.array(kind!='fbraw'),k=np.array(5.),G=c['G'],scale=np.array(c['scale']),shrink=np.array(.001)))
        inputs=np.stack([raw(x[r].astype(np.float32)) for r in rows]);model=torch.jit.load(str(trace),map_location='cuda').eval()
        before=fingerprint(model,STATE);records=[]
        for n in (17,31,len(inputs)):
            t=torch.from_numpy(inputs[:n]).cuda();reset_stream(model)
            with torch.no_grad():
                for i in range(17):model(t[i:i+1])
                reset_stream(model);a=torch.cat([model(t[i:i+1]) for i in range(n)]).double().softmax(1).cpu().numpy()
                counts={k:float(v) for k,v in model.named_buffers() if k.rsplit('.',1)[-1]=='n'}
                reset_stream(model);b=torch.cat([model(t[i:i+1]) for i in range(n)]).double().softmax(1).cpu().numpy()
                reset_stream(model);batch=model(t).double().softmax(1).cpu().numpy()
            delta=np.abs(a-batch);records.append(dict(prefix=n,window_repeat_max=float(np.abs(a-b).max()),batch_window_max=float(delta.max()),
                batch_window_relative_max=float((delta/np.maximum(np.abs(a),1e-12)).max()),changed_labels=int(np.count_nonzero(a.argmax(1)!=batch.argmax(1))),counts=counts))
            reset_stream(model)
            if any(float(v.abs().sum()) for k,v in model.named_buffers() if k.rsplit('.',1)[-1] in STATE):raise ValueError('reset buffer nonzero')
        if fingerprint(model,STATE)!=before:raise ValueError('immutable weights/buffers changed')
        reports.append(dict(kind=kind,trace_sha256=bank.sha(trace),records=records,immutable_unchanged=True));print(json.dumps(reports[-1]),flush=True)
    Path(f'diagnostic_tf32_{tf32}.json').write_text(json.dumps(dict(status='numerical_diagnostic_complete_NOT_guard_pass',
        reports=reports,torch=torch.__version__,cuda=torch.version.cuda,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,accuracy_computed=False,original_failure_preserved=True,
        source_sha256=bank.sha('source.sha256')),indent=2))


if __name__=='__main__':main(int(sys.argv[1]))
