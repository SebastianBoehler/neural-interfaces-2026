"""No-label source-mask and historical frozen-probability guards for existing returning members."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from returning_train import prepare,sha

MAIN=Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
FAMILIES={'fb':('fb_sbn_rot05',2.,'fb'),'fbl':('fblight_ea_rot05_sbn',5.,'fbl'),
          'fbraw':('fbcnet_raw_sbn',20.,'raw')}


def main(device='cpu'):
    if device not in ('cpu','cuda'):raise ValueError('explicit device required')
    if device=='cuda' and (not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0)):raise ValueError('A100 required')
    torch.set_num_threads(1);reports=[]
    for fold in ('P04_s23','P10_s23'):
        d,_,sb,train,held,_,_=prepare(fold,0)
        features=MAIN/f'runs/replay_data/feats_exact/seen_session-{fold}.npz'
        with np.load(features,allow_pickle=False) as z:
            if not np.array_equal(np.sort(z['idx_tr']),train):raise ValueError('classic/gate training rows differ')
        with np.load(MAIN/'runs/shuffle_perms.npz') as z:delivery=z['within|0|seen_session'].copy()
        for family,(cfg,k,stem) in FAMILIES.items():
            path=MAIN/f'runs/diag/pdeep/nets/{stem}-{fold}-s0.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if set(checkpoint)!={'state','G','scale'}:raise ValueError('checkpoint schema differs')
            model=sb.make(cfg,d,0);model.dev='cpu'
            if not np.array_equal(checkpoint['G'],model.C[train].mean(0)):
                raise ValueError('checkpoint normalization not exact source-fold mean')
            model.G,model.scale=checkpoint['G'],checkpoint['scale']
            if abs(model.scale-float(np.sqrt(np.trace(model.G)/41)))>1e-12:raise ValueError('source scale differs')
            from deep import build
            model.net=build(model.arch,41,480,model.pretrained,**model.net_kw)
            model._prepare();model.net.load_state_dict(checkpoint['state'],strict=True);model.net.eval();model.set_k(k);model.net.to(device)
            archive=MAIN/f'runs/ttt/probs18/ttt18within0QF_{family}_base-seen_session-s0.npz'
            with np.load(archive,allow_pickle=False) as z:idx=z['index'].copy();ref=z['proba'].copy()
            if not np.all(np.diff(idx)>0):raise ValueError('archive canonical index differs')
            errors=[];changed=0;count=0
            with torch.no_grad():
                for session in dict.fromkeys(d['stream'][held]):
                    rows=delivery[d['stream'][delivery]==session]
                    if not np.array_equal(np.sort(rows),held[d['stream'][held]==session]):raise ValueError('delivery/held source differs')
                    x=torch.from_numpy(np.ascontiguousarray(model._aligned(rows,causal=True),dtype=np.float32)).to(device)
                    got=model.net(x)[:,:3].double().softmax(1).cpu().numpy();repeat=model.net(x)[:,:3].double().softmax(1).cpu().numpy()
                    if float(np.abs(got-repeat).max())>1e-8:raise ValueError('same-checkpoint self-repeat differs')
                    pos=np.searchsorted(idx,rows)
                    if not np.array_equal(idx[pos],rows):raise ValueError('archive row mismatch')
                    errors.append(float(np.abs(got-ref[pos]).max()));changed+=int(np.count_nonzero(got.argmax(1)!=ref[pos].argmax(1)));count+=len(rows)
            report=dict(device=device,torch=torch.__version__,tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,fold=fold,family=family,seed=0,max_abs=max(errors),changed_labels=changed,rows=count,
                        checkpoint_sha256=sha(path),archive_sha256=sha(archive),classic_source_sha256=sha(features))
            reports.append(report);print(json.dumps(report),flush=True)
            if max(errors)>1e-5 or changed:raise ValueError('existing fold checkpoint historical frozen parity failed')
    Path('existing_guard_'+device+'.json').write_text(json.dumps(dict(status='existing_seed0_returning_sources_and_frozen_parity_passed',
        reports=reports,source_sha256=sha('source.sha256'),accuracy_computed=False),indent=2))


if __name__=='__main__':main(*sys.argv[1:])
