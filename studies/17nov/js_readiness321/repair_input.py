"""Repair EA input metadata; prove real raw-input export against all original held archives."""
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from reference import SOURCE,original,sha
from proofs import ORDERS

OLD=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_recovery6')
BASE=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_active4/base_v66.zip')


def main():
    ws=Path.cwd();torch.set_num_threads(1)
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('explicit A100 raw-input guard required')
    sys.path.insert(0,'/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/2026-competition/tracks/bci_decoding')
    import benchmark_utils.base_solver
    with zipfile.ZipFile(BASE) as archive:
        namespace={'__name__':'original_input321'};exec(compile(archive.read('submission.py'),'original_input321','exec'),namespace)
        z=np.load(io.BytesIO(archive.read('weights.npz')));expected={k:z['net35_'+k] for k in ('G','scale','k','align','shrink')}
    if float(expected['k'])!=5.:raise ValueError('actual original causal EA input-k differs')
    m=original();d=m.protocol.load();held,train,test=list(m.protocol.folds(d,'all_loso'))[0]
    old=json.loads((OLD/'recovery.json').read_text());report=dict(old,models=[])
    folder=ws/'held';folder.mkdir(exist_ok=False);common=None
    for row in old['models']:
        row=dict(row);seed=row['seed']
        for kind in ('checkpoint','trace'):
            if sha(row[kind])!=row[kind+'_sha256']:raise ValueError('historical recovered artifact changed')
            target=folder/Path(row[kind]).name;shutil.copy2(row[kind],target);row[kind]=str(target)
        ck=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        p=dict(expected,G=ck['G'],scale=ck['scale'])
        if common is None:common=p
        elif any(not np.array_equal(common[k],p[k]) for k in p):raise ValueError('source priors differ by seed')
        model=torch.jit.load(row['trace'],map_location='cuda');checks={}
        with np.load(SOURCE/f'runs/coral304/conditional30_s{seed}_P01.npz',allow_pickle=False) as archive:
            for order in ORDERS:
                idx=archive[order+'|index'];proba=archive[order+'|proba'];output=[]
                for session in dict.fromkeys(d['stream'][idx]):
                    loc=np.flatnonzero(d['stream'][idx]==session);stream=namespace['InputStream'](p);m.SB.reset_stream(model)
                    with torch.no_grad():
                        for i in loc:
                            value=stream(d['X'][idx[i]].astype(np.float32));x=torch.from_numpy(value[None]).cuda()
                            output.append((i,torch.softmax(model(x).double(),1)[0].cpu().numpy()))
                got=np.empty_like(proba)
                for i,value in output:got[i]=value
                error=float(abs(got-proba).max());margin=np.diff(np.sort(proba,axis=1)[:,-2:],axis=1)[:,0]
                changed=int(((got.argmax(1)!=proba.argmax(1))&(margin>2e-4)).sum())
                checks[order]=dict(max_error=error,non_tied_changes=changed)
                if error>=1e-4 or changed:raise ValueError('corrected real raw-input export differs from historical source')
        row['raw_input_guard']=checks;report['models'].append(row)
    np.savez(ws/'held_input.npz',**common)
    report.update(source_sha256=sha(ws/'source.sha256'),held_input_sha256=sha(ws/'held_input.npz'),
                  raw_input_k=5.,input_prior_k=5.,latent_bn_k=2.,latent_sessionbn_k=2.,raw_input_repair='oldk2metadata_invalid;newk5matchesactualsource',
                  raw_input_all_seeds_orders_passed=True)
    (ws/'recovery.json').write_text(json.dumps(report,indent=2));print('real raw-input6seeds8orderspassed',flush=True)


if __name__=='__main__':main()
