"""Prepared deployment-only CC30 training; requires both historical and active-package parity."""
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from reference import export,original,sha

ZIP_SHA='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed'


def main(seed):
    if seed not in range(6):raise ValueError('expected deployment seed0..5')
    ws=Path.cwd();guard=json.loads((ws/'result.json').read_text())
    active=json.loads((ws/'outer_active_path.json').read_text())
    if guard['status']!='historical_member_and_export_guard_passed':
        raise ValueError('historical heldfold/source-member guard has not passed')
    if not all(active.get(k) is True for k in ('disabled_identity','original_inner_gate_identity',
                'original_ft_label_identity','study_outer_mix_parity','route_share17_verified')):
        raise ValueError('active outer-package path has not passed all required guards')
    recovery=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_inputrepair1/recovery.json')
    recovered=json.loads(recovery.read_text())
    if recovered['status']!='six_seed_historical_and_export_guards_passed':
        raise ValueError('six historical seeds have not passed')
    for order in ('within0','within1','within2','within3','full0','full1','runs0','runs1'):
        recipe=json.loads((ws/f'study_recipe_{order}.json').read_text())
        if recipe['status']!='actual318_six_draw_mix_capture_passed' or len(recipe['bases'])!=6:
            raise ValueError('complete eight-order/six-base recipe proof required')
        if recipe['generated_bundle_sha256']!=sha(ws/'draft_jsall/submission.py'):
            raise ValueError('actual-study recipe proof belongs to another generated bundle')
    if sha(ws/'base_v66.zip')!=ZIP_SHA:raise ValueError('original v66input ZIP differs')
    with zipfile.ZipFile(ws/'base_v66.zip') as archive:
        weights=np.load(io.BytesIO(archive.read('weights.npz')),allow_pickle=False)
        expected={k:weights['net35_'+k].copy() for k in ('G','scale','k','align','shrink')}
        original_code=archive.read('submission.py').decode()
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise ValueError('explicit A100 training route required')
    torch.set_num_threads(1);module=original();d=module.protocol.load()
    rows=np.arange(len(d['y']));people=sorted(set(d['subject'].tolist()))
    if len(people)!=10:raise ValueError('all10 deployment source set changed')
    folder=ws/'all10';folder.mkdir(exist_ok=True)
    checkpoint=folder/f'cc30_all10_s{seed}.pt';trace=folder/f'cc30_all10_k2_s{seed}.pt'
    if checkpoint.exists() or trace.exists():raise FileExistsError('never overwrite trained deployment artifact')
    model=module.make(d,'conditional30',seed);started=time.monotonic();model.fit(rows);model.set_k(2.)
    if set(model.fit_subjects)!=set(people) or model.repair_stats['gradient_norm']<=0:
        raise ValueError('deployment source/activity guard failed')
    if (not np.allclose(model.G,expected['G'],atol=1e-9,rtol=1e-9)
            or abs(model.scale-float(expected['scale']))>=1e-9 or float(expected['k'])!=5. or model.k!=5.
            or model.align is not True or not bool(expected['align']) or float(expected['shrink'])!=1e-3):
        raise ValueError('CC30 normalization differs from original package FBk2 input')
    metadata=dict(input_prior_k=float(model.k),latent_bn_k=2.,seed=seed,fit_people=people,held=None,use='all10 deployment only;never a heldfold prediction',
                  source_manifest=sha(ws/'source.sha256'),guard_sha256=sha(ws/'result.json'),
                  active_path_sha256=sha(ws/'outer_active_path.json'),base_zip_sha256=ZIP_SHA,
                  parameter_sha256=module.weights(model.net),fit_seconds=time.monotonic()-started)
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.net.state_dict().items()},G=model.G,
                    scale=model.scale,conditional_coral=30.,**metadata),checkpoint)
    first=d['stream']==d['stream'][0]
    inputs=model._aligned(np.flatnonzero(first),causal=True).astype(np.float32)
    namespace={'__name__':'pinned_v66_input_reference'}
    exec(compile(original_code,'pinned_v66_input_reference','exec'),namespace)
    stream=namespace['InputStream'](expected)
    package_input=np.stack([stream(x.astype(np.float32)) for x in d['X'][first]])
    metadata['input_stream_max_error']=float(np.abs(package_input-inputs).max())
    if metadata['input_stream_max_error']>=1e-4:
        raise ValueError('all10CC30 source input differs from original package stream')
    metadata['export']=export(model,module,trace,inputs)
    metadata['checkpoint_sha256']=sha(checkpoint)
    (folder/f'cc30_all10_s{seed}.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':main(int(sys.argv[1]))
