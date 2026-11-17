"""Collect all six deployment seeds and stage a private JS-all copy, without scoring."""
import json
import shutil
from pathlib import Path

import numpy as np

from proofs import ACTIVE,manifest,read,verify
from reference import sha


def main():
    ws=Path.cwd();source=manifest(ws);proof=verify();rows=[]
    target=ws/'prepared_jsall';target.mkdir(exist_ok=False)
    for seed in range(6):
        folder=ws/f'seed{seed}/all10';row=read(folder/f'cc30_all10_s{seed}.json')
        if row['seed']!=seed or row['held'] is not None or len(row['fit_people'])!=10 or row['source_manifest']!=source:
            raise ValueError('all10 seed/source/population differs')
        if row['active_path_sha256']!=proof['active_proof'] or row['input_stream_max_error']>=1e-4:
            raise ValueError('all10 active/input proof differs')
        cp=folder/f'cc30_all10_s{seed}.pt';trace=folder/f'cc30_all10_k2_s{seed}.pt'
        e=row['export']
        if sha(cp)!=row['checkpoint_sha256'] or sha(trace)!=e['trace_sha256']:
            raise ValueError('all10 artifact hash differs')
        if not e['saved_reset'] or not e['deepcopy_interleaved'] or max(e['cpu_probability_errors'].values())>=1e-4 or e['cuda_probability_error']>=1e-3:
            raise ValueError('all10 export guard failed')
        shutil.copy2(trace,target/trace.name);rows.append(row)
    for path in (ACTIVE/'draft_jsall').iterdir():
        if path.name not in ('weights.npz',) and not path.name.startswith('cc30_P01_'):shutil.copy2(path,target/path.name)
    with np.load(ACTIVE/'draft_jsall/weights.npz',allow_pickle=False) as z:weights={k:z[k].copy(order='K') for k in z.files}
    weights['js_cc_files']=np.array([f'cc30_all10_k2_s{s}.pt' for s in range(6)])
    for key in ('G','scale','align','k','shrink'):weights['js_'+key]=weights['net35_'+key].copy(order='K')
    np.savez(target/'weights.npz',**weights)
    report=dict(status='six_seed_all10_complete_private_copy_prepared',source_sha256=source,proof=proof,seeds=rows,
                files={p.name:sha(p) for p in target.iterdir()},policy='JS-all;outer continues after recognition',
                limits='deployment training only;no new dev score;worker/returning readiness not established')
    (ws/'complete.json').write_text(json.dumps(report,indent=2));print(report['status'],flush=True)


if __name__=='__main__':main()
