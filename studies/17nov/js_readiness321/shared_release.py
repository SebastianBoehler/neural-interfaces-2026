"""Verify real same-weight source parity, then release independent worker probes."""
import json
import os
import subprocess
from pathlib import Path

from dispatch import submit
from proofs import manifest
from reference import sha

SHARED=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_shared2')


def main(retry_script='shared_release.sbatch'):
    if retry_script not in ('shared_release.sbatch','input_release.sbatch'):raise ValueError('unregistered release script')
    ws=Path.cwd();manifest(ws);source=manifest(SHARED)
    r=json.loads((SHARED/'optimization_guard.json').read_text())
    if r['status']!='same_weight_sharedFIR_raw_active_reset_guards_passed' or r['source_sha256']!=source or r['complete_sha256']!=sha(SHARED/'complete.json'):
        raise ValueError('optimized source/proof/artifact mismatch')
    if len(r['raw'])!=8:raise ValueError('eight raw orders required')
    for value in r['raw'].values():
        if value['seeds']!=6 or value['full_trace_max_error']>1e-8 or value['historical_max_error']>=1e-4:raise ValueError('raw guard failed')
    for case in r['cases'].values():
        if case['max_output_error']>1e-8 or not all(case[k] for k in ('inner_bitwise','off45_bitwise','ft_labels_bitwise','reset_deepcopy_prefix')):
            raise ValueError('optimized parity/state guard failed')
    late=r['cases']['unrecognized']
    if late['new_cp_fused']<=0 or late['new_ft_refits']<=0 or late['old_cp_fused']!=late['new_cp_fused']:raise ValueError('late CP/FT inactive')
    lines=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%T'],text=True).splitlines()
    if len(lines)+2>244 or sum(s in ('RUNNING','COMPLETING') for s in lines)+2>32:
        print('capacity deferred',submit('--dependency=singleton','--begin=now+5minutes',retry_script));return
    if (ws/'worker_jobs.json').exists():raise FileExistsError('already released')
    jobs={device:submit(f'shared_worker_{device}.sbatch') for device in ('cpu','cuda')}
    (ws/'worker_jobs.json').write_text(json.dumps(dict(guard_sha256=sha(SHARED/'optimization_guard.json'),jobs=jobs),indent=2))


if __name__=='__main__':main()
