"""Release fresh standard worker probes only after strict controlled semantic validation."""
import json
import os
import subprocess
import sys
from pathlib import Path

from proofs import manifest
from dispatch import submit
from reference import sha

PHASE=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_phase1')
ARTIFACT={'fir':Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_shared2'),'reuse':Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_inputreuse2')}


def main(variant):
    ws=Path.cwd();manifest(ws);source=manifest(PHASE);folder=ARTIFACT[variant]
    r=json.loads((PHASE/f'phase_{variant}.json').read_text())
    if r['status']!='phase_matched_optimization_guard_passed' or r['source_sha256']!=source or r['complete_sha256']!=sha(folder/'complete.json') or not r['deterministic_algorithms']:
        raise ValueError('controlled proof/source/artifact mismatch')
    for case in r['cases'].values():
        if case['max_output_error']>1e-8 or not case['original_inner_labels_bitwise'] or not case['off45_bitwise'] or not case['reset']['deepcopy_interleaved']:
            raise ValueError('strict paired semantic invariant failed')
    if not r['cpu']['cpu_self_bitwise']:raise ValueError('CPU reset invariant failed')
    late=r['cases']['unrecognized']['activity']
    if late['new_ft']!=20 or late['new_cp']!=10 or late['new_fused']!=125:raise ValueError('late FT/CP inactive')
    lines=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%T'],text=True).splitlines()
    if len(lines)+2>244 or sum(s in ('RUNNING','COMPLETING') for s in lines)+2>32:
        print('capacity deferred',submit('--dependency=singleton','--begin=now+5minutes','phase_release.sbatch',variant));return
    path=ws/f'worker_jobs_{variant}.json'
    if path.exists():raise FileExistsError('variant already released')
    jobs={device:submit(f'phase_worker_{variant}_{device}.sbatch') for device in ('cpu','cuda')}
    path.write_text(json.dumps(dict(variant=variant,proof_sha256=sha(PHASE/f'phase_{variant}.json'),jobs=jobs),indent=2));print(jobs,flush=True)


if __name__=='__main__':main(sys.argv[1])
