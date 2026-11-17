"""Release remaining five only after canonical all10 seed0 normalization/export smoke."""
import json
import os
import subprocess
from pathlib import Path

from proofs import manifest,verify,read
from reference import sha
from dispatch import submit


def main():
    ws=Path.cwd();source=manifest(ws);proof=verify()
    if (ws/'released.json').exists():raise FileExistsError('already released')
    row=read(ws/'seed0/all10/cc30_all10_s0.json');e=row['export']
    if row['seed']!=0 or row['source_manifest']!=source or row['input_stream_max_error']>=1e-4:
        raise ValueError('canonical all10 normalization/source smoke failed')
    if sha(ws/'seed0/all10/cc30_all10_s0.pt')!=row['checkpoint_sha256'] or sha(ws/'seed0/all10/cc30_all10_k2_s0.pt')!=e['trace_sha256']:
        raise ValueError('canonical all10 artifact changed')
    if not e['saved_reset'] or not e['deepcopy_interleaved'] or max(e['cpu_probability_errors'].values())>=1e-4 or e['cuda_probability_error']>=1e-3:
        raise ValueError('canonical all10 export/state smoke failed')
    lines=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%T'],text=True).splitlines()
    if len(lines)+7>244 or sum(s in ('RUNNING','COMPLETING') for s in lines)+2>32:
        print('capacity deferred',submit('--dependency=singleton','--begin=now+5minutes','remaining.sbatch'));return
    train=submit('--array=1-5%2','train.sbatch');collector=submit('--dependency=afterok:'+train,'collect.sbatch')
    result=dict(proof=proof,smoke=read(ws/'smoke_jobs.json'),train=train,collector=collector,
                canonical_seed0_sha256=row['checkpoint_sha256'],training='all10 deployment only;no score')
    (ws/'released.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
