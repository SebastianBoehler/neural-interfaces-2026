"""Capacity-checked afterok release; never reads accuracy or weakens proof guards."""
import json
import os
import subprocess
from pathlib import Path

from proofs import ACTIVE,manifest,verify


def submit(*args):
    return subprocess.check_output(['sbatch','--parsable',*args],text=True).strip().split(';')[0]


def main():
    ws=Path.cwd();manifest(ws);proof=verify()
    if (ws/'released.json').exists() or (ws/'smoke_jobs.json').exists():raise FileExistsError('release already recorded')
    lines=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%T'],text=True).splitlines()
    running=sum(s in ('RUNNING','COMPLETING') for s in lines)
    if running+2>32 or len(lines)+5>244:
        retry=submit('--dependency=singleton','--begin=now+5minutes','controller.sbatch')
        with (ws/'capacity_deferrals.jsonl').open('a') as f:f.write(json.dumps(dict(retry=retry,running=running,active=len(lines)))+'\n')
        return
    smoke=submit('--array=0','train.sbatch')
    follow=submit('--dependency=afterok:'+smoke,'remaining.sbatch')
    (ws/'smoke_jobs.json').write_text(json.dumps(dict(proof=proof,smoke=smoke,remaining_controller=follow),indent=2))
    print('one all10seed smoke released',smoke,follow,flush=True)



if __name__=='__main__':main()
