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
    if (ws/'released.json').exists():raise FileExistsError('release already recorded')
    lines=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%T'],text=True).splitlines()
    running=sum(s in ('RUNNING','COMPLETING') for s in lines)
    if running+2>32 or len(lines)+5>244:
        retry=submit('--dependency=singleton','--begin=now+5minutes','controller.sbatch')
        with (ws/'capacity_deferrals.jsonl').open('a') as f:f.write(json.dumps(dict(retry=retry,running=running,active=len(lines)))+'\n')
        return
    train=submit('--array=0-5%2','train.sbatch')
    collect=submit('--dependency=afterok:'+train,'collect.sbatch')
    result=dict(proof=proof,train=train,collector=collect,running_at_release=running,active_at_release=len(lines),
                active_workspace=str(ACTIVE),training='all10 deployment only;no score')
    (ws/'released.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
