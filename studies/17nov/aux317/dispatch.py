"""Three immutable24/24/12cell CPU waves, never reading outer scores."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WS=Path.cwd()


def submit(*args):
    return subprocess.check_output(['sbatch','--parsable',*args],text=True).strip().split(';')[0]


def main(wave):
    if wave not in (0,1,2):
        raise ValueError('expected wave0/1/2')
    tasks=[a*20+s*10+p for a in range(3) for s in range(2)
           for p in range(wave*4,min(10,(wave+1)*4))]
    # Reserve28 account-wide submission slots; no submitted array expansion beyond stated cap.
    for _ in range(50):
        queued=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-o','%i'],text=True).splitlines()
        running=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-t','R','-o','%i'],text=True).splitlines()
        slots=min(4,32-len(running))
        if len(queued)+len(tasks)+3<=272-28 and slots>0:
            break
        time.sleep(60)
    else:
        raise RuntimeError('capacity did not become available; no production released')
    smoke=WS/'smoke_output/C_s0_P01.npz'
    if not smoke.is_file():
        raise ValueError('real no-score smoke required')
    array=submit('--array='+','.join(map(str,tasks))+f'%{slots}','work.sbatch','work')
    if wave<2:
        following=submit(f'--dependency=afterok:{array}','dispatch.sbatch',str(wave+1))
        record=dict(array=array,next_dispatch=following,tasks=tasks,slots=slots)
    else:
        collector=submit(f'--dependency=afterok:{array}','work.sbatch','collect')
        consumer=submit(f'--dependency=afterok:{collector}','consumer.sbatch')
        record=dict(array=array,collector=collector,consumer=consumer,tasks=tasks,slots=slots)
    with (WS/f'wave{wave}_jobs.json').open('x') as f:
        json.dump(record,f,indent=2)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main(int(sys.argv[1]))
