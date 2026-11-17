"""After complete whole collector: private proven-consumer bootstrap, no evaluator changes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WS=Path.cwd()
CHAIN=Path('/mnt/lustre/work/gehler/gfh527/ni26_aux317_20261117_chain1')


def submit(*args):
    return subprocess.check_output(['sbatch','--parsable',*args],text=True).strip().split(';')[0]


def main():
    report=json.loads((WS/'whole_results.json').read_text())
    if report['status']!='complete60fold_guarded' or len(report['hashes'])!=60:
        raise ValueError('successful complete owning collector required')
    subprocess.run([sys.executable,'consumer/prepare.py','317',str(CHAIN)],check=True)
    env=dict(os.environ,NI26_WS=str(CHAIN))
    subprocess.run([sys.executable,str(CHAIN/'studies/17nov/representation_chain/stage.py')],env=env,check=True)
    os.chdir(CHAIN)
    for _ in range(40):
        queued=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-o','%i'],text=True).splitlines()
        running=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-t','R','-o','%i'],text=True).splitlines()
        if len(queued)+11<=272-28 and len(running)<32:
            break
        time.sleep(60)
    else:
        raise RuntimeError('consumer capacity unavailable;prepared chain preserved')
    prefix='studies/17nov/representation_chain/'
    replay=submit(prefix+'replay.sbatch')
    control=submit('--array=0-7%2',f'--dependency=afterok:{replay}',prefix+'job.sbatch','control')
    smoke=submit(f'--dependency=afterok:{control}',prefix+'job.sbatch','smoke')
    (CHAIN/'bootstrap_jobs.json').write_text(json.dumps(dict(replay=replay,controls=control,smoke=smoke),indent=2))
    print('317complete matrix→private stage→replay→8controls→zero/one smoke→guardedfullrelease',replay,control,smoke,flush=True)


if __name__=='__main__':
    main()
