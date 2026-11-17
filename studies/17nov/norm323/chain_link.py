"""Attach registered interaction report to dynamically released complete scorer."""
import json
import os
import subprocess
from pathlib import Path
HERE=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_interactions_20261117')
CHAIN=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_chain_20261117')

def run(*args):return subprocess.check_output(args,text=True,cwd=HERE).strip()
def submit(*args):return run('sbatch','--parsable',*args).split(';')[0]

def main():
 released=CHAIN/'released.json'
 if not released.is_file():
  retry=submit('--begin=now+10minutes','chain_link.sbatch')
  with (HERE/'chain-link-retries.log').open('a') as f:f.write(f'retry {retry}\n')
  print('chain release still capacity-deferred; retry',retry,flush=True);return
 config=json.loads(released.read_text())
 if config['task_count']!=70:raise ValueError('wrong complete chain matrix size')
 score=str(config['scorer'])
 if not score.isdecimal():raise ValueError('bad score id')
 live=run('squeue','-h','-j',score,'-o','%i')
 deps=[]
 if live:deps=[f'--dependency=afterok:{score}']
 else:
  state=run('sacct','-X','-n','-P','-j',score,'-o','State,ExitCode').splitlines()
  if state!=['COMPLETED|0:0'] or not (CHAIN/'results.txt').is_file():raise ValueError('complete chain scorer unavailable')
 active=run('squeue','-h','-u',os.environ['USER'],'-r','-o','%i').splitlines()
 if len(active)+1>272:
  retry=submit('--begin=now+10minutes','chain_link.sbatch')
  with (HERE/'chain-link-retries.log').open('a') as f:f.write(f'capacity retry {retry}\n')
  return
 intent=HERE/'chain-link.intent'
 with intent.open('x') as f:f.write(f'link {os.environ["SLURM_JOB_ID"]}\n')
 job=submit(*deps,'chain_interactions.sbatch')
 with intent.open('a') as f:f.write(f'score {score}\ninteraction {job}\n')
 print(f'complete scorer {score} → chain interaction {job}',flush=True)

if __name__=='__main__':main()
