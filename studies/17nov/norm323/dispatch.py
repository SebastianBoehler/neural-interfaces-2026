"""Capacity-checked complete producer/chain DAG after the canonical P10 smoke."""
import json
import os
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROD=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_normsplit_20261117')
CHAIN=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_chain_20261117')
PY='/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python'

def run(*args,cwd=None,env=None):return subprocess.check_output(args,cwd=cwd,env=env,text=True).strip()
def submit(*args,cwd):return run('sbatch','--parsable',*args,cwd=cwd).split(';')[0]
def live(job):return bool(run('squeue','-h','-j',job,'-o','%i'))
def complete(job):
 lines=run('sacct','-X','-n','-P','-j',job,'-o','JobID,State,ExitCode').splitlines()
 return bool(lines) and all(line.split('|')[1:3]==['COMPLETED','0:0'] for line in lines)

def main():
 for root,name,digest in ((PROD,'source323.sha256','88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5'),
                          (CHAIN,'representation-source.sha256','baf212bddfc79ed4d8bce9a9600c5c3db56d039d72c08f8fcfc92997dafbd55c')):
  import hashlib
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('source lock identity changed')
  subprocess.run(['sha256sum','-c',name],cwd=root,stdout=subprocess.DEVNULL,check=True)
 env=dict(os.environ,PYTHONPATH=f'{PROD}:{PROD}/experiments:{PROD}/studies/14nov/newmember',NI26_DEVICE='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
 print(run(PY,'-B','verify_fold.py','P10','0',cwd=PROD,env=env),flush=True)
 controls=json.loads((CHAIN/'controls323.json').read_text());dependencies=[]
 for key in ('replay','control'):
  job=controls[key]
  if live(job):dependencies.append(job)
  else:
   if not complete(job):raise ValueError(f'{key} job not completed successfully')
   if key=='replay' and 'OK' not in (CHAIN/f'slurm-{job}.out').read_text().splitlines():raise ValueError('replay missing OK')
   if key=='control':print(run(PY,'-B',str(HERE/'control_guard.py'),cwd=CHAIN,env=env),flush=True)
 active=run('squeue','-h','-u',os.environ['USER'],'-r','-o','%i').splitlines()
 running=run('squeue','-h','-u',os.environ['USER'],'-r','-t','RUNNING','-o','%i').splitlines()
 if len(active)+23>272 or len(running)+4>32:
  retry=submit('--begin=now+10minutes','dispatch.sbatch',cwd=HERE)
  print(f'capacity {len(active)}/{len(running)}; retry {retry}',flush=True);return
 intent=HERE/'dispatch.intent'
 with intent.open('x') as f:f.write(f'dispatch {os.environ["SLURM_JOB_ID"]}\n')
 def record(name,value):
  with intent.open('a') as f:f.write(f'{name} {value}\n')
 producer=submit('--array=0-8,10-19%4','produce.sbatch',cwd=PROD);record('producer',producer)
 collector=submit(f'--dependency=afterok:{producer}','collect.sbatch',cwd=PROD);record('collector',collector)
 stage=submit('--dependency=afterok:'+':'.join([collector,*dependencies]),'studies/17nov/representation_chain/stage323.sbatch',cwd=CHAIN);record('stage',stage)
 smoke=submit(f'--dependency=afterok:{stage}','studies/17nov/representation_chain/job.sbatch','smoke',cwd=CHAIN);record('chain_smoke',smoke)
 print('guarded 19-fold producer → complete whole collector → source-locked 8-arm chain stage → smoke',flush=True)

if __name__=='__main__':main()
