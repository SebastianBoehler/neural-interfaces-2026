"""Shorten only safely held, still-pending §323 chain tasks."""
import json
import os
import subprocess
from pathlib import Path
HERE=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_dispatch_20261117')
CHAIN=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_chain_20261117')

def run(*args,check=True):return subprocess.run(args,text=True,capture_output=True,check=check)
def submit_retry():
 j=run('sbatch','--parsable','--begin=now+10minutes','schedule_chain.sbatch').stdout.strip()
 with (HERE/'scheduler-retries.log').open('a') as f:f.write(f'retry {j}\n')
 print('chain release pending; retry',j,flush=True)

def main():
 released=CHAIN/'released.json'
 if not released.is_file():submit_retry();return
 r=json.loads(released.read_text());array=str(r['treatments']);score=str(r['scorer'])
 if r['task_count']!=70 or not array.isdecimal() or not score.isdecimal():raise ValueError('wrong released matrix')
 queue=run('squeue','-h','-j',array,'-r','-t','PENDING','-o','%i|%R').stdout.splitlines()
 targets=[]
 for line in queue:
  job,reason=line.split('|',1)
  if reason not in ('JobHeldUser','(JobHeldUser)') and job.startswith(array+'_'):targets.append((job,'00:15:00','ni26-s315-chain'))
 q=run('squeue','-h','-j',score,'-t','PENDING','-o','%i|%R').stdout.splitlines()
 if len(q)==1:
  job,reason=q[0].split('|',1)
  if job==score and reason not in ('JobHeldUser','(JobHeldUser)'):targets.append((job,'00:10:00','ni26-s315-score'))
 changed=[];skipped=[]
 for job,limit,name in targets:
  info=run('scontrol','show','job',job,check=False)
  if info.returncode or 'JobState=PENDING' not in info.stdout or f'JobName={name}' not in info.stdout or f'WorkDir={CHAIN}' not in info.stdout:
   skipped.append(job);continue
  held=run('scontrol','hold',job,check=False)
  if held.returncode:skipped.append(job);continue
  try:
   detail=run('scontrol','show','job',job).stdout
   if 'JobState=PENDING' not in detail or 'Reason=JobHeldUser' not in detail:
    skipped.append(job);continue
   run('scontrol','update',f'JobId={job}',f'TimeLimit={limit}','Partition=cpu-galvani,2080-preemptable-galvani')
   changed.append(job)
  finally:
   run('scontrol','release',job,check=False)
 result={'array':array,'scorer':score,'changed':changed,'skipped':skipped,'limits':'15-minute treatment; 10-minute scorer; no GPU'}
 with (HERE/'scheduler-result.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result),flush=True)

if __name__=='__main__':main()
