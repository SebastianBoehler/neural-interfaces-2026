"""Prepare a physical private §315-derived chain for all eight §323 members."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_chain_20261117')
TEMPLATE=Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain')
ORIGINAL=Path('/mnt/lustre/work/gehler/gfh527/ni26_s315_setup_20261117')
ARMS=('P0F','P0C','P1F','P1C','S0F','S0C','S1F','S1C')
FAMILIES=tuple('n323'+a.lower() for a in ARMS)
CONTRASTS=list(dict.fromkeys([('S0C','P0F'),('S0F','P0F'),('S0C','S0F'),('P0C','P0F')]
 +[(f'{t}{z}C',f'{t}{z}F') for t in 'PS' for z in (0,1)]
 +[(f'S{z}{i}',f'P{z}{i}') for z in (0,1) for i in 'FC']
 +[(f'{t}1{i}',f'{t}0{i}') for t in 'PS' for i in 'FC']))


def main():
 if ROOT.exists():raise FileExistsError(ROOT)
 subprocess.run(['cp','-a',str(TEMPLATE),str(ROOT)],check=True)
 rel=ROOT/'studies/17nov/representation_chain';rel.parent.mkdir(parents=True,exist_ok=True)
 if rel.exists():raise FileExistsError(rel)
 shutil.copytree(ORIGINAL/'representation_chain',rel,symlinks=False,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for n in ('stage323.py','stage323.sbatch','retry323.sbatch'):shutil.copy2(HERE/n,rel/n)
 folder=ROOT/'studies/14nov/newmember'
 for n in ('s281_provenance.py','run_s.py','s281_cp.py','score_s_v66p.py','run_s_v66.py'):
  p=folder/n;shutil.copyfile(ORIGINAL/'original'/n,p);p.chmod(0o644)
 for n in ('chainout','newmember','s281_v66_cp','representation_streams','representation_inputs'):
  p=ROOT/'runs'/n
  if p.is_symlink():raise ValueError('refuse to remove linked output directory')
  if p.exists():shutil.rmtree(p)
  if n not in ('representation_streams','representation_inputs'):p.mkdir()
 logs=ROOT/'logs'
 if logs.is_symlink():raise ValueError('refuse linked logs')
 shutil.rmtree(logs);logs.mkdir()
 cfg=rel/'config.py';cfg.chmod(0o644)
 with cfg.open('a') as f:f.write('\nGROUPS["323"] = '+repr(dict(families=FAMILIES,arms=ARMS,contrasts=CONTRASTS))+'\n')
 sys.path.insert(0,str(rel));from patch import main as patch
 patch('323',ROOT)
 release=rel/'release.py';src=release.read_text()
 old="""    if len(queued) + len(tasks) + 1 > 272:
        raise RuntimeError('insufficient submitted capacity for complete treatment group; nothing released')
    slots = min(4, 32 - len(running))
    if slots < 1:
        raise RuntimeError('no safe running capacity for treatment group; nothing released')
"""
 new="""    slots = min(4, 32 - len(running))
    if len(queued) + len(tasks) + 1 > 272 or slots < 1:
        job = subprocess.check_output(['sbatch', '--parsable', '--begin=now+10minutes',
              '--dependency=afterok:' + os.environ['SLURM_JOB_ID'],
              'studies/17nov/representation_chain/retry323.sbatch'], text=True).strip()
        print('capacity deferred safely; retry', job, flush=True)
        return
"""
 if src.count(old)!=1:raise ValueError('release capacity anchor differs')
 release.chmod(0o644);release.write_text(src.replace(old,new,1))
 import config
 records={}
 for name in ('studies','replay','experiments','scripts','lean'):
  for p in sorted((ROOT/name).rglob('*.py')):
   if p.is_symlink():raise ValueError('linked source')
   records[str(p.relative_to(ROOT))]=config.sha(p)
 for p in sorted(rel.glob('*.sbatch')):records[str(p.relative_to(ROOT))]=config.sha(p)
 records['representation_config.json']=config.sha(ROOT/'representation_config.json')
 manifest=ROOT/'representation-source.sha256'
 if manifest.exists():manifest.unlink()
 manifest.write_text(''.join(f'{h}  {n}\n' for n,h in sorted(records.items())))
 for n in records:(ROOT/n).chmod(0o444)
 manifest.chmod(0o444)
 print(json.dumps({'workspace':str(ROOT),'manifest_sha256':config.sha(manifest),'arms':ARMS}),flush=True)

if __name__=='__main__':main()
