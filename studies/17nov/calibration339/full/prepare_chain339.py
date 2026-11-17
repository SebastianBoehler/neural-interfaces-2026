"""Physical private existing323/315 consumer/scorer, six fixed339 members."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_calibration339_chain_v1_20261117')
TEMPLATE=Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain')
ORIGINAL=Path('/mnt/lustre/work/gehler/gfh527/ni26_s315_setup_20261117')
ARMS=('RF','RB','RV','CF','CB','CV');FAMILIES=tuple('n339'+a.lower() for a in ARMS)
CONTRASTS=[('CF','RF'),('CB','RB'),('CV','RV'),('RV','RB'),('RV','RF'),('CV','CB'),('CB','CF')]
REL=Path('studies/17nov/representation_chain')


def main():
    if ROOT.exists():raise FileExistsError(ROOT)
    subprocess.run(['cp','-a',str(TEMPLATE),str(ROOT)],check=True)
    rel=ROOT/REL;rel.parent.mkdir(parents=True,exist_ok=True)
    if rel.exists():raise FileExistsError(rel)
    shutil.copytree(ORIGINAL/'representation_chain',rel,symlinks=False,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    folder=ROOT/'studies/14nov/newmember'
    for n in ('s281_provenance.py','run_s.py','s281_cp.py','score_s_v66p.py','run_s_v66.py'):
        p=folder/n;shutil.copyfile(ORIGINAL/'original'/n,p);p.chmod(0o644)
    for n in ('chainout','newmember','s281_v66_cp','representation_streams','representation_inputs'):
        p=ROOT/'runs'/n
        if p.is_symlink():raise ValueError('refuselinkedoutput')
        if p.exists():shutil.rmtree(p)
        if n not in ('representation_streams','representation_inputs'):p.mkdir()
    shutil.rmtree(ROOT/'logs');(ROOT/'logs').mkdir()
    cfg=rel/'config.py';cfg.chmod(0o644)
    with cfg.open('a') as f:f.write('\nGROUPS["339"] = '+repr(dict(families=FAMILIES,arms=ARMS,contrasts=CONTRASTS))+'\n')
    sys.path.insert(0,str(rel));import config
    from patch import main as patch
    patch('339',ROOT)
    # Honest operation provenance: calibrated members applyT AFTER the two-seed mean.
    path=folder/'run_s.py';text=path.read_text()
    old='assert metadata["seeds"] == [0, 1] and metadata["operation"] == "arithmetic_probability_mean"'
    operations={f:'arithmetic_probability_mean' if a[0]=='R' else 'temperature_after_arithmetic_probability_mean' for a,f in zip(ARMS,FAMILIES)}
    if text.count(old)!=1:raise ValueError('actualstreammetadataanchorchanged')
    text=text.replace(old,'assert metadata["seeds"] == [0, 1] and metadata["operation"] == '+repr(operations)+'[fam]')
    compile(text,str(path),'exec');path.write_text(text)
    for name in ('stage339.py','release339.py','final339.py','cell339.py','job339.sbatch','stage339.sbatch','score339.sbatch'):
        shutil.copy2(HERE/name,rel/name)
    records={}
    for name in ('studies','replay','experiments','scripts','lean'):
        for p in sorted((ROOT/name).rglob('*.py')):
            if p.is_symlink():raise ValueError('linkedsource')
            records[str(p.relative_to(ROOT))]=config.sha(p)
    for p in rel.glob('*.sbatch'):records[str(p.relative_to(ROOT))]=config.sha(p)
    records['representation_config.json']=config.sha(ROOT/'representation_config.json')
    manifest=ROOT/'representation-source.sha256';manifest.unlink(missing_ok=True)
    manifest.write_text(''.join(f'{h}  {n}\n' for n,h in sorted(records.items())))
    for n in records:(ROOT/n).chmod(0o444)
    manifest.chmod(0o444)
    print(json.dumps(dict(root=str(ROOT),source_sha256=config.sha(manifest))),flush=True)


if __name__=='__main__':main()
