"""Execute one preregistered chain cell and its six CP children."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECIPES = {'zero318':('fb_cc30','none','1'), 'SA':('fb_cc30','none','0'),
           'SS':('none','fb_cc30','0'), 'FBL2':('none','none','0'),
           'JA':('fb_cc30','none','0'), 'JS':('none','fb_cc30','0')}
ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')
BASES = ('B2_pool','B2_s345','B2_s012','B2_s0','B2_s1','B2_s2')


def main(arm, order):
    if arm not in RECIPES or order not in ORDERS or Path.cwd().resolve() != ROOT:
        raise ValueError('invalid §318 cell')
    add, swap, zero = RECIPES[arm]
    env = dict(os.environ, NM_NAME=arm, ADD=add, SWAP=swap, ZERO=zero, NM_SEEDS='0,1,2,3,4,5')
    subprocess.run([sys.executable,'-B','-I',str(ROOT/'studies/14nov/newmember/run_s_v66.py'),order],env=env,check=True)
    path=ROOT/'studies/15nov/s281/galv_v66_job.py'
    spec=importlib.util.spec_from_file_location('children318',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    logdir=ROOT/'logs/cp318';logdir.mkdir(parents=True,exist_ok=True)
    cp=ROOT/'studies/14nov/newmember/s281_cp.py'
    commands=[(f'{arm}_{order}_{b}',[sys.executable,'-B','-I',str(cp),arm,order,b]) for b in BASES]
    mod.run_children(commands,logdir,env)

if __name__=='__main__': main(*sys.argv[1:])
