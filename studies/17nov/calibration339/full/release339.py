"""Original chain guards plus measured CPU6 budget before remaining full matrix."""
import importlib
import json
import os
import subprocess
import time
from pathlib import Path
from release import verify_smoke
from config import BASES,ORDERS
ROOT=Path(os.environ['NI26_WS'])


def main():
    cfg=verify_smoke();s=importlib.import_module('score_s_v66p');cache={}
    for b in BASES:s.load_output(ROOT,'RV','within0',b,2655,cache)
    times=[json.loads(p.read_text())['seconds'] for p in ROOT.glob('timing_*.json')]
    projection=1.2*max(times)*64*6/3600+1.5
    if projection>32:raise ValueError(f'CPUfullmatrixcoststop:{projection}')
    tasks=[(a,o) for a in ('zero_repr',*cfg['arms']) for o in ORDERS if (a,o) not in [('zero_repr','within0'),('RF','within0'),('RV','within0')]]
    active=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-o','%i'],text=True).splitlines()
    running=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-t','RUNNING','-o','%i'],text=True).splitlines()
    slots=min(8,32-len(running))
    if slots<1 or len(active)+len(tasks)+1>244:raise ValueError('reservedcapacity')
    (ROOT/'remaining_cells.json').write_text(json.dumps(tasks))
    def submit(*args):return subprocess.check_output(['sbatch','--parsable',*args],text=True).strip().split(';')[0]
    array=submit(f'--array=0-{len(tasks)-1}%{slots}','studies/17nov/representation_chain/job339.sbatch','remaining')
    score=submit(f'--dependency=afterok:{array}','studies/17nov/representation_chain/score339.sbatch')
    (ROOT/'released.json').write_text(json.dumps(dict(treatments=array,scorer=score,task_count=len(tasks),slots=slots,projected_CPUcorehours=projection)))
    print(array,score,projection,flush=True)


if __name__=='__main__':main()
