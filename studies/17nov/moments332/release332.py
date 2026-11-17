"""Passed canonical P10smoke releases the remaining independent parallelcells."""
import json
import os
import subprocess
from pathlib import Path
import numpy as np
from source332 import sha


def main():
    root=Path(os.environ['MOMENT_ROOT']);path=root/'output/P10_s0.npz'
    with np.load(path) as z:meta=json.loads(str(z['metadata']))
    if (meta['held'],meta['seed'],meta['held_labels_opened'],meta['source_sha256'])!=('P10',0,False,sha(root/'source.sha256')):raise ValueError('fullsmokeidentity')
    if len([k for k in z.files if '|' in k])!=80:raise ValueError('incompletesmoke')
    active=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-o','%i'],text=True).splitlines()
    running=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-t','RUNNING','-o','%i'],text=True).splitlines()
    if len(active)+20>244 or len(running)+4>32:raise RuntimeError('capacityunavailable;noarrayreleased')
    # Task18 isP10seed0, already canonical; never duplicate it.
    array=subprocess.check_output(['sbatch','--parsable','--array=0-17,19%4','produce.sbatch'],cwd=root,text=True).strip().split(';')[0]
    collector=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{array}','collect.sbatch'],cwd=root,text=True).strip().split(';')[0]
    (root/'released.json').write_text(json.dumps(dict(array=array,collector=collector,smoke_seconds=meta['seconds'],smoke_sha256=sha(path))))
    print(f'332remaining19parallelcells{array};completecollector{collector}',flush=True)


if __name__=='__main__':main()
