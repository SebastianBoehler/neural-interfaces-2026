"""Separate scheduling-only8-slot completion; never edit the scientific snapshot."""
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2')
HERE=Path(__file__).resolve().parent
SOURCE='528284dc7c07c835534ca94e6637c00df44a38755b62313d5cfbfac25a265ef4'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def verify(ordinal):
    if ordinal not in (19,20) or sha(ROOT/'source.sha256')!=SOURCE:raise ValueError('wrong completion stage/source')
    for line in (ROOT/'source.sha256').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        if sha(ROOT/name)!=expected:raise ValueError('scientific source changed')
    tasks=[i for i in range(160) if i!=80]
    done=set(tasks[:ordinal*8])|{80};records={}
    with np.load(ROOT/'data/proteus_windows.npz') as z:subjects=z['subject']
    for task in sorted(done):
        held=f'P{task%10+1:02d}';seed=task//10%2;arch=('fb','sh')[task//20%2];arm=('S','F','T','C')[task//40]
        path=ROOT/f'output/{arch}_{arm}_s{seed}_{held}.npz'
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata']));rows=np.flatnonzero(np.char.endswith(subjects,'/'+held))
            if (m['held'],m['architecture'],m['arm'],m['seed'],m['source_manifest'])!=(held,arch,arm,seed,SOURCE):raise ValueError('completed fold identity')
            if not np.array_equal(rows,z['index']):raise ValueError('completed fold indices')
            target=str(subjects[rows[0]])
            if target in m['fit_people'] or target in m['inner_fit_people'] or len(m['fit_people'])!=9 or len(m['inner_fit_people'])!=8:raise ValueError('source exclusion')
            if sha(path.with_suffix('.pt'))!=m['checkpoint_sha256']:raise ValueError('checkpoint changed')
        records[path.name]=sha(path)
    (HERE/f'predecessor_{ordinal}.json').write_text(json.dumps(dict(source=SOURCE,folds=records),indent=2))


def main():
    ordinal=int(sys.argv[1]);verify(ordinal)
    source=(ROOT/'dispatch_cpu.py').read_text()
    if source.count('slots = min(4, 32 - running)')!=1 or source.count('> 272')!=2:raise ValueError('scheduler anchors differ')
    source=source.replace('slots = min(4, 32 - running)','slots = min(8, 32 - running)').replace('> 272','> 244')
    module=types.ModuleType('unchanged312_science_new_schedule');module.__file__=str(ROOT/'dispatch_cpu.py')
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    original=subprocess.check_output
    def scheduling(argv,*args,**kwargs):
        command=list(argv)
        if command[0]=='sbatch':
            if 'work_cpu.sbatch' in command:command.insert(1,'--time=00:10:00')
            for old,new in (('dispatch_cpu.sbatch','wave.sbatch'),('collect_cpu.sbatch','collect.sbatch')):
                if old in command:command[command.index(old)]=str(HERE/new)
        return original(command,*args,**kwargs)
    module.subprocess.check_output=scheduling;module.main(ordinal)


if __name__=='__main__':main()
