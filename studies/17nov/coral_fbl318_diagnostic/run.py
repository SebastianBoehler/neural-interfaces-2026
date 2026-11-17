"""Post-hoc §318 saved-stage analysis through unchanged §300 metric functions."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117')
HERE=ROOT/'studies/14nov/newmember'
RECOVERY=ROOT/'studies/16nov/followups/coral_repair/chain/recovery'
METRICS=Path('/mnt/lustre/work/gehler/gfh527/ni26_s318_diagnostic_20261117')
sys.path[:0]=[str(HERE),str(RECOVERY),str(METRICS)]
from control_source import install
import metrics
spec=importlib.util.spec_from_file_location('score318',HERE/'score_s_v66p.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
install(s,ROOT)
with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:
    y=z['y|all_loso'];cells=z['cell|all_loso'];people=z['person|all_loso']
    sessions=len(np.unique(z['sess|all_loso']))
outputs=s.collect(ROOT,len(y),sessions)
scores={}
for order in s.ORDERS:
    for arm in ('control','SA','SS','FBL2','JA','JS'):
        with np.load(ROOT/f'runs/chainout/s281_v66p_{arm}_{order}.npz') as z:
            for base in s.BASES:
                scores[arm,'V6Fr',order,base]=metrics.probability(z[f'{base}|V6Fr'].copy())
                scores[arm,'CP',order,base]=outputs[arm,order,base]
source=(METRICS/'report.py').read_text()
old="for arm in ('E05', 'E1CS'):"; new="for arm in ('JA', 'JS'):"
if source.count(old)!=1:raise ValueError('unexpected saved-stage report source')
namespace={}
exec(compile(source.replace(old,new,1),str(METRICS/'report.py'),'exec'),namespace)
print('§318 post-hoc selected saved-stage diagnostic: complete source-locked matrix, not independent validation',flush=True)
namespace['report'](scores,y,cells,people,s,metrics,('SA','SS','FBL2','JA','JS'),('V6Fr','CP'))
