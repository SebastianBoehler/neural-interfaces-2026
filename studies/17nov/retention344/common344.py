"""Immutable 337 source and full target instrument; CPU6 only."""
import sys
from pathlib import Path
PARENT=Path('/mnt/lustre/work/gehler/gfh527/ni26_pretrain337_pilot_v3_20261117')
sys.path.insert(0,str(PARENT))
from common import *
if sha(PARENT/'pretext.py')!='bf7cda25194120cdb2a5eff87a81e11c60e327eb213693316421a8e27ccd271b':raise ValueError('337pretext pin')
TEACHER=Path('/mnt/lustre/work/gehler/gfh527/ni26_pretrain337_targets_v1_20261117')

def teacher_load():
    import json
    card=json.loads((TEACHER/'teacher.json').read_text())
    if sha(TEACHER/'teacher.npz')!=card['teacher_sha256']:raise ValueError('teacher hash')
    with np.load(TEACHER/'teacher.npz',allow_pickle=False) as z:values={k:z[k] for k in z.files}
    return values,card
