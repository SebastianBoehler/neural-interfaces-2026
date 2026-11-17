"""Measured budget authority only; untouched native recipe and own originalcollector."""
import json
import os
import sys
from pathlib import Path
import hashlib

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_reference_bridge336_v1_20261117')
SOURCE='7fa3da6d63270d9911de7d302c1f84e954ab02f0797818a6ae08ebca3b857383'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if os.environ['BRIDGE_ROOT']!=str(ROOT) or sha(ROOT/'source.sha256')!=SOURCE:raise ValueError('native336sourcechanged')
    for line in (ROOT/'source.sha256').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        if sha(ROOT/name)!=expected:raise ValueError('scientificAPIchanged')
    guard=json.loads((ROOT/'guard.json').read_text())
    if not guard['passed'] or not guard['native_observer_exact'] or guard['projected_full_arm_seconds']>600:raise ValueError('explicitmeasured600sbudgetnotmet')
    collector=sys.argv[1:] == ['collect']
    filename='collect336.py' if collector else 'produce336.py';source=(ROOT/filename).read_text()
    if collector:
        records=[json.loads((ROOT/'output'/a/'record.json').read_text()) for a in ('A','H','U','W','B')]
        if len({r['trace']['steps'] for r in records})!=1:raise ValueError('onefactoroptimizerstepsdiffer')
    else:
        anchor="not guard['budget_passed']"
        if source.count(anchor)!=1:raise ValueError('budgetanchorchanged')
        source=source.replace(anchor,"guard['projected_full_arm_seconds']>600",1)
    sys.path.insert(0,str(ROOT))
    exec(compile(source,str(ROOT/filename),'exec'),{'__name__':'__main__','__file__':str(ROOT/filename)})


if __name__=='__main__':main()
