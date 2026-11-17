"""Explicit measured budget-only authority; scientific source remains immutable."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_oscillatory331_v2_20261117')
SOURCE='b0fec93e6c3c6e3f66cc1d967e2c3b7ba1c87e01ca8308d3d56a6f5e0fe8fde9'
GUARD='04d01e3469b9bc6a19f1bccbd93bd5d912743ecbba7b997dbb519e8698c27125'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if os.environ['OSC_ROOT']!=str(ROOT) or sha(ROOT/'source.sha256')!=SOURCE or sha(ROOT/'guard.json')!=GUARD:
        raise ValueError('budget authority source/guard changed')
    guard=json.loads((ROOT/'guard.json').read_text())
    if not guard['passed'] or guard['budget_passed'] or guard['projected_full_pilot_seconds']>3600:
        raise ValueError('explicit60minute projectedbudget authority mismatch')
    if sha(ROOT/'scaler.npz')!=guard['scaler_sha256']:raise ValueError('guarded scalerchanged')
    for line in (ROOT/'source.sha256').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        if sha(ROOT/name)!=expected:raise ValueError('scientific sourcechanged')
    source=(ROOT/'pilot331.py').read_text();anchor="not guard['budget_passed']"
    if source.count(anchor)!=1:raise ValueError('budget predicate anchor differs')
    # Only this engineering gate changes. Models/loss/scalers/training/metric remain exact.
    source=source.replace(anchor,"guard['projected_full_pilot_seconds']>3600",1)
    authority=Path(__file__).resolve().parent
    record=dict(scientific_source_sha256=SOURCE,guard_sha256=GUARD,
        budget_wrapper_sha256=sha(authority/'budget.sha256'),projected_seconds=guard['projected_full_pilot_seconds'],
        admitted_projection_seconds=3600,requested_allocation_minutes=75,scientific_recipe_unchanged=True)
    with (ROOT/'budget_authority.json').open('x') as out:json.dump(record,out,indent=2)
    sys.path.insert(0,str(ROOT))
    exec(compile(source,str(ROOT/'pilot331.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'pilot331.py')})


if __name__=='__main__':main()
