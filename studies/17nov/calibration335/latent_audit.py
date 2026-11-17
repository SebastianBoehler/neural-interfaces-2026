"""Identify selected finite geometry in the completed fixed source-held P04 fold."""
import json
import os
from pathlib import Path
import numpy as np
from model import fit,transform
from feature_io import load
from data import metadata,labels,sha
from likelihood import components,ordered

root=Path(os.environ['CAL_ROOT']);d,people,fitrows,queryrows,perms=metadata()
bank,record,manifest=load('exclude_P04');source=bank['fit'];test=np.flatnonzero(bank['people']=='P04');rows=bank['rows']
if 'P04' in bank['people'][source] or set(bank['people'][source])!={f'P{i:02d}' for i in range(5,11)}:raise ValueError('source6 exclusion')
model=fit(bank['features'][source],labels(rows[source],len(people)),bank['people'][source])
z=transform(model,bank['features'][test]);a,r,shape=components(model,z,'mixture');results={}
for rho in (1.,100.,10000.):
    _,states=ordered(.01*(a+rho*r),rows[test],d,perms['within0']);values=[]
    for ev in states:
        p=np.exp(ev).reshape(shape)
        values.append({name:p.sum(axis=tuple(i for i in range(3) if i!=axis)).tolist() for axis,name in enumerate(('g','b','pi'))})
    results[str(rho)]=values
report=dict(scope='same P04source6 deterministicrefit;noheldlabels/newscore;within0indexdiagnostic',source_people=record['fit_people'],
    feature_manifest_sha256=manifest,source_sha256=sha(root/'source.sha256'),geometry_order=['global_zero']+model['people'],results=results)
(root/'latent_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
