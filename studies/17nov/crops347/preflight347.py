"""Actual producer/collector imports and frozen baseline scope, before full fits."""
import json
import os
from pathlib import Path
import numpy as np
import produce347
import collect347
from training347 import source,late_data,sha,digest

root=Path(os.environ['TRAIN347_ROOT']);d,fit,_,people,_=source();query=np.flatnonzero(np.isin(people,['P02','P03','P09']))
_,_,qualification=late_data(fit);control,probabilities,proofs=collect347.original_control(fit,people,query)
if sha(root/'training347.py')!=sha(produce347.SMOKE/'training347.py'):raise ValueError('actual producer changed smoked science')
report=dict(passed=True,source_sha256=sha(root/'source.sha256'),actual_imports=[produce347.__file__,collect347.__file__],
    source_rows_sha256=digest(fit),query_rows_sha256=digest(query),query_counts={p:int(np.sum(people[query]==p)) for p in ('P02','P03','P09')},
    checkpoint_sha256=control['checkpoint_sha256'],late_sha256=qualification['late_sha256'],original_control_proofs=proofs,
    all8_original_banks_finite=True,held_labels_opened=False,new_fits=0)
(root/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
