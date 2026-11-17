import json
from pathlib import Path
import numpy as np
from held import compare

root = Path('/mnt/lustre/work/gehler/gfh527')
candidate = np.load(root/'ni26_solver343_span_v1_20261117/full_fit.npz')['theta']
reference = np.load(root/'ni26_solver343_baseline_v1_20261117/cold_sklearn_1.npz')['theta']
result = compare('P01', candidate, reference)
Path('held_p01.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)

if not all(r['passed'] for r in result.values()):
    raise ValueError('strict held prediction qualification failed')
