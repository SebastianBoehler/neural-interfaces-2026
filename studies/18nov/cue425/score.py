"""Paired changes in official cell-mean accuracy across genuine Scherer crop times."""
from pathlib import Path
import json
import numpy as np

files = [Path('../ni26_cross411b_20261118/result.json'),Path('offset30/result.json'),Path('offset60/result.json')]
records = [json.loads(p.read_text()) for p in files]
assert records[0]['people'] == records[1]['people'] == records[2]['people']
assert records[0]['arms'] == records[1]['arms'] == records[2]['arms']
vals = np.array([r['person_diff'] for r in records])
np.testing.assert_allclose(vals.mean(1), [r['diff'] for r in records],atol=1e-10,rtol=0)
sample = np.random.default_rng(425).integers(9,size=(10000,9))
def paired(d):
    return dict(mean=float(d.mean()),ci80=np.quantile(d[sample].mean(1),[.1,.9]).tolist(),people=d.tolist())
result = dict(package_accuracy=[r['acc'] for r in records],delta=vals.mean(1).tolist(),
              offset_vs_original=paired(vals[1]-vals[0]),later_vs_offset=paired(vals[2]-vals[1]),
              first_minus_later_shift=paired(2*vals[1]-vals[0]-vals[2]),
              crop_max_error_uv=max(json.loads(Path(f'{sid}.json').read_text())['max_error_uv'] for sid in 'ACDEFGHJL'))
Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
