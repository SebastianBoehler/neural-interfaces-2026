"""Compare the two prespecified B2_pool contrasts separately for each episode seed."""
import json
from pathlib import Path
from fusion_inputs import FITS, sha

reports={}
for seed in FITS:
    p=Path(f'{seed}_result.json')
    before=Path(f'{seed}_before_labels.json')
    result=json.loads(p.read_text())
    proof=json.loads(before.read_text())
    assert result['seed']==seed==proof['seed'] and result['before']==sha(before)
    assert proof['source']==sha(Path('source.sha256')) and proof['predictions']==sha(Path(f'{seed}.npz'))
    assert proof['labels_opened'] is False
    d=result['data']['JS|B2_pool']['contrasts']
    reports[seed]=dict(result_sha=sha(p),primary={k:d[k] for k in ('J_real_5-zero','J_real_5-J_donor_5','J_real_5-K_5')})
out=dict(seeds=reports,incremental_history=all(s['primary'][k]['paired80'][0]>0 for s in reports.values()
         for k in ('J_real_5-zero','J_real_5-J_donor_5')))
Path('result.json').write_text(json.dumps(out,indent=2))
print(out,flush=True)
