"""Use the locked §326 official helper on complete guarded §327 source predictions."""
import json
from pathlib import Path
import sys
import numpy as np

OLD = Path('/mnt/lustre/work/gehler/gfh527/ni26_s326_conditional_head_20261117')
sys.path.insert(0,str(OLD/'studies/17nov/conditional_head'))
from score import balanced_accuracy, group_scores
from prepare import OLD as DATA_ROOT, QUERY, ORDERS, sha


def main():
    root=Path('runs/s327'); guard=json.loads((root/'guards.json').read_text())
    prep=json.loads((OLD/'runs/s326/prepare.json').read_text())
    if (guard['query_labels_read_for_accuracy'] is not False or
        guard['prediction_sha256']!=sha(root/'predictions.npz') or
        guard['prepare_sha256']!=sha(OLD/'runs/s326/prepare.json') or
        sha(DATA_ROOT/'data/proteus_windows.npz')!=prep['data_sha256']):
        raise ValueError('guard or data changed')
    with np.load(root/'predictions.npz',allow_pickle=False) as pred:
        with np.load(DATA_ROOT/'data/proteus_windows.npz',allow_pickle=False) as z:
            labels=z['y']; people=np.asarray([str(v).rsplit('/',1)[-1] for v in z['subject']])
            session=z['session']; interface=z['interface']
        scores={}
        for person in QUERY:
            rows=pred[f'{person}|rows']; scores[person]={}
            if not np.array_equal(rows,np.flatnonzero(people==person)): raise ValueError('query rows changed')
            groups=np.column_stack((session[rows],interface[rows]))
            for arm in ('zero','mild','moderate'):
                values={}
                for order in ORDERS:
                    labels_pred=pred[f'{person}|{order}|{arm}']
                    values[order]=float(100*group_scores(balanced_accuracy,labels[rows],labels_pred,groups).mean())
                values['scheme_equal']=float(np.mean([np.mean([values[o] for o in ORDERS[:4]]),
                                                     np.mean([values[o] for o in ORDERS[4:6]]),
                                                     np.mean([values[o] for o in ORDERS[6:]])]))
                scores[person][arm]=values
    report={'queries':QUERY,'results':scores,'guard_sha256':sha(root/'guards.json'),
            'scope':'posthoc seven-person source diagnostic, not outer or full-system evidence'}
    with (root/'score.json').open('x') as f: json.dump(report,f,indent=2,sort_keys=True)
    print(json.dumps(report,sort_keys=True),flush=True)


if __name__=='__main__': main()
