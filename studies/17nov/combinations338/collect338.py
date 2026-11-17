"""Complete ten new cells plus immutable336 controls before ANY query metrics."""
import json
import os
from pathlib import Path
import numpy as np
from recipe338 import old,source,CELLS,ORDERS,ROOT336


def main():
    root=Path(os.environ['COMBO_ROOT']);banks={};records={}
    cells=CELLS+[('source7',a) for a in ('A','H','W','B')]
    for scope,arm in cells:
        reuse=scope=='source7' and arm in ('A','H','W','B')
        folder=(ROOT336/'output'/arm) if reuse else (root/'output'/scope/arm)
        m=json.loads((folder/'record.json').read_text());d,fit,query,people,yfit=source(scope)
        if m['device']!='cpu' or m['query_labels_opened'] or m['checkpoint_sha256']!=old.sha(folder/'model.pt'):raise ValueError('cellpin')
        if m['source_rows_sha256']!=old.digest(fit) or m['query_rows_sha256']!=old.digest(query):raise ValueError('rowschanged')
        expected=old.sha((ROOT336 if reuse else root)/'source.sha256')
        if m['source_sha256']!=expected:raise ValueError('sourcechanged')
        prediction_hash=(json.loads((ROOT336/'results.json').read_text())['prediction_sha256'][arm] if reuse else m['prediction_sha256'])
        if old.sha(folder/'predictions.npz')!=prediction_hash:raise ValueError('posteriorhashchanged')
        t=m['trace']
        if len(t['batches'])!=40 or t['hyper']['epochs']!=40 or not t['optimizer_owns_actual_head']:raise ValueError('incompletefit')
        if ('H' in arm)==t['head_is_parametrized'] or ('W' in arm)!=(t['criterion_weights'] is not None):raise ValueError('interactionguard')
        with np.load(folder/'predictions.npz') as z:
            if set(z.files)!={'index'}|set(ORDERS) or not np.array_equal(z['index'],query):raise ValueError('orders/indices')
            p={o:z[o].copy() for o in ORDERS}
        if any(x.shape!=(len(query),3) or not np.isfinite(x).all() or not np.allclose(x.sum(1),1,atol=1e-6) for x in p.values()):raise ValueError('probabilities')
        banks[scope,arm]=p;records[scope,arm]=m
    for scope in ('source7','source6'):
        a=records[scope,'A']['trace']
        for (s,arm),m in records.items():
            if s!=scope:continue
            t=m['trace']
            for k in ('initial_state_sha256','scale','steps','torch_rng_before','torch_rng_after','augment_rng_before','augment_rng_after'):
                if t[k]!=a[k]:raise ValueError('unintendedfactorchange:'+k)
            if (t['batches']==a['batches'])!=('B' not in arm):raise ValueError('batchfactor')
    metric=old.official();scores={}
    for scope in ('source7','source6'):
        d,fit,query,people,yfit=source(scope);y=old.labels(query,len(d['X']));results={}
        for (s,arm),p in banks.items():
            if s!=scope:continue
            result={}
            for person in sorted(set(people[query])):
                selected=people[query]==person;mask=people==person
                by={o:metric.score(y[selected],v[selected].argmax(1),d,mask)['balanced_accuracy_cell_mean'] for o,v in p.items()}
                result[person]=dict(by_order=by,scheme_equal=float(np.mean([np.mean([v for o,v in by.items() if o.startswith(k)]) for k in ('within','full','runs')])))
            results[arm]=dict(people=result,person_mean=float(np.mean([v['scheme_equal'] for v in result.values()])))
        scores[scope]=results
    output=dict(scores=scores,records={s+'/'+a:m for (s,a),m in records.items()},source_sha256=old.sha(root/'source.sha256'),
                scope='fixedcombinationdiagnostic;P02P03exposed;P09sourceheld;noautomaticchain/outer')
    (root/'results.json').write_text(json.dumps(output,indent=2));print('338completeallcells/guardsbeforemetrics',flush=True)


if __name__=='__main__':main()
