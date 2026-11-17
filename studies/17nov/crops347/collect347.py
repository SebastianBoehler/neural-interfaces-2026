"""Both matched fits/all three people/all orders before any347held labels."""
import json
import os
from pathlib import Path
import numpy as np
from training347 import source,sha,digest,CONTROL
from common346 import labels,official,ORDERS

OLD=Path('/mnt/lustre/work/gehler/gfh527/ni26_direct346_v1_20261117')
EXT=Path('/mnt/lustre/work/gehler/gfh527/ni26_direct346_extension_v1_20261117')


def original_control(fit,people,query):
    control=json.loads((CONTROL/'record.json').read_text());proofs={};bb={o:np.full((len(query),3),np.nan) for o in ORDERS}
    if control['query_labels_opened'] or control['checkpoint_sha256']!='eadd5c23c9f7cac239bf78a99993d8d3e47392d24d166e0ede6fce5b0e6ac9f8' or sha(CONTROL/'model.pt')!=control['checkpoint_sha256'] or control['source_rows_sha256']!=digest(fit):raise ValueError('actualA6 frozen source control changed')
    for path in (OLD,EXT):
        proof=json.loads((path/'before_labels.json').read_text());key='prediction_sha256' if path==OLD else 'predictions_sha256'
        if proof['held_labels_opened'] or sha(path/'predictions.npz')!=proof[key]:raise ValueError('originalA no-label control bank changed')
        scope=json.loads((path/'heads_before_query.json').read_text()) if path==OLD else proof
        if scope['checkpoint_sha256']!=control['checkpoint_sha256'] or scope['source_rows_sha256']!=digest(fit):raise ValueError('A6 bank/source/body scope differs')
        if path==OLD and max(proof['identity'].values())!=0:raise ValueError('originalA6 historical identity not exact')
        if path==EXT and proof['new_fits']!=0:raise ValueError('extension substituted a new fitted control')
        proofs[str(path)]=sha(path/'before_labels.json')
        with np.load(path/'predictions.npz') as z:
            for person in (('P09',) if path==OLD else ('P02','P03')):
                rr=z['index'] if path==OLD else z[person+'_index'];local=np.searchsorted(query,rr)
                if not np.array_equal(query[local],rr) or not np.all(people[rr]==person):raise ValueError('originalA person/row mapping')
                for order in ORDERS:bb[order][local]=z[('A_' if path==OLD else person+'_A_')+order]
    if any(not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6) for p in bb.values()):raise ValueError('incomplete originalA probabilities')
    return control,bb,proofs


def main():
    root=Path(os.environ['TRAIN347_ROOT']);d,fit,_,people,_=source();query=np.flatnonzero(np.isin(people,['P02','P03','P09']))
    control,original,proofs=original_control(fit,people,query);records={};banks={}
    for arm in ('B','C'):
        folder=root/'output'/arm;r=json.loads((folder/'record.json').read_text());t=r['trace']
        if r['held_labels_opened'] or r['source_sha256']!=sha(root/'source.sha256') or r['source_rows_sha256']!=digest(fit) or r['query_rows_sha256']!=digest(query):raise ValueError('producer source/exclusion/provenance failed')
        if r['checkpoint_sha256']!=sha(folder/'model.pt') or r['prediction_sha256']!=sha(folder/'predictions.npz') or r['source_people']!=['P04','P05','P06','P07','P08','P10']:raise ValueError('producer source/file identity changed')
        if t['steps']!=2480 or t['epochs']!=40 or t['base_batches']!=control['trace']['batches'] or len(t['extra_batches'])!=40 or t['initial_state_sha256']!=control['trace']['initial_state_sha256']:raise ValueError('exact source/step/nativeinitializer/40epoch trace failed')
        with np.load(folder/'predictions.npz') as z:
            if set(z.files)!={'index'}|set(ORDERS) or not np.array_equal(z['index'],query):raise ValueError('complete8querybank schema')
            banks[arm]={o:z[o].copy() for o in ORDERS}
        records[arm]=r
    b,c=records['B']['trace'],records['C']['trace']
    for key in ('initial_state_sha256','scale','source_order_sha256','original_input_sha256','extra_rows_sha256','original_labels_sha256','extra_labels_sha256','base_batches','extra_batches','torch_rng_before','torch_rng_after','augment_rng_before','augment_rng_after'):
        if b[key]!=c[key]:raise ValueError('B/C matched recipe failed:'+key)
    if not b['extra_is_original_copy'] or c['extra_is_original_copy'] or b['extra_input_sha256']==c['extra_input_sha256'] or b['final_state_sha256']==c['final_state_sha256']:raise ValueError('extra EEG route inactive')
    banks['A']=original
    for arm,bb in banks.items():
        if any(p.shape!=(len(query),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6) for p in bb.values()):raise ValueError('incomplete/nonfinite canonical probabilities')
    np.savez(root/'complete_predictions.npz',index=query,**{a+'_'+o:p for a,bb in banks.items() for o,p in bb.items()})
    before=dict(all3arms_all3people_all8orders=True,source_sha256=sha(root/'source.sha256'),control_checkpoint_sha256=control['checkpoint_sha256'],
        original_source_proofs=proofs,predictions_sha256=sha(root/'complete_predictions.npz'),held_labels_opened=False)
    (root/'before_labels.json').write_text(json.dumps(before,indent=2));metric=official();scores={}
    def scheme(bb):return float(np.mean([np.mean([bb[o] for o in ORDERS if o.startswith(s)]) for s in ('within','full','runs')]))
    for person in ('P02','P03','P09'):
        mask=people[query]==person;rr=query[mask];y=labels(rr,len(people));scores[person]={}
        for arm,bb in banks.items():
            by={o:metric.score(y,p[mask].argmax(1),d,people==person)['balanced_accuracy_cell_mean'] for o,p in bb.items()}
            nll={o:float(-np.log(np.maximum(p[mask][np.arange(len(y)),y],1e-12)).mean()) for o,p in bb.items()}
            scores[person][arm]=dict(accuracy=scheme(by),by_order=by,pooled_windowNLL=scheme(nll))
    summaries={a:{k:v for k,v in r.items() if k!='trace'}|dict(trace={k:v for k,v in r['trace'].items() if k!='step_seconds'}) for a,r in records.items()}
    result=dict(source_sha256=sha(root/'source.sha256'),scores=scores,records=summaries,
        delta_pp={p:{a+'-'+b:100*(s[a]['accuracy']-s[b]['accuracy']) for a,b in [('C','B'),('C','A'),('B','A')]} for p,s in scores.items()},
        before_labels_sha256=sha(root/'before_labels.json'),metric_sha256=sha(metric.__file__),scope='INDIVIDUAL-MODEL source6/exposedP02/P03/P09/seed0;not independent/fullsystem/rank')
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(scores=scores,delta_pp=result['delta_pp'])),flush=True)


if __name__=='__main__':main()
