"""Complete matched four-arm/source6-control banks before P09 labels and quality."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from common import source,native,capture,embedding,sha,digest,ARMS,labels,official
from retention import prepare,metrics

ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
CONTROL=Path('/mnt/lustre/work/gehler/gfh527/ni26_combinations338_v2_20261117')
CONTROL_SHA='37ee61961a09c79344dd5c92d0395a28e971b5c0f6db7aa2fba94f5aa1a9ec6a'


def main():
    root=Path(os.environ['PRETRAIN_ROOT']);torch.set_num_threads(6);d,fit,held,people,yfit=source()
    # Existence requires complete338 collector, but its scores are not read to choose337 paths.
    if not (CONTROL/'results.json').exists() or sha(CONTROL/'source.sha256')!=CONTROL_SHA:raise ValueError('complete338 control dependency')
    path=CONTROL/'output/source6/A';control=json.loads((path/'record.json').read_text())
    if control['scope']!='source6' or control['arm']!='A' or control['source_sha256']!=CONTROL_SHA or control['query_labels_opened']:raise ValueError('source6 control identity')
    if control['checkpoint_sha256']!='eadd5c23c9f7cac239bf78a99993d8d3e47392d24d166e0ede6fce5b0e6ac9f8' or control['prediction_sha256']!='a3ae189611f5a4d46048f19ec397d758bb07fe47ab7f8984817eb66bdd2596c0' or control['prediction_sha256']!=sha(path/'predictions.npz') or control['checkpoint_sha256']!=sha(path/'model.pt') or control['source_rows_sha256']!=digest(fit) or control['query_rows_sha256']!=digest(held):raise ValueError('control checkpoint/rows')
    reference=control['trace'];records={'native':control};banks={};embeddings={}
    with np.load(path/'predictions.npz') as z:
        if not np.array_equal(z['index'],held):raise ValueError('control query order')
        banks['native']={o:z[o].copy() for o in ORDERS}
    for arm in ARMS:
        folder=root/'output'/arm;record=json.loads((folder/'record.json').read_text());t=record['fine'];p=record['pretext']
        if record['source_sha256']!=sha(root/'source.sha256') or record['held_labels_opened'] or record['source_rows_sha256']!=digest(fit) or record['held_rows_sha256']!=digest(held):raise ValueError('producer scope')
        if record['checkpoint_sha256']!=sha(folder/'model.pt') or record['prediction_sha256']!=sha(folder/'predictions.npz') or record['embedding_sha256']!=sha(folder/'embeddings.npz'):raise ValueError('producer file identity')
        if p['steps']!=64 or p['initial_native_state_sha256']!=reference['initial_state_sha256'] or not p['fine_rng_restored']:raise ValueError('native initialization/pretext/restore')
        for key in ('scale','steps','batches','torch_rng_before','torch_rng_after','augment_rng_before','augment_rng_after'):
            if t[key]!=reference[key]:raise ValueError('unintended finefit recipe change:'+key)
        if not t['head_is_parametrized'] or t['criterion_weights'] is not None or not t['optimizer_owns_actual_head']:raise ValueError('finefit head/loss/optimizer')
        with np.load(folder/'predictions.npz') as z:
            if set(z.files)!={'index'}|set(ORDERS) or not np.array_equal(z['index'],held):raise ValueError('prediction schema')
            banks[arm]={o:z[o].copy() for o in ORDERS}
        with np.load(folder/'embeddings.npz') as z:
            if not np.array_equal(z['rows'],np.r_[fit,held]):raise ValueError('representation rowmap')
            embeddings[arm]=z['features'].copy()
        records[arm]=record
    for p in banks.values():
        if any(v.shape!=(len(held),3) or not np.isfinite(v).all() or not np.allclose(v.sum(1),1,atol=1e-6) for v in p.values()):raise ValueError('posterior finite/complete')
    pretexts=[records[a]['pretext'] for a in ARMS]
    for key in ('pretext_rows_sha256','shuffle_trace_sha256','torch_rng_pretext_before','torch_rng_pretext_after'):
        if len({p[key] for p in pretexts})!=1:raise ValueError('pretext matching:'+key)
    model=native(d);capture(model,fit);state=torch.load(path/'model.pt',map_location='cpu',weights_only=False)
    model.net.load_state_dict(state['state'],strict=True)
    if model.scale!=state['scale'] or not np.array_equal(model.G,state['G']):raise ValueError('control preprocessing differs')
    embeddings['native']=embedding(model,np.r_[fit,held]);n=len(fit)
    with np.load(Path(os.environ['TEACHER_ROOT'])/'teacher.npz') as z:
        if not np.array_equal(z['rows'],np.sort(np.r_[fit,held])):raise ValueError('teacher rows')
        positions=np.searchsorted(z['rows'],np.r_[fit,held]);teacher=z['covariance'][positions].copy()
    with torch.no_grad():native_source_probs=torch.softmax(model.net.final_layer(torch.from_numpy(embeddings['native'][:n])),dim=1).numpy()
    native_quality=dict(CE=float(-np.log(np.clip(native_source_probs[np.arange(n),yfit],1e-12,1)).mean()),class_recalls=[float(np.mean(native_source_probs[yfit==k].argmax(1)==k)) for k in range(3)])
    maps={};checks={}
    for arm in ('native',)+ARMS:maps[arm],checks[arm]=prepare(embeddings[arm],teacher,n)
    np.savez(root/'ridge_maps.npz',**maps)
    (root/'before_labels.json').write_text(json.dumps(dict(all5banks_guards_maps_complete=True,source_sha256=sha(root/'source.sha256'),control_checkpoint_sha256=control['checkpoint_sha256'],held_labels_opened=False)))
    reconstruction={a:metrics(maps[a],teacher,n,checks[a]['source_target_mean']) for a in maps}
    y=labels(held,len(people));metric=official();scores={}
    from diagnostics import probability_quality,scheme
    for arm,values in banks.items():
        by={o:metric.score(y,p.argmax(1),d,people=='P09')['balanced_accuracy_cell_mean'] for o,p in values.items()}
        quality={o:probability_quality(d,held,y,p) for o,p in values.items()}
        scores[arm]=dict(accuracy=scheme(by),by_order=by,NLL=scheme({o:v['NLL'] for o,v in quality.items()}),
            Brier=scheme({o:v['Brier'] for o,v in quality.items()}),class_recalls=[scheme({o:v['class_recalls'][k] for o,v in quality.items()}) for k in range(3)])
    checks={a:{k:v for k,v in c.items() if k!='source_target_mean'} for a,c in checks.items()}
    report=dict(source_sha256=sha(root/'source.sha256'),scores=scores,reconstruction=reconstruction,probe_checks=checks,records=records,
        control_checkpoint_sha256=control['checkpoint_sha256'],native_source_quality=native_quality,adequate_native_anchor=bool(native_quality['CE']<1 and min(native_quality['class_recalls'])>.10 and scores['native']['accuracy']>.40),before_labels_sha256=sha(root/'before_labels.json'),metric_sha256=sha(metric.__file__),P02P03_inference=False,
        scope='P09sourceheld;singleperson/seed selectedinstrument;notfamilyclosure')
    (root/'result.json').write_text(json.dumps(report));print(json.dumps(scores),flush=True)


if __name__=='__main__':main()
