"""Actual frozen-stage fold injection parity; recording EEG engineering only, never held accuracy."""
import ast
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

import worker
from returning_stage_ref import Reference

ROOT=Path('/mnt/lustre/work/gehler/gfh527')


def inner(model):
    return dict(out=np.asarray(model.trace['out_v5']).copy(),contrast=model.trace['contrast'],identity=model.trace['identity'],
                labels=np.asarray(model.ft.lab).copy())


def compare_inner(a,b):
    for k in a:
        if a[k] is None or b[k] is None:
            if a[k] is not b[k]:raise ValueError('original inner gate presence changed')
        elif not np.array_equal(a[k],b[k],equal_nan=True):raise ValueError('original inner label/gate computation changed')


def main():
    torch.set_num_threads(1)
    if torch.__version__!='2.14.1+cu130' or 'neural-interfaces-2026/.venv/' in Path('/proc/self/maps').read_text():raise ValueError('actual clean worker image required')
    context=Path('/context/context_P04_s23.npz');meta=json.loads(Path('/context/context_P04_s23.json').read_text())
    if meta['status']!='fold_personal_context_source_fit_complete_ENGINEERING_ONLY' or meta['arrays_sha256']!=worker.sha(context):raise ValueError('source context proof absent')
    with np.load(context) as z:inject={k:z[k].copy(order='K') for k in z.files if k!='metadata'}
    source=ROOT/'neural-interfaces-2026/runs/replay_data/feats_exact/seen_session-P04_s23.npz'
    if worker.sha(source)!=meta['feature_basis_sha256']:raise ValueError('feature context/basis differs')
    with np.load(source) as f:
        key=list(f['sessions']).index('Dreyer2026Proteus/P04|2');rows=f[f'{key}|idx'][:80];U=f[f'{key}|pc'][:80,:256].astype(np.float64)
    if not np.all(np.diff(rows)>0):raise ValueError('engineering stimulus must be acquisition order')
    with np.load(ROOT/'neural-interfaces-2026/data/proteus_windows.npz',allow_pickle=True) as d:x=d['X'][rows].copy();channels=d['ch_names'].copy()
    results={};scores={};original=[]
    for arm,path in (('base',Path('/active/draft_base')),('off',Path('/experiment/draft_off')),('js',Path('/experiment/prepared_jsall'))):
        template=worker.load_model(path,channels,'actual_fold_injection_'+arm,'cpu')
        module_path=Path(type(template).step.__globals__['__file__']).resolve()
        if module_path!=(path/'submission.py').resolve():raise ValueError('actual requested module path differs')
        namespace=type(template.routeB).stepx.__globals__
        if arm=='base':
            untouched=worker.fresh_model(template)
            for raw in x:untouched.predict(torch.from_numpy(raw[None]));original.append(inner(untouched))
        model=worker.fresh_model(template)
        with np.load(path/'weights.npz') as z:weights={k:z[k].copy(order='K') for k in z.files}
        if not np.array_equal(weights['route_names'],inject['route_names']):raise ValueError('gallery name/index differs')
        weights.update(inject)
        if not int(weights['v6_tpri_on']) or float(weights['v6_tpri_kappa'])!=20.:raise ValueError('actual outer personal prior disabled')
        old_inner_W=model.route.W.copy();old_inner_c=model.route.c.copy();old_inner_prior=model.route.own.prior.copy()
        model.routeB.W=weights['route_ps1_W'].copy();model.routeB.c=weights['route_ps1_c'].copy()
        model.routeB.own=namespace['OwnStage'](weights);model.routeB.pri=weights['v6_tpri'].copy();model.routeB.kappa=20.
        if not np.array_equal(old_inner_W,model.route.W) or not np.array_equal(old_inner_c,model.route.c) or not np.array_equal(old_inner_prior,model.route.own.prior):raise ValueError('fold injection aliased inner original arrays')
        if not np.isclose(model.route.a,1/15,atol=1e-15) or not np.isclose(model.routeB.a,1/(17 if arm=='js' else 16),atol=1e-15):raise ValueError('inner/outer normalization changed')
        model.reset_state();reference=Reference(model,weights,namespace);captured={};index=[0]
        actual_step=model.routeB.stepx
        def fold_step(raw,u,mix,core,extra,contrast,identity):
            # Explicit engineering coordinate adapter. The inner backbone stays untouched.
            view=U[index[0]];captured['args']=(raw,view,mix.copy(),core.copy(),extra.copy(),contrast,identity)
            return actual_step(raw,view,mix,core,extra,contrast,identity)
        model.routeB.stepx=fold_step
        actual_cp=model.cp.step
        def capture_cp(*args):
            out=actual_cp(*args);captured['score']=out.copy();return out
        model.cp.step=capture_cp
        errors=[];outputs=[]
        for t,raw in enumerate(x):
            index[0]=t;label=int(model.predict(torch.from_numpy(raw[None]))[0]);compare_inner(original[t],inner(model))
            expected_raw,expected_score=reference.step(*captured['args'],model.recB)
            error=max(float(np.abs(expected_raw-model.trace['out']).max()),float(np.abs(expected_score-captured['score']).max()))
            if error>1e-8 or label!=int(expected_score.argmax()):raise ValueError(f'actual RouteB/final CF/CP composition differs {error}')
            errors.append(error);outputs.append(captured['score'])
            if model.dead or model.log6['err'] or model.routeB.log['err'] or model.cf.dead or model.cp.dead:raise ValueError('actual worker source path failed')
        if model.log6['rec']!=10 or model.ft.refits or model.cp.stopped!=10 or model.routeB.log['staged']<=0:
            raise ValueError('known recognition/stops/own-stage activation differ')
        if reference.tpri_checks<=0 or reference.prior_effect<=1e-12 or reference.source_heads<=0:raise ValueError('head/personalTSEM inactive')
        scores[arm]=np.asarray(outputs);results[arm]=dict(max_error=max(errors),recognized_at=model.log6['rec'],tpri_checks=reference.tpri_checks,
             tpri_max_effect=reference.prior_effect,source_head_windows=reference.source_heads,own_staged=model.routeB.log['staged'],
             ft_refits=model.ft.refits,cp_stopped=model.cp.stopped,original_inner_identity=True)
    if not np.array_equal(scores['base'],scores['off']):raise ValueError('disabled same-path source context differs')
    delta=float(np.abs(scores['js']-scores['base']).max());pre=float(np.abs(scores['js'][:10]-scores['base'][:10]).max());post=float(np.abs(scores['js'][10:]-scores['base'][10:]).max())
    if min(pre,post)<=1e-12:raise ValueError('JS outer source context inactive before/after recognition')
    Path('/out/stage_guard.json').write_text(json.dumps(dict(status='actual_fold_injected_stage_composition_passed_ENGINEERING_ONLY',
        cases=results,disabled_bitwise=True,active_max_delta=delta,active_pre_recognition_delta=pre,active_post_recognition_delta=post,accuracy_computed=False,
        source_sha256=worker.sha('source.sha256'),scope='real recording-order source-basis stimulus; inner all10 backbone retained;NOT whole heldfold accuracy/rawview proof'),indent=2))


if __name__=='__main__':main()
