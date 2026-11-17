"""Copy-only genuine source-neural returning bridge; conditional fixed classical priors, never promotion."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from returning_assets import ROOT,asset,sha

ARRAYS=ROOT/'ni26_js321_20261117_returning_plain_arrays_all1'
SOURCES={'base':ROOT/'ni26_js321_20261117_active5/draft_base',
         'off':ROOT/'ni26_js321_20261117_inputreuse2/draft_off',
         'js':ROOT/'ni26_js321_20261117_inputreuse2/prepared_jsall'}


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy(order='K') for k in z.files}


def verified(path,status):
    p=json.loads(path.read_text())
    if p['status']!=status:raise ValueError('owning proof missing: '+str(path))
    return p


def replace_file(source,target):
    if target.exists():target.chmod(0o644)
    shutil.copyfile(source,target)
    if sha(source)!=sha(target):raise ValueError('private source file copy failed')


def main():
    archive=ROOT/'ni26_js321_20261117_active5/base_v66.zip';original_sha=sha(archive)
    if original_sha!='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed':raise ValueError('frozenZIP changed')
    verified(ROOT/'ni26_js321_20261117_returning_plain_complete1/plain_complete.json','all28_new_genuine_source_plain_weights_complete_NOT_accuracy')
    array=verified(ARRAYS/'array_guard.json','original_plain_runtime_source_array_export_passed_NOT_accuracy')
    if array['mode']!='all' or len(array['cases'])!=28:raise ValueError('all28 original inference export guards required')
    teacher=ROOT/'ni26_js321_20261117_returning_teacher_map1'
    verified(teacher/'teacher_complete.json','both_source_teacher_maps_complete_NOT_accuracy')
    records=[]
    for fold,index in (('P04_s23',0),('P10_s23',1)):
        bank=ROOT/'ni26_js321_20261117_returning_view_bank1'/fold
        verified(bank/'proof.json','full_native_raw_causal256_eightorder_bank_passed_NOT_accuracy')
        context=ROOT/'ni26_js321_20261117_returning_context_new1'/f'context_{index}'
        c=verified(context/f'context_{fold}.json','fold_personal_context_source_fit_complete_ENGINEERING_ONLY')
        if sha(context/f'context_{fold}.npz')!=c['arrays_sha256']:raise ValueError('source context changed')
        gal=ROOT/'ni26_js321_20261117_returning_galleries3'/fold
        g=verified(gal/'proof.json','new_source_coordinate_galleries_complete_CONDITIONAL_oldOOF')
        if sha(gal/'galleries.npz')!=g['arrays_sha256'] or not g['old_OOF_outer_exposure_RETAINED']:raise ValueError('source calibration scope changed')
        fit=ROOT/('ni26_js321_20261117_returning_view_fit1' if index==0 else 'ni26_js321_20261117_returning_view_fit_p10_1')
        viewfile=fit/f'source_view_{fold}.npz'
        if sha(viewfile)!=c['feature_basis_sha256'] or sha(viewfile)!=g['source_basis_sha256']:raise ValueError('source coordinate mismatch')
        updates={k:v for k,v in load(viewfile).items() if k.startswith('ts_') or k in ('tsem_m','tsem_V','route_V_hi')}
        updates.update(load(context/f'context_{fold}.npz'));updates.update(load(gal/'galleries.npz'));updates.pop('metadata',None)
        for r in array['cases']:
            if r['fold']!=fold:continue
            name=f"{r['family']}_{fold}_s{r['seed']}_arrays.npz";file=ARRAYS/name
            if sha(file)!=r['array_sha256']:raise ValueError('original-class source arrays changed')
            start={'shallow_ea':0,'shallow_raw':5,'fbcnet_ea_tf':20,'fbcnet_raw_tf':22}[r['family']]
            updates.update({f"net{start+r['seed']}_{k}":v for k,v in load(file).items()})
        tp=verified(teacher/fold/'proof.json','new_source_teacher_taskmap_export_passed_NOT_accuracy')
        if sha(teacher/fold/'teacher.pt')!=tp['teacher_sha256']:raise ValueError('source task map changed')
        for arm,origin in SOURCES.items():
            target=Path(fold)/arm;target.parent.mkdir(exist_ok=True);shutil.copytree(origin,target)
            weights=load(target/'weights.npz');weights.update(updates)
            replace_file(teacher/fold/'teacher.pt',target/'teacher.pt')
            for kind,start in (('fb',35),('fb5',41),('fbraw',47),('sinc',53),('fbl',59)):
                for seed in range(6):
                    checkpoint,trace,proof=asset(kind,fold,seed);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    slot=start+seed;name=str(weights[f'net{slot}_sjit_file']);replace_file(trace,target/name)
                    for k,v in dict(align=kind!='fbraw',k=5.,G=ck['G'],scale=ck['scale'],shrink=.001).items():weights[f'net{slot}_{k}']=np.asarray(v)
            if arm=='js':
                for seed,name in enumerate(weights['js_cc_files']):
                    checkpoint,trace,proof=asset('cc30',fold,seed);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    replace_file(trace,target/str(name))
                    p=dict(align=True,k=5.,G=ck['G'],scale=ck['scale'],shrink=.001)
                    if seed and any(not np.array_equal(weights['js_'+k],v) for k,v in p.items()):raise ValueError('six source auxiliary input priors differ')
                    for k,v in p.items():weights['js_'+k]=np.asarray(v)
                for k in ('align','k','G','scale','shrink'):
                    if not np.array_equal(weights['js_'+k],weights['net35_'+k]):raise ValueError('aux current-input reuse fingerprint differs')
            for slot in range(10):
                if f'net{slot}_jit_file' in weights:raise ValueError('Shallow inference class changed')
            for slot in range(20,24):
                if f'net{slot}_jit_file' in weights:raise ValueError('FBC inference class changed')
            file=target/'weights.npz';file.chmod(0o644);np.savez(file,**weights)
            reread=load(file)
            if any(v.dtype!=reread[k].dtype or v.shape!=reread[k].shape or v.strides!=reread[k].strides or not np.array_equal(v,reread[k],equal_nan=True) for k,v in weights.items() if v.dtype.kind not in 'US'):
                raise ValueError('private NPZ layout/value serialization changed')
            records.append(dict(fold=fold,arm=arm,path=str(target),source_package=str(origin),
                                source_module_sha256=sha(origin/'submission.py'),module_sha256=sha(target/'submission.py'),
                                files={p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()}))
    if sha(archive)!=original_sha:raise ValueError('frozen original changed')
    Path('bundle_complete.json').write_text(json.dumps(dict(status='twofold_three_arm_private_source_bridge_built_NOT_validated',artifacts=records,
        source_sha256=sha('source.sha256'),accuracy_computed=False,scope='ENGINEERING_CONDITIONAL_FIXED_PRODUCTION_PRIOR_ONLY',
        retained='ORIGINAL rest4/eye auxiliary productionNN;other classical member weights outside core1;globalMAD/event/scalar priors;teacherG;oldOOFposterior calibration',
        source_fitted='PRIMARY+INNER NetBank neural task weights/maps;core1/PCA256;personalheads/Tpri/OwnTCS;truecontrast/bio galleries and centering;NOT rest4/eyeaux',
        required_next='actualmodule/source/offidentity/innerFT/gate/reset/deepcopy/fullraw active guards BEFORE any wholepaired conditional score'),indent=2))


if __name__=='__main__':main()
