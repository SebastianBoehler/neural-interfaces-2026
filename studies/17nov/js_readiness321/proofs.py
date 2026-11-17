"""Fail-closed proof verification for the separate §321 preparation DAG."""
import json
import subprocess
from pathlib import Path

from reference import sha

ACTIVE=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_active5')
RECOVERY=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_inputrepair1')
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
BASES={'B2_pool','B2_s012','B2_s345','B2_s0','B2_s1','B2_s2'}


def read(path):return json.loads(Path(path).read_text())


def manifest(folder):
    subprocess.run(['sha256sum','-c','source.sha256'],cwd=folder,check=True,stdout=subprocess.DEVNULL)
    return sha(folder/'source.sha256')


def verify():
    source=manifest(ACTIVE);manifest(RECOVERY)
    recovery=read(RECOVERY/'recovery.json')
    if recovery.get('raw_input_all_seeds_orders_passed') is not True or recovery.get('raw_input_k')!=5.:
        raise ValueError('corrected raw-input six-seed/eight-order guard missing')
    if recovery['status']!='six_seed_historical_and_export_guards_passed' or [r['seed'] for r in recovery['models']]!=list(range(6)):
        raise ValueError('complete six historical source seeds required')
    for row in recovery['models']:
        if not row['historical_guard']['passed'] or row['historical_guard']['max_probability_error']!=0:
            raise ValueError('historical exact recovery failed')
        for kind in ('trace','checkpoint'):
            if sha(row[kind])!=row[kind+'_sha256']:raise ValueError('recovered artifact hash changed')
        e=row['export']
        if not e['saved_reset'] or not e['deepcopy_interleaved'] or max(e['cpu_probability_errors'].values())>=1e-4 or e['cuda_probability_error']>=1e-3:
            raise ValueError('historical export/state proof failed')
    build=read(ACTIVE/'build.json');active=read(ACTIVE/'outer_active_path.json')
    if active['status']!='generated_bundle_active_engineering_guards_passed' or active['source_sha256']!=source or active['build_sha256']!=sha(ACTIVE/'build.json'):
        raise ValueError('active proof source/build provenance differs')
    if build['source_sha256']!=source or build['recovery_sha256']!=sha(RECOVERY/'recovery.json') or not build['original_keys_unchanged']:
        raise ValueError('generated artifact provenance differs')
    for folder,files in build['files'].items():
        for name,digest in files.items():
            if sha(ACTIVE/folder/name)!=digest:raise ValueError('generated artifact changed')
    for key in ('disabled_identity','original_inner_gate_identity','original_ft_label_identity','study_outer_mix_parity','route_share17_verified'):
        if active.get(key) is not True:raise ValueError('missing active guard: '+key)
    if set(active['cases'])!={'known','unrecognized'}:raise ValueError('missing policy case')
    for case in active['cases'].values():
        for key in ('disabled_final_bitwise','inner_bitwise','original_ft_labels_bitwise','reset_deepcopy_prefix'):
            if case.get(key) is not True:raise ValueError('missing state/identity guard')
        if case['owning_mix_max_error']>1e-10 or case['cc_calls']!=45 or abs(case['route_share']-1/17)>1e-15:
            raise ValueError('active path numerical/activity proof failed')
    for order in ORDERS:
        r=read(ACTIVE/f'study_recipe_{order}.json')
        if r['status']!='actual318_six_draw_mix_capture_passed' or r['order']!=order or r['accuracy_computed'] or set(r['bases'])!=BASES:
            raise ValueError('incomplete no-score eight-order/six-draw proof')
        if r['generated_bundle_sha256']!=sha(ACTIVE/'draft_jsall/submission.py'):raise ValueError('recipe bundle differs')
        for b in r['bases'].values():
            if b['max_error']>1e-10 or abs(b['core1_share']-1/17)>1e-15 or not b['old_fb_removed'] or b['whole_fbl_factor']!=2:
                raise ValueError('study recipe mismatch')
    return dict(active_source=source,active_proof=sha(ACTIVE/'outer_active_path.json'),recovery=sha(RECOVERY/'recovery.json'),policy='JS-all')
