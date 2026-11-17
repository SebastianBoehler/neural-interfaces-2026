"""Capture actual318post-FT mix inputs for six draws; abort before any stages or scoring."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from reference import sha

OFFICIAL=Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/2026-competition/tracks/bci_decoding')
sys.path.insert(0,str(OFFICIAL))
import benchmark_utils.base_solver

STUDY=Path('/mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117')


class Captured(Exception):pass


def main(order):
    if order not in ('within0','within1','within2','within3','full0','full1','runs0','runs1'):
        raise ValueError('registered pinned order required')
    ws=Path.cwd();sys.path.insert(0,str(STUDY/'studies/14nov/newmember'))
    path=STUDY/'studies/14nov/newmember/run_s_v66.py'
    spec=importlib.util.spec_from_file_location('owning318_wrapper',path)
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    spec=importlib.util.spec_from_file_location('generated321_recipe',ws/'draft_jsall/submission.py')
    generated=importlib.util.module_from_spec(spec);spec.loader.exec_module(generated)
    os.environ.update(NM_NAME='JS',ADD='none',SWAP='fb_cc30',ZERO='0',NM_SEEDS='0,1,2,3,4,5')
    runner=STUDY/'studies/14nov/newmember/run_s.py';source=wrapper.patch_runner(runner.read_text())
    context={'__name__':'no_score321_runner','__file__':str(runner),'CURRENT':{'arm':None}}
    chain_context={}

    def prepare_chain():
        chain=STUDY/'studies/13nov/shufcombo.py';text=wrapper.patch_chain(chain.read_text())
        marker='ident = np.arange(N); R = {}; Pperm ='
        if text.count(marker)!=1:raise ValueError('owning chain setup boundary changed')
        prefix=text[:text.index(marker)]
        chain_context.update(context,__file__=str(chain),__name__='no_score321_chain')
        chain_context['capture_gate']=lambda *a:(_ for _ in ()).throw(ValueError('stage gate unexpectedly executed'))
        exec(compile(prefix,str(chain),'exec'),chain_context)
        chain_context['SC'].person=lambda *a:(_ for _ in ()).throw(ValueError('accuracy unexpectedly requested'))
    context['execute_chain']=prepare_chain
    original_argv=sys.argv;sys.argv=[str(runner),order]
    try:exec(compile(source,str(runner),'exec'),context)
    finally:sys.argv=original_argv
    A=chain_context['A'];actual=A.mix;original=context['_mix'];results={}
    permutation=chain_context['S'].perm_for(order[:-1],int(order[-1]))['all_loso']

    def capture(bn,proto,families,extra=(),*args,**kwargs):
        before=original(bn,proto,families,extra,*args,**kwargs)
        after=actual(bn,proto,families,extra,*args,**kwargs)
        members=A.members(bn,proto);weights=dict(extra)
        if weights.get('sc_fb')!=1. or weights.get('sc_fbl')!=1. or weights.get('sc_rest')!=1.:
            raise ValueError('actual study family unit weights changed')
        cc=context['stream']('fb_cc30',bn)[context['CUR']['P']]
        expected=generated.js_outer_kernel(before[0],members['sc_fb'],cc,members['sc_fbl'],1/16)
        difference=float(np.abs(expected-after[0]).max())
        if difference>1e-10 or abs(after[1]-1/17)>1e-15 or abs(before[1]-1/16)>1e-15:
            raise ValueError('generated outer kernel differs from actual study mix')
        results[bn]=dict(max_error=difference,rows=len(expected),core1_share=after[1],
                         actual_source_seeds=context['BS'][bn],old_fb_removed=True,whole_fbl_factor=2.)
        raise Captured()
    A.mix=capture
    for base in ('B2_pool','B2_s345','B2_s012','B2_s0','B2_s1','B2_s2'):
        try:chain_context['chain'](base,'V6Fr',permutation)
        except Captured:pass
        else:raise ValueError('owning chain reached stages/scoring without capture')
    report=dict(status='actual318_six_draw_mix_capture_passed',order=order,bases=results,
                runner_sha256=sha(runner),wrapper_sha256=sha(path),generated_bundle_sha256=sha(ws/'draft_jsall/submission.py'),
                accuracy_computed=False,scope='actual post-FT cached study inputs;generated kernel only;not whole package held-fold parity')
    (ws/f'study_recipe_{order}.json').write_text(json.dumps(report,indent=2))
    print('actual318mix recipe matched without stage/label evaluation',order,flush=True)


if __name__=='__main__':main(sys.argv[1])
