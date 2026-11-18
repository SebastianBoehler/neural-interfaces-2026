"""Tap literal §395c inference stages. The old runner can only write into memory."""
import json
import os
from pathlib import Path
import sys
import numpy as np

WORK = Path('/mnt/lustre/work/gehler/gfh527')
ORIG = WORK/'ni26_policy395_chain_v1_20261118'/'chain'
HERE = Path(__file__).resolve().parent
RUN = ORIG/'studies/14nov/newmember'
sys.path.insert(0, str(RUN))
import run_s_v66 as original


def main(arm, order, transform=None, destination=None):
    add, swap, zero = json.loads((ORIG/'representation_config.json').read_text())['recipes'][arm]
    os.environ.update(ADD=add, SWAP=swap, ZERO=str(int(zero)), NM_NAME=arm, NM_SEEDS='0,1', NI26_WS=str(ORIG))
    os.chdir(ORIG)
    sys.argv = ['run_s_v66.py',order]
    stages = {}
    runner = RUN/'run_s.py'
    ctx = {'__name__':'__main__','__file__':str(runner),'CURRENT':{'arm':None}}
    archive = ORIG/f'runs/chainout/s281_v66p_{arm}_{order}.npz'

    def tap(bn, mode, perm, stage, value):
        if mode == 'V6Fr':
            key=f'{bn}|{stage}'
            assert key not in stages and value.shape == (2655,3)
            canonical=value[np.argsort(perm)].copy()
            if transform is not None:
                canonical=transform(bn,stage,canonical)
                value=canonical[perm]
            stages[key]=canonical
        return value

    def checked(outputs):
        with np.load(archive) as z:
            for k in outputs:
                if transform is not None and k.endswith('|V6Fr'):
                    assert np.isfinite(outputs[k]).all()
                    continue
                assert np.array_equal(z[k].argmax(1),outputs[k].argmax(1))
                assert np.allclose(z[k],outputs[k],rtol=0,atol=2e-6)
        for k,v in outputs.items():
            if k.endswith('|V6Fr'):
                stages[k]=v.copy()

    def execute():
        p=ORIG/'studies/13nov/shufcombo.py'
        code=original.patch_chain(p.read_text())
        code=original.replace_once(code,'b = gm.M.rebal(A.mix(bn, p0, fams, extra)[0], p0, None)',
            'b = tap(bn,arm,P,"rebal",gm.M.rebal(tap(bn,arm,P,"mix",A.mix(bn,p0,fams,extra)[0]),p0,None))')
        code=original.replace_once(code,'gm.tsem_g(gm.em_g(b, p0, g), p0, g)',
                                    'gm.tsem_g(tap(bn,arm,P,"em",gm.em_g(b,p0,g)),p0,g)')
        ctx.update(__file__=str(p),capture_gate=lambda b,a,f,v,p:f(v,p),checked_save=checked,tap=tap)
        exec(compile(code,str(p),'exec'),ctx)

    ctx['execute_chain']=execute
    code=original.patch_runner(runner.read_text())
    start=code.index('def savez(path,')
    end=code.index('np.savez = savez',start)
    # Historical summary is already archived. Forbid every disk write from the instrumented runner.
    code=code[:start]+'''def savez(path,*a,**k):
    assert os.path.basename(path).startswith("shufcombo_")
'''+code[end:]
    old=np.savez
    try:
        exec(compile(code,str(runner),'exec'),ctx)
    finally:
        np.savez=old
    assert len(stages)==24
    np.savez_compressed(destination or HERE/f'{arm}_{order}.npz',**stages)


if __name__=='__main__':
    main(*sys.argv[1:])
