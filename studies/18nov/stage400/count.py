"""Recover the exact causal count stage from the archived pre-personalization actor."""
import sys
import numpy as np
from capture import HERE,ORIG


def main(order):
    sys.path.insert(0,str(ORIG/'studies/14nov/rundetect'))
    import package as K
    assert np.array_equal(K.ROWS,np.arange(2655))
    cache=K.J.hyp_cache(order,'wprior',K.X,K.LAM,sessions=K.SESS,proto=K.PROTO)
    latch=K.latch(order)
    result={}
    for arm in ('control','RB','L','M','D'):
        name=f's281_v66p_{arm}_{order}'
        with np.load(ORIG/f'runs/chainout/{name}.npz') as z, np.load(ORIG/f'runs/newmember/{name}_gate.npz') as audit:
            for k in z:
                if not k.endswith('|V6Fr'):
                    continue
                raw=z[k].astype(np.float64)
                assert np.array_equal(raw,audit[k])
                filtered=K.gate(raw,K.J.filt_joint(raw,cache,'urn'),order,latch)
                result[arm+'|'+k.split('|')[0]]=filtered
    assert len(result)==30
    np.savez_compressed(HERE/f'count_{order}.npz',**result)


if __name__=='__main__':
    main(sys.argv[1])
