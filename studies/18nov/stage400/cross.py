"""Separate the family mix supplying current evidence from the native rebalancing state."""
import json
import sys
import numpy as np
from capture import HERE,ORIG,main as run_actor

TRACE=ORIG.parent.parent/'ni26_stage400_v1_20261118'


def main(arm,order):
    assert arm in ('L','D','RB')
    with np.load(TRACE/f'RB_{order}.npz') as z,np.load(TRACE/f'{arm}_{order}.npz') as a:
        native={k:z[k] for k in z if k.endswith(('|mix','|rebal'))}
        original={k:a[k] for k in native}
    seen={}
    record={}
    def transform(base,stage,value):
        if stage=='mix':
            assert np.allclose(value,original[base+'|mix'],rtol=0,atol=1e-10)
            seen[base]=value
        if stage!='rebal':
            return value
        n=native[base+'|mix']
        ref=native[base+'|rebal']
        assert np.allclose(value,original[base+'|rebal'],rtol=0,atol=1e-10) and (n>0).all()
        out=seen[base]*(ref/n)
        out=out/out.sum(1,keepdims=True)
        delta=float(np.max(abs(out-value)))
        if arm=='RB':
            assert delta<1e-10
        else:
            assert delta>1e-4
        # The supplied multiplier is rowwise and causal by the owning RB rebalancing implementation.
        assert np.isfinite(out).all() and np.allclose(out.sum(1),1)
        record[base]=dict(max_delta=delta,changed=int((out.argmax(1)!=value.argmax(1)).sum()))
        return out
    name='cross'+arm
    dest=HERE/f'{name}_{order}.npz'
    run_actor(arm,order,transform,dest)
    if arm=='RB':
        with np.load(dest) as z,np.load(TRACE/f'RB_{order}.npz') as old:
            for k in z:
                assert np.allclose(z[k],old[k],rtol=0,atol=2e-6)
                assert np.array_equal(z[k].argmax(1),old[k].argmax(1))
    (HERE/f'{name}_{order}.json').write_text(json.dumps(record))


if __name__=='__main__':
    main(*sys.argv[1:])
