"""Verify the existing §376 gate traces required by the new member execution wrapper."""
import json
from pathlib import Path
import shutil

import numpy as np
from context import CHAIN, HERE, ORDERS, PARENT, pin, require, sha


def main():
    pin(HERE/'runtime.sha256','cc501aa5e8ab7ce6f4a7cfe820658bb410843739dcff52fbb8b06f8d822b5940')
    pin(PARENT/'runtime.sha256','a0e38b986672e3820c23b90f1620df6a5f792893063dbfcc4dc26c6fcb8ae7f9')
    result = {}
    for order in ORDERS:
        rel = Path(f'runs/newmember/s281_v66p_control_{order}_gate.npz')
        reference, target = PARENT/'chain'/rel,CHAIN/rel
        side = PARENT/f'chain/runs/s281_v66_cp/control_{order}_B2_pool.json'
        record = json.loads(side.read_text())
        require((record['arm'],record['order'],record['base'])==('control',order,'B2_pool'),'bad parent sidecar')
        pin(reference,record['inputs_sha256'][str(reference)])
        if order=='within0':
            with np.load(target) as a,np.load(reference) as b:
                require(set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),
                        'fresh within0 gate differs')
        else:
            with reference.open('rb') as src,target.open('xb') as dest:
                shutil.copyfileobj(src,dest)
            pin(target,sha(reference))
        result[order] = dict(sidecar=sha(side),old_gate=sha(reference),local_gate=sha(target))
    (HERE/'gates.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
