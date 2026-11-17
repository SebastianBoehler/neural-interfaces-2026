"""Post-hoc B2_pool contrast, only after unchanged complete318owning collector."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117')


def main():
    sys.path[:0]=[str(ROOT/'studies/14nov/newmember'),
                  str(ROOT/'studies/16nov/followups/coral_repair/chain/recovery')]
    from control_source import install
    path=ROOT/'studies/14nov/newmember/score_s_v66p.py'
    spec=importlib.util.spec_from_file_location('pinned318_pool_contrast',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);install(module,ROOT)
    with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        n=len(z['y|all_loso']);sessions=len(np.unique(z['sess|all_loso']))
    # Owning collector independently rechecks every complete source/zero/order/base before labels.
    outputs=module.collect(ROOT,n,sessions)
    with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        y,cells,people=(z[k+'|all_loso'] for k in ('y','cell','person'))
    counts=np.array([len(np.unique(cells[people==p])) for p in module.PEOPLE])
    results={}
    for base in module.BASES:
        control=np.stack([module.person_accuracy(outputs['control',o,base],y,cells,people,module.PEOPLE) for o in module.ORDERS])
        joint=np.stack([module.person_accuracy(outputs['JS',o,base],y,cells,people,module.PEOPLE) for o in module.ORDERS])
        results[base]=dict(control_official=module.cell_weighted_summary(control,module.ORDERS,counts),
                           js_official=module.cell_weighted_summary(joint,module.ORDERS,counts),
                           official_effect=module.cell_weighted_summary(joint-control,module.ORDERS,counts),
                           people_order_effect=module.summarize(joint-control,module.ORDERS,module.PEOPLE))
    report=dict(status='complete318_guarded_posthoc_B2_pool',primary_base='B2_pool',bases=results,
                scorer_sha256=module.sha256(path),scope='selected reused-dev new-person routing emulation;notpackage orreturning validation')
    Path('pool_contrast.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(results['B2_pool'],indent=2))


if __name__=='__main__':main()
