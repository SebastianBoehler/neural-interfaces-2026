"""Source-only diagnosis of preserved feature-anchor failure; no fitting/scoring."""
import json
import platform
from pathlib import Path

import numpy as np

from common import OLD, inputs, layouts, load, sha

FAILED=Path('/mnt/lustre/work/gehler/gfh527/ni26_spatial342_pilot_v2_20261117')


def differences(actual, expected, rows):
    error=np.abs(actual-expected);relative=error/np.maximum(np.abs(expected),1e-12)
    bad=error>(2e-4+1e-4*np.abs(expected));worst=np.unravel_index(error.argmax(),error.shape)
    return dict(max_abs=float(error.max()),max_relative=float(relative.max()),
                max_abs_coordinate=list(map(int,worst)),max_abs_official_row=int(rows[worst[0]]),
                values_at_worst=[float(actual[worst]),float(expected[worst])],
                failed_coordinates=int(bad.sum()),failed_rows=rows[np.any(bad,axis=1)].tolist(),
                error_quantiles=np.quantile(error,[.5,.9,.99,1]).tolist())


def main():
    d,people,source,query=load();old=np.load(OLD/'source_banks.npz')
    oldrows=np.r_[old['source_rows'],old['query_rows']]
    rows=np.r_[source,query]
    positions=np.array([np.flatnonzero(oldrows==row)[0] for row in rows])
    patch=np.load(FAILED/'pretrained_patch.npy',mmap_mode='r')
    actual=layouts(patch);report={}
    for layout,kind in (('pooled400','pooled'),('temporal1600','retained')):
        report[layout]=differences(actual[layout],old['pretrained_'+kind][positions],rows)
    oldcpu=np.load(OLD/'cpu_a100_embeddings.npz')
    pos=np.array([np.flatnonzero(source==row)[0] for row in oldcpu['rows']])
    cpu=layouts(oldcpu['pretrained_cpu']);oldgpu=layouts(oldcpu['pretrained_A100'])
    for layout in actual:
        report[layout+'_oldCPU']=differences(actual[layout][pos],cpu[layout],oldcpu['rows'])
        report[layout+'_oldGPU']=differences(actual[layout][pos],oldgpu[layout],oldcpu['rows'])
    current=inputs(d['X'][oldcpu['rows']]).numpy().reshape(-1,41,800)
    result=dict(scope='cached SOURCE6 and P09 feature-array diagnostic only; no new encoder inference/fitting/labels/metrics',node=platform.node(),
                failed_manifest_sha256=sha(FAILED/'source.sha256'),old_bank_sha256=sha(OLD/'source_banks.npz'),
                old_probe_sha256=sha(OLD/'cpu_a100_embeddings.npz'),pretrained_patch_sha256=sha(FAILED/'pretrained_patch.npy'),
                input_probe_max_abs=float(np.max(np.abs(current-oldcpu['input']))),
                current_source_rows_sha256=__import__('hashlib').sha256(source.tobytes()).hexdigest(),
                source_people=sorted(set(people[source])),comparisons=report)
    Path('diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
