"""Real-source cost preflight for original TS feature recipe; no PCA/decoder score."""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from returning_assets import ROOT,sha


def main():
    source=ROOT/'ni26_js321_20261117_returning2/source';sys.path[:0]=[str(source/'experiments'),str(ROOT/'neural-interfaces-2026/2026-competition/tracks/bci_decoding')]
    import ts_core as T
    data=source/'data/proteus_windows.npz'
    with np.load(data,allow_pickle=True) as z:
        sub=z['subject'];session=z['session'].astype(str);rows=np.flatnonzero((sub=='Dreyer2026Proteus/P04')&(session=='1'))[:31]
        d=dict(X=z['X'][rows].copy(),y=z['y'][rows].copy(),stream=sub[rows]+'|'+session[rows],ch_names=z['ch_names'].copy())
    if len(rows)!=31 or d['X'].shape!=(31,41,480):raise ValueError('real source sample differs')
    cfg=T.parse('car_ov17_fs2_t0.1_le_k2');times=[]
    for repeat in range(2):
        start=time.monotonic();fe=T.Feats(d,cfg);times.append(time.monotonic()-start)
        if not np.isfinite(fe.F).all():raise ValueError('nonfinite original source features')
        shape=fe.F.shape;blocks=len(fe.C);del fe
    report=dict(status='original_TS_source_cost_preflight_complete',source_rows=rows.tolist(),samples=31,features_shape=shape,
                covariance_blocks=blocks,seconds=times,linear_full2655_feature_seconds=max(times)*2655/31,
                maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                PCA_cost_UNMEASURED=True,scope='cost-only31realtrainingwindows;notfullfold/score/parity/qualification',
                original_module=str(T.__file__),module_sha256=sha(T.__file__),source_sha256=sha('source.sha256'),accuracy_computed=False)
    Path('cost.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
