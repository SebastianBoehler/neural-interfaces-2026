"""Only complete calibrated source matrix may enter the immutable current chain."""
import json
import os
import shutil
from pathlib import Path
import numpy as np
from config import sha,ORDERS
ASSEMBLY=Path('/mnt/lustre/work/gehler/gfh527/ni26_calibration339_setup_v1_20261117')
ARMS=('RF','RB','RV','CF','CB','CV')


def main():
    root=Path(os.environ['NI26_WS']);v=json.loads((ASSEMBLY/'validation.json').read_text())
    if v['outer_metrics_computed'] or not v['raw_FB_bitwise323'] or not v['raw_V_hashverified332']:raise ValueError('completeendpointguards')
    expected={f'ttt18{o}_n339{a.lower()}_base-all_loso-ensemble2.npz' for a in ARMS for o in ORDERS}
    if set(v['streams'])!=expected:raise ValueError('incomplete48members')
    inputs=dict(v['inputs']);inputs[str(ASSEMBLY/'validation.json')]=sha(ASSEMBLY/'validation.json');inputs[str(ASSEMBLY/'source.sha256')]=sha(ASSEMBLY/'source.sha256')
    for p,h in inputs.items():
        if sha(Path(p))!=h:raise ValueError('source/producerchanged:'+p)
    directory=root/'runs/representation_streams';target=root/'runs/representation_inputs';directory.mkdir();target.mkdir();records={}
    values={}
    for name,h in v['streams'].items():
        p=ASSEMBLY/'streams'/name
        if sha(p)!=h:raise ValueError('memberchanged')
        inputs[str(p)]=h;shutil.copyfile(p,directory/name);records[name]=h
        with np.load(p) as z:
            if not np.array_equal(z['index'],np.arange(2655)) or not np.isfinite(z['proba']).all():raise ValueError('badmember')
            values[name]=z['proba'].copy()
    for view in 'FBV':
        diffs=[]
        for o in ORDERS:
            raw=values[f'ttt18{o}_n339r{view.lower()}_base-all_loso-ensemble2.npz'];cal=values[f'ttt18{o}_n339c{view.lower()}_base-all_loso-ensemble2.npz']
            if not np.array_equal(raw.argmax(1),cal.argmax(1)):raise ValueError('temperatureargmax')
            diffs.append(float(abs(raw-cal).max()))
        if max(diffs)<=1e-7:raise ValueError('calibrationgloballyinactive:'+view)
    manifest=target/'streams.sha256';manifest.write_text(''.join(f'{h}  {n}\n' for n,h in sorted(records.items())))
    (target/'lock.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest),producer_inputs_sha256=inputs,group='339'),indent=2))
    print('48completeguardedmembers;calibrationactiveandargmaxpreserving;NOoutermetrics',flush=True)


if __name__=='__main__':main()
