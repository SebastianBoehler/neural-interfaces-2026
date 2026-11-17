"""Build only private copies; keep every original source and model file unchanged."""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

from returning_assets import ROOT,sha

PARENT=ROOT/'ni26_js321_20261117_inputreuse2'
BUNDLE=ROOT/'ni26_js321_20261117_returning_bundle2'


def main():
    spec=importlib.util.spec_from_file_location('policy350_bundler',PARENT/'bundle.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    source=Path('policy_src');shutil.copytree(PARENT/'shared_src',source)
    shutil.copyfile('adapter_policy350.py',source/'adapter_policy350.py')
    solver=source/'solver.py';solver.chmod(0o644)
    text=solver.read_text()
    for before,after in [('from .adapter_inputreuse import ReuseEnsemble','from .adapter_policy350 import RecognitionEnsemble'),('model = ReuseEnsemble(z,','model = RecognitionEnsemble(z,')]:
        if text.count(before)!=1:raise ValueError('exact original constructor/import required')
        text=text.replace(before,after)
    solver.write_text(text)
    builder.SRC=source
    builder.MODULES=('numerics','tangent','inputs','nets','stages','gates','personal','routing','ensemble','extras','finetune','countfilter','cpers','v6chain','adapter_shared','adapter_inputreuse','adapter_policy350','solver')
    compiled=builder.bundle();compile(compiled,'private_policy350','exec');records=[]
    for fold in ('P04_s23','P10_s23'):
        origin=BUNDLE/fold/'js'
        before={p.name:sha(p) for p in origin.iterdir()}
        for mode in (0,1,2):
            target=Path('bundles')/fold/str(mode);shutil.copytree(origin,target)
            path=target/'submission.py';path.chmod(0o644);path.write_text(compiled)
            with np.load(origin/'weights.npz') as z:weights={k:z[k].copy(order='K') for k in z.files}
            weights['recognition_policy350']=np.int64(mode);weights['js_on']=np.int64(mode!=0)
            path=target/'weights.npz';path.chmod(0o644);np.savez(path,**weights)
            with np.load(path) as z:
                for k,v in weights.items():
                    if z[k].dtype!=v.dtype or z[k].shape!=v.shape or z[k].tobytes(order='A')!=v.tobytes(order='A') or z[k].flags.f_contiguous!=v.flags.f_contiguous:raise ValueError('K-order copied arrays changed')
            records.append(dict(fold=fold,mode=mode,files={p.name:sha(p) for p in target.iterdir()}))
        if before!={p.name:sha(p) for p in origin.iterdir()}:raise ValueError('original bundle changed')
    Path('build.json').write_text(json.dumps(dict(status='private_policy350_built_NOT_validated',source_sha256=sha('source.sha256'),original_bundle_sha256=sha(BUNDLE/'bundle_complete.json'),artifacts=records),indent=2))


if __name__=='__main__':main()
