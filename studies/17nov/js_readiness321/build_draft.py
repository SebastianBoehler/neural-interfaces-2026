"""Generate one-file copy-only JS draft from exact frozen v66 sources and real six-seed traces."""
import hashlib
import importlib.util
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from reference import sha

RECOVERY=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_inputrepair1')
ZIP_SHA='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed'
MODULES=('numerics','tangent','inputs','nets','stages','gates','personal','routing','ensemble',
         'extras','finetune','countfilter','cpers','v6chain','solver')


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('exact pinned source anchor differs: '+old)
    return text.replace(old,new,1)


def main():
    ws=Path.cwd()
    if sha(ws/'base_v66.zip')!=ZIP_SHA:raise ValueError('original ZIP changed')
    recovery=json.loads((RECOVERY/'recovery.json').read_text())
    if recovery['status']!='six_seed_historical_and_export_guards_passed' or [r['seed'] for r in recovery['models']]!=list(range(6)):
        raise ValueError('six real source/seed reconstructions required')
    for r in recovery['models']:
        if sha(r['checkpoint'])!=r['checkpoint_sha256'] or sha(r['trace'])!=r['trace_sha256']:
            raise ValueError('recovered checkpoint/trace changed')
    spec=importlib.util.spec_from_file_location('owning_pinned_bundle',ws/'bundle.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    builder.SRC=ws/'source66';builder.MODULES=MODULES
    with zipfile.ZipFile(ws/'base_v66.zip') as archive:
        original=archive.read('submission.py').decode()
        if builder.bundle()!=original:raise ValueError('copied source tree does not reproduce frozen bundle exactly')
        with np.load(io.BytesIO(archive.read('weights.npz')),allow_pickle=False) as z:
            weights={k:z[k].copy(order='K') for k in z.files}
        names=archive.namelist()
        if any(Path(n).name!=n for n in names):raise ValueError('non-root package file')
        for folder in ('draft_off','draft_jsall'):
            target=ws/folder;target.mkdir(exist_ok=False);archive.extractall(target)
    source=ws/'draft_src';shutil.copytree(ws/'source66',source)
    shutil.copy2(ws/'adapter.py',source/'adapter.py')
    path=source/'solver.py';path.chmod(0o644);text=path.read_text()
    text=once(text,'from .v6chain import Ensemble6','from .adapter import EnsembleJS')
    text=once(text,'model = Ensemble6(z,','model = EnsembleJS(z,');path.write_text(text)
    builder.SRC=source;builder.MODULES=MODULES[:-1]+('adapter','solver');bundle=builder.bundle()
    with np.load(RECOVERY/'held_input.npz',allow_pickle=False) as z:
        inputs={k:z[k].copy() for k in z.files}
    if sha(RECOVERY/'held_input.npz')!=recovery['held_input_sha256']:
        raise ValueError('recovered source input prior changed')
    records={}
    for on,folder in ((0,'draft_off'),(1,'draft_jsall')):
        target=ws/folder;new=dict(weights,js_on=np.int64(on))
        if on:
            files=[]
            for r in recovery['models']:
                f=Path(r['trace']);shutil.copy2(f,target/f.name);files.append(f.name)
            new.update(js_cc_files=np.array(files),js_seeds=np.arange(6),**{'js_'+k:v for k,v in inputs.items()})
        if any(not np.array_equal(new[k],v) or new[k].dtype!=v.dtype or new[k].strides!=v.strides for k,v in weights.items()):
            raise ValueError('an original package weight/key changed')
        (target/'submission.py').write_text(bundle);np.savez(target/'weights.npz',**new)
        with np.load(target/'weights.npz',allow_pickle=False) as saved:
            if any(saved[k].strides!=v.strides for k,v in weights.items()):
                raise ValueError('original serialized array memory layout changed')
        for name in names:
            if name not in ('submission.py','weights.npz'):
                with zipfile.ZipFile(ws/'base_v66.zip') as archive:
                    if sha(target/name)!=hashlib.sha256(archive.read(name)).hexdigest():
                        raise ValueError('original member file changed')
        records[folder]={p.name:sha(p) for p in target.iterdir()}
    report=dict(status='generated_two_drafts',original_zip=ZIP_SHA,recovery_sha256=sha(RECOVERY/'recovery.json'),
                source_sha256=sha(ws/'source.sha256'),bundle_sha256=hashlib.sha256(bundle.encode()).hexdigest(),
                original_keys_unchanged=True,files=records,policy='JS-all;continues after recognition;not a promoted package')
    (ws/'build.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
