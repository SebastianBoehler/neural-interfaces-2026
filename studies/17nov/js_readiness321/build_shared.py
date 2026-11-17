"""Same-weight private SharedFIR artifact; original files preserved byte-for-byte."""
import importlib.util
import json
import shutil
from pathlib import Path

from build_draft import MODULES,once
from reference import sha

PARENT=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_orchestration4')
ACTIVE=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_active5')


def main(reuse=False):
    ws=Path.cwd();parent=json.loads((PARENT/'complete.json').read_text())
    if parent['status']!='six_seed_all10_complete_private_copy_prepared':raise ValueError('complete six seeds required')
    for name,digest in parent['files'].items():
        if sha(PARENT/'prepared_jsall'/name)!=digest:raise ValueError('parent artifact changed')
    spec=importlib.util.spec_from_file_location('owning_shared_builder',ws/'bundle.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    builder.SRC=ws/'source66';builder.MODULES=MODULES
    if sha(ACTIVE/'base_v66.zip')!='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed':raise ValueError('frozen original ZIP changed')
    import zipfile
    with zipfile.ZipFile(ACTIVE/'base_v66.zip') as archive:
        if builder.bundle()!=archive.read('submission.py').decode():raise ValueError('original source no longer reproduces frozen bundle')
    source=ws/'shared_src';shutil.copytree(ws/'source66',source);shutil.copy2(ws/'adapter_shared.py',source/'adapter_shared.py')
    if reuse:shutil.copy2(ws/'adapter_inputreuse.py',source/'adapter_inputreuse.py')
    path=source/'solver.py';path.chmod(0o644)
    name='ReuseEnsemble' if reuse else 'EnsembleJS';module='adapter_inputreuse' if reuse else 'adapter_shared'
    text=once(path.read_text(),'from .v6chain import Ensemble6',f'from .{module} import {name}')
    path.write_text(once(text,'model = Ensemble6(z,',f'model = {name}(z,'))
    builder.SRC=source;builder.MODULES=MODULES[:-1]+(('adapter_shared','adapter_inputreuse','solver') if reuse else ('adapter_shared','solver'));bundle=builder.bundle()
    for name,origin in (('prepared_jsall',PARENT/'prepared_jsall'),('draft_off',ACTIVE/'draft_off')):
        target=ws/name;shutil.copytree(origin,target);path=target/'submission.py';path.chmod(0o644);path.write_text(bundle)
        for file in origin.iterdir():
            if file.name!='submission.py' and sha(file)!=sha(target/file.name):raise ValueError('same-weight copy changed a parameter/file')
    report=dict(parent,parent_complete_sha256=sha(PARENT/'complete.json'),execution_variant='sharedFIR_readonly_current_original_input' if reuse else 'sharedFIR_auxiliary_bank_only',
                optimization_source_sha256=sha(ws/'source.sha256'),files={p.name:sha(p) for p in (ws/'prepared_jsall').iterdir()})
    (ws/'complete.json').write_text(json.dumps(report,indent=2));print('sameweight_private_sharedFIR_copy_built',flush=True)


if __name__=='__main__':main()
