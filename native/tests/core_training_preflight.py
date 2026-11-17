"""Read-only fixture dependency/schema audit. No EEG/features/fits/accuracy execute."""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits


def header(archive, key):
    with zipfile.ZipFile(archive) as source, source.open(key + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError('unsupported NPY header version: ' + str(version))
        shape, fortran, dtype = readers[version](stream)
    return dict(shape=list(shape), dtype=str(dtype), fortran=bool(fortran))


def main():
    if len(sys.argv) != 2:
        raise ValueError('usage: core_training_preflight.py NEW_PREFLIGHT_DIRECTORY')
    out = Path(sys.argv[1])
    if out.exists() or not out.parent.is_dir() or not os.access(out.parent, os.W_OK):
        raise ValueError('new preflight directory with existing writable parent required')
    fixture = Path(__file__).with_name('core_training_fixture.py')
    namespace = runpy.run_path(str(fixture))  # main is deliberately not called
    sha = namespace['sha']; pins = namespace['PINS']
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('pinned input differs: ' + str(path))
    # Execute the ACTUAL fixture's path initialization and reference import only.
    tree = ast.parse(fixture.read_text())
    body = next(n.body for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    selected = []
    for node in body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) and
                isinstance(t.value, ast.Attribute) and ast.unparse(t.value) == 'sys.path' for t in node.targets):
            selected.append(node)
        if isinstance(node, ast.Import) and any(a.name == 'ts_core' for a in node.names):
            selected.append(node)
    if len(selected) != 2:
        raise ValueError('fixture reference import/path initialization changed')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(fixture), 'exec'), namespace)
    source, fit, package, root = (namespace[k] for k in ('SOURCE', 'FIT', 'PACKAGE', 'ROOT'))
    official = root / 'neural-interfaces-2026/2026-competition/tracks/bci_decoding'
    expected_owners = dict(ts_core=source/'experiments/ts_core.py', features=source/'experiments/features.py',
        protocol=source/'experiments/protocol.py', convert_package=fixture.parents[1]/'tools/convert_package.py',
        **{'benchmark_utils.metrics': official/'benchmark_utils/metrics.py'})
    owners = {}
    for name, expected in expected_owners.items():
        actual = Path(sys.modules[name].__file__).resolve()
        if actual != expected.resolve():
            raise ValueError('wrong imported owner: ' + name)
        owners[name] = dict(path=str(actual), sha256=sha(actual))
    fit_proof = fit/'fit.json'
    proof = json.loads(fit_proof.read_text())
    if proof['source_rows'] != 2278 or proof['query_labels_used'] or proof['accuracy_computed']:
        raise ValueError('original source-only proof differs')
    reference = fit/'source_view_P04_s23.npz'; data = source/'data/proteus_windows.npz'
    shapes = {'ts_bands':(17,2),'ts_log_glob':(51,41,41),'ts_mu':(44239,), 'ts_sd':(44239,),
        'ts_W':(3,44239),'ts_b':(3,), 'tsem_m':(44239,), 'tsem_V':(44239,128),
        'route_V_hi':(44239,128),'idx_tr':(2278,),'pc_tr':(2278,256),'y_tr':(2278,),
        'singular_values':(2278,)}
    for key in ('ts_k','ts_shrink','ts_slow_cutoff','ts_slow_step','ts_car','ts_nsub'):
        shapes[key] = ()
    reference_headers = {}
    with np.load(reference, allow_pickle=False) as archive:
        if set(archive.files) != set(shapes):
            raise ValueError('original reference key set differs')
        for key, shape in shapes.items():
            info = header(reference,key); reference_headers[key] = info
            dtype = 'int64' if key in ('idx_tr','y_tr','ts_slow_step','ts_car','ts_nsub') else (
                'float32' if key in ('tsem_V','route_V_hi','pc_tr') else 'float64')
            if tuple(info['shape']) != shape or info['dtype'] != dtype:
                raise ValueError('reference field schema differs: ' + key)
        ref = {key: archive[key].copy() for key in archive.files}
    rows = ref['idx_tr']; row_sha = hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()
    if row_sha != 'ca4a869ff981142b692b2a4e2ecb44f075ea2494e398280f103c9a9a36bcea2b':
        raise ValueError('source row order differs')
    data_headers = {key:header(data,key) for key in ('X','subject','session','ch_names')}
    if data_headers['X']['shape'] != [2655,41,480] or data_headers['X']['dtype'] != 'float32':
        raise ValueError('canonical raw source schema differs')
    with np.load(data, allow_pickle=False) as archive:
        subject, session, channels = archive['subject'], archive['session'].astype(str), archive['ch_names']
    held = (subject == 'Dreyer2026Proteus/P04') & (session != '1')
    if subject.shape != (2655,) or session.shape != (2655,) or channels.shape != (41,) or not np.array_equal(rows,np.flatnonzero(~held)):
        raise ValueError('metadata/full-session source exclusion differs')
    if ref['y_tr'].dtype != np.int64 or not np.array_equal(np.unique(ref['y_tr']),np.arange(3)):
        raise ValueError('existing source-only labels differ')
    native = torch.load(package/'native.pt', weights_only=True)
    recipe_shapes = dict(sos=(17,4,6),slow_sos=(2,6),bands=(17,2),car=(),nsub=(),shrink=(),
                         k=(),slow_cutoff=(),slow_step=())
    recipe = {}
    for key, shape in recipe_shapes.items():
        value = native['ts_'+key]
        expected_type = torch.int64 if key in ('car','nsub','slow_step') else torch.float64
        if tuple(value.shape) != shape or value.dtype != expected_type or not value.isfinite().all():
            raise ValueError('immutable recipe field schema differs: ' + key)
        recipe[key] = dict(shape=list(value.shape),dtype=str(value.dtype))
        if key not in ('sos','slow_sos') and not np.array_equal(value.numpy(),ref['ts_'+key]):
            raise ValueError('fixed coefficient recipe differs from original fit: ' + key)
    contexts = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.With) and
                any('threadpool_limits' in ast.unparse(i.context_expr) for i in n.items)]
    if len(contexts) != 2 or any('limits=1' not in context for context in contexts):
        raise ValueError('actual fixture one-thread feature/projection contexts differ')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if torch.get_num_threads()!=1 or any(p['num_threads']!=1 for p in pools):
            raise ValueError('one-thread context not applied')
    versions = {name:importlib.metadata.version(name) for name in
                ('numpy','scipy','scikit-learn','torch','pyriemann','joblib','threadpoolctl')}
    modules = {}
    for name in sorted(sys.modules):
        path = getattr(sys.modules[name],'__file__',None)
        if path and Path(path).is_file():
            path = Path(path).resolve()
            modules[name] = dict(path=str(path),sha256=sha(path))
        else:
            modules[name] = dict(path=path,kind='builtin/namespace')
    direct_imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    # Only inspect target/output metadata; never create fixture or read EEG/query labels.
    fixture_output = out.parent/'fixture'
    native_output = out.parent/'native_core.pt'
    if fixture_output.exists() or native_output.exists():
        raise ValueError('next actual fixture/native output must not exist')
    disk = shutil.disk_usage(out.parent)
    if disk.free < 4*1024**3:
        raise ValueError('insufficient space for explicit source/reference archives')
    out.mkdir()
    module_file = out/'imported_modules.json'; module_file.write_text(json.dumps(modules,indent=2))
    record = dict(status='COMPLETE_READ_ONLY_PREFLIGHT_PASS',fixture_sha256=sha(fixture),
        preflight_script_sha256=sha(Path(__file__)),python_executable=str(Path(sys.executable).resolve()),
        python_executable_sha256=sha(Path(sys.executable).resolve()),python_version=sys.version,package_versions=versions,
        direct_fixture_imports=direct_imports,reference_import_initialization=[ast.unparse(n) for n in selected],
        owners=owners,imported_module_count=len(modules),imported_modules_sha256=sha(module_file),
        pinned_inputs={str(p):h for p,h in pins.items()},fit_proof_sha256=sha(fit_proof),
        reference_headers=reference_headers,data_headers=data_headers,recipe_headers=recipe,
        fit_rows_sha256=row_sha,source_rows=2278,excluded_complete_session_rows=377,
        reference_source_labels_from='y_tr only',held_person_empty_reason='P04/session1 is source',
        feature_projection_thread_contexts=contexts,threadpools=pools,torch_threads=torch.get_num_threads(),
        next_fixture_output=str(fixture_output),next_native_output=str(native_output),
        output_absent=True,disk_free_bytes=disk.free,
        EEG_read=False,query_labels_opened=False,source_features_computed=False,
        Python_fit_run=False,native_fit_run=False,accuracy_computed=False)
    (out/'proof.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
