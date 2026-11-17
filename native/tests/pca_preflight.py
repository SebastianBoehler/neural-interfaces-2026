"""Complete read-only PCA dependency/schema/ELF/thread/output preflight, no SVD."""
import ast
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_info,threadpool_limits
import pca_fixture as fixture
import pca_reference as reference
import convert_package


def header(path,key):
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        readers={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}
        if version not in readers:raise ValueError('unsupported NPY format')
        shape,order,dtype=readers[version](stream)
    return dict(shape=list(shape),dtype=str(dtype),fortran=bool(order))


def main():
    if len(sys.argv)!=1:raise ValueError('run from the final snapshot root, with no arguments')
    root=Path.cwd();out=root/'preflight'
    if out.exists() or any((root/name).exists() for name in ('common.pt','common.json','comparison')):
        raise ValueError('all preflight/test outputs must be new')
    if not os.access(root,os.W_OK) or shutil.disk_usage(root).free<4*1024**3:
        raise ValueError('PCA snapshot parent needs write access and >=4GiB free')
    expected={fixture:root/'native/tests/pca_fixture.py',reference:root/'native/tests/pca_reference.py',
              convert_package:root/'native/tools/convert_package.py'}
    for module,path in expected.items():
        if Path(module.__file__).resolve()!=path.resolve():raise ValueError('wrong helper owner')
    for path,digest in fixture.PINS.items():
        if fixture.sha(path)!=digest:raise ValueError('existing source/reference changed: '+str(path))
    if fixture.sha(reference.OWNER)!=reference.OWNER_SHA:raise ValueError('original pc_basis owner changed')
    body=[n for n in ast.parse(reference.OWNER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='pc_basis']
    if len(body)!=1:raise ValueError('original PCA function missing')
    namespace=dict(np=np,svd=reference.svd,svd_flip=reference.svd_flip)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(reference.OWNER),'exec'),namespace)
    fields=torch.load(fixture.FEATURES,weights_only=True,mmap=True)
    features,rows=fields['features'],fields['rows']
    if features.shape!=(2278,44239) or features.dtype!=torch.float64 or not features.is_contiguous() or features.requires_grad:
        raise ValueError('existing source feature matrix schema differs')
    if rows.dtype!=torch.int64 or rows.shape!=(2278,):raise ValueError('source row schema differs')
    row_sha=hashlib.sha256(rows.numpy().astype('<i8').tobytes()).hexdigest()
    if row_sha!='ca4a869ff981142b692b2a4e2ecb44f075ea2494e398280f103c9a9a36bcea2b':
        raise ValueError('PCA source row order differs')
    shapes=dict(ts_mu=(44239,),ts_sd=(44239,),idx_tr=(2278,),tsem_m=(44239,),
                tsem_V=(44239,128),route_V_hi=(44239,128),singular_values=(2278,),pc_tr=(2278,256))
    headers={}
    with np.load(fixture.REFERENCE,allow_pickle=False) as archive:
        for key,shape in shapes.items():
            info=header(fixture.REFERENCE,key);headers[key]=info
            dtype='int64' if key=='idx_tr' else 'float32' if key in ('tsem_V','route_V_hi','pc_tr') else 'float64'
            if info['shape']!=list(shape) or info['dtype']!=dtype:raise ValueError('reference schema differs: '+key)
        if not np.array_equal(archive['idx_tr'],rows.numpy()):raise ValueError('source indices differ')
    provider=fixture.ROOT/'neural-interfaces-2026/.venv/lib/python3.12/site-packages/scipy.libs/libscipy_openblas-5f890258.so'
    provider_sha=fixture.sha(provider)
    if provider_sha!='2a5b4c1a6311873c59132b2ceaaf3c26258a17350b7cd27ea8782a47e3322981':
        raise ValueError('qualified PCA provider changed')
    lib=ctypes.CDLL(str(provider));get_threads=lib.scipy_openblas_get_num_threads
    get_threads.argtypes=[];get_threads.restype=ctypes.c_int
    torch.set_num_threads(6)
    with threadpool_limits(limits=6):
        pools=threadpool_info()
        if torch.get_num_threads()!=6 or get_threads()!=6 or any(p['num_threads']!=6 for p in pools):
            raise ValueError('actual six-thread contexts failed')
    base=fixture.ROOT/'ni26_core_car_repair_v1_20261117/build/libni26_base.so'
    exe=root/'build/ni26_pca_source'
    dependencies=subprocess.check_output(['ldd',str(exe)],text=True)
    dynamic=subprocess.check_output(['readelf','-d',str(exe)],text=True)
    if 'not found' in dependencies or str(base) not in dependencies or str(provider) not in dependencies:
        raise ValueError('wrong or unresolved native PCA executable dependency')
    if str(fixture.ROOT/'ni26_native353_parity_v3_20261117/build') in dynamic:
        raise ValueError('old base path survived in executable RPATH')
    build=json.loads((root/'build_proof.json').read_text())
    if build['base_sha256']!=fixture.sha(base) or build['executable_sha256']!=fixture.sha(exe):
        raise ValueError('built PCA binary/base changed')
    if build['base_sha256']!='70d149780e714f2763beb631804a14b5b77ed7fa2c8d974393b9f36bffe41cb5':
        raise ValueError('unqualified base')
    modules={}
    for name in sorted(sys.modules):
        path=getattr(sys.modules[name],'__file__',None)
        if path and Path(path).is_file():modules[name]=dict(path=str(Path(path).resolve()),sha256=fixture.sha(Path(path).resolve()))
        else:modules[name]=dict(path=path,kind='builtin/namespace')
    out.mkdir();module_path=out/'imported_modules.json';module_path.write_text(json.dumps(modules,indent=2))
    record=dict(status='PCA_COMPLETE_READ_ONLY_PREFLIGHT_PASS',fixture_sha256=fixture.sha(Path(fixture.__file__)),
        reference_helper_sha256=fixture.sha(Path(reference.__file__)),preflight_sha256=fixture.sha(Path(__file__)),
        source_rows_sha256=row_sha,source_feature_shape=list(features.shape),source_feature_dtype=str(features.dtype),
        source_rows=2278,width=44239,reference_headers=headers,owner_sha256=reference.OWNER_SHA,
        function_imports=dict(svd=inspect.getfile(inspect.unwrap(reference.svd)),svd_flip=inspect.getfile(reference.svd_flip)),
        pinned_inputs={str(p):s for p,s in fixture.PINS.items()},provider_path=str(provider),provider_sha256=provider_sha,
        executable_sha256=fixture.sha(exe),base_sha256=fixture.sha(base),threadpools=pools,native_provider_threads=6,
        imported_module_count=len(modules),imported_modules_sha256=fixture.sha(module_path),
        python_executable=str(Path(sys.executable).resolve()),python_executable_sha256=fixture.sha(Path(sys.executable).resolve()),
        setup_output='common.pt',comparison_output='comparison',outputs_absent=True,
        features_mapped_not_computed=True,standardization_run=False,SVD_run=False,head_fit=False,
        query_EEG_used=False,query_labels_opened=False,accuracy_computed=False)
    (out/'proof.json').write_text(json.dumps(record,indent=2));(out/'ldd.txt').write_text(dependencies)
    (out/'readelf.txt').write_text(dynamic);print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
