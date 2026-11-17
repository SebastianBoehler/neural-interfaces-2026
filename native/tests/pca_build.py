"""Build only PCA module/test; reuse exact qualified CAR base and native provider."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path('/mnt/lustre/work/gehler/gfh527')
PARENT=ROOT/'ni26_native353_parity_v3_20261117'
BASE=ROOT/'ni26_core_car_repair_v1_20261117/build/libni26_base.so'
BASE_SHA='70d149780e714f2763beb631804a14b5b77ed7fa2c8d974393b9f36bffe41cb5'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path.cwd();build=root/'build';build.mkdir()
    if sha(BASE)!=BASE_SHA:raise ValueError('qualified CAR base changed')
    old=PARENT/'build';link=shlex.split((old/'CMakeFiles/ni26_probe.dir/link.txt').read_text())
    flags={}
    for line in (old/'CMakeFiles/ni26_base.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            key,value=line.split(' = ',1);flags[key]=shlex.split(value)
    sources=[root/'native/src/pca.cpp',root/'native/tests/pca_source.cpp']
    commands=[];objects=[]
    for source in sources:
        obj=build/(source.stem+'.o');objects.append(obj)
        commands.append([link[0],*flags['CXX_DEFINES'],'-I'+str(root/'native/include'),
                         *flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-o',str(obj),'-c',str(source)])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda command:subprocess.run(command,check=True),commands))
    result=[]
    for token in link:
        if token.endswith('/probe.cpp.o'):result.extend(map(str,objects))
        elif token=='libni26_base.so':result.append(str(BASE))
        else:result.append(token.replace(str(old),str(BASE.parent)))
    result[result.index('-o')+1]='ni26_pca_source';subprocess.run(result,cwd=build,check=True)
    dependencies=subprocess.check_output(['ldd',str(build/'ni26_pca_source')],text=True)
    if 'not found' in dependencies or str(BASE) not in dependencies:raise ValueError(dependencies)
    (root/'preflight_ldd.txt').write_text(dependencies)
    proof=dict(status='PCA_BUILD_LINK_LDD_PASS_NO_NUMERICAL_EXECUTION',base_sha256=sha(BASE),
        sources={str(p.relative_to(root)):sha(p) for p in sources},compile_commands=commands,
        link_command=result,objects={str(p.name):sha(p) for p in objects},
        executable_sha256=sha(build/'ni26_pca_source'),base_rebuilt=False)
    (root/'build_proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
