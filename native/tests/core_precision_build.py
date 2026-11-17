"""Recompile only changed core TUs, reuse pinned v3 objects, and relink its commands."""
import concurrent.futures
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

PARENT=Path('/mnt/lustre/work/gehler/gfh527/ni26_native353_parity_v3_20261117')
CXX='/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++'
CHANGED=('math.cpp','core.cpp','core_training_window.cpp')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path.cwd();build=root/'build';build.mkdir()
    old=PARENT/'build';base=old/'CMakeFiles/ni26_base.dir'
    reused={}
    for obj in base.rglob('*.o'):
        relative=obj.relative_to(old);target=build/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(obj,target)
        if obj.name.removesuffix('.o') not in CHANGED:reused[str(relative)]=sha(obj)
    for relative in ('libni26_lbfgsb.a','CMakeFiles/ni26_probe.dir/tests/probe.cpp.o'):
        target=build/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/relative,target);reused[relative]=sha(old/relative)
    shutil.copytree(PARENT/'include',root/'include');shutil.copy2(root/'math.h',root/'include/ni26/math.h')
    flags={}
    for line in (base/'flags.make').read_text().splitlines():
        if ' = ' in line:key,value=line.split(' = ',1);flags[key]=shlex.split(value)
    compile_flags=flags['CXX_DEFINES']+['-I'+str(root/'include')]+flags['CXX_INCLUDES']+flags['CXX_FLAGS']
    commands=[]
    def compile(name):
        output=build/'CMakeFiles/ni26_base.dir/src'/f'{name}.o'
        command=[CXX,*compile_flags,'-o',str(output),'-c',str(root/name)]
        subprocess.run(command,check=True);return command
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:commands=list(pool.map(compile,CHANGED))
    car_object=build/'car_test.o'
    subprocess.run([CXX,*compile_flags,'-o',str(car_object),'-c',str(root/'core_precision_car.cpp')],check=True)
    links={}
    for target in ('ni26_base','ni26_probe'):
        command=shlex.split((old/f'CMakeFiles/{target}.dir/link.txt').read_text())
        command=[part.replace(str(old),str(build)) for part in command]
        subprocess.run(command,cwd=build,check=True);links[target]=command
    car=links['ni26_probe'].copy();car[car.index('-o')+1]='core_precision_car'
    car=[str(car_object) if token.endswith('/probe.cpp.o') else token for token in car]
    subprocess.run(car,cwd=build,check=True)
    for relative,digest in reused.items():
        if sha(build/relative)!=digest:raise ValueError('reused object changed '+relative)
    proof=dict(parent_source_manifest_sha256=sha(PARENT/'source.sha256'),reused_objects=reused,
               changed_sources={name:sha(root/name) for name in CHANGED},math_header_sha256=sha(root/'math.h'),
               compile_commands=commands,link_commands=links,new_base_sha256=sha(build/'libni26_base.so'),
               probe_object_unchanged=True,full_rebuild=False)
    (root/'build_proof.json').write_text(json.dumps(proof,indent=2))


if __name__=='__main__':main()
