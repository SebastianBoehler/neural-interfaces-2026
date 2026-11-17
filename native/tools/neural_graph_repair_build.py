"""Compile only the original multiplication graph and complete numeric observer."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

BASE=Path('/mnt/lustre/work/gehler/gfh527')
OLD=BASE/'ni26_native353_parity_v3_20261117/build'
SOURCE=BASE/'ni26_native353_training_composed_v1_20261117'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path.cwd();build=root/'build';build.mkdir(exist_ok=False)
    proof=json.loads((SOURCE/'build_proof.json').read_text())
    flags={line.split(' = ',1)[0]:shlex.split(line.split(' = ',1)[1]) for line in (OLD/'CMakeFiles/ni26_neural_source.dir/flags.make').read_text().splitlines() if ' = ' in line}
    compiler=proof['link_command'][0]
    commands=[[compiler,*flags['CXX_DEFINES'],'-I'+str(root/'native/include'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(root/'native'/source),'-o',str(build/(Path(source).stem+'.o'))] for source in ['src/session_bn.cpp','tests/neural_training_source.cpp']]
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda command:subprocess.run(command,check=True),commands))
    command=proof['link_command'].copy();command[command.index('-o')+1]=str(build/'neural_training_source');count=0
    for i,arg in enumerate(command):
        if arg==str(SOURCE/'build/neural_training_source.o'):
            command[i:i+1]=[str(build/'neural_training_source.o'),str(build/'session_bn.o'),str(OLD/'CMakeFiles/ni26_runtime.dir/src/training_fbc.cpp.o')];count+=1;break
    if count!=1:raise ValueError('original composed test object differs')
    subprocess.run(command,cwd=OLD,check=True)
    binary=build/'neural_training_source'
    loaded=subprocess.run(['ldd',str(binary)],capture_output=True,text=True,check=True).stdout
    expected=BASE/'ni26_native353_ft_repair_v1_20261117/build'
    if 'not found' in loaded:raise ValueError('missing graph-test dependency')
    for library in ('libni26_runtime.so','libni26_base.so'):
        values=[line.split('=>',1)[1].split()[0] for line in loaded.splitlines() if line.strip().startswith(library+' =>')]
        if values!=[str(expected/library)]:raise ValueError('wrong actual library:'+str(values))
    symbols=subprocess.run(['nm','--defined-only','-C',str(binary)],capture_output=True,text=True,check=True).stdout
    active=[line for line in symbols.splitlines() if ' T ' in line and any(name in line for name in ('ni26::neural::SessionBN::forward','ni26::neural::RawFBC::forward'))]
    if len(active)!=2:raise ValueError('changed SessionBN and SAME qualified RawFBC not direct executable consumers')
    (root/'worker_ldd.txt').write_text(loaded)
    (root/'build_proof.json').write_text(json.dumps(dict(compile_commands=commands,link_command=command,ldd=loaded,direct_consumer_symbols=active,executable_sha256=sha(binary),qualified_RawFBC_object_sha256=sha(OLD/'CMakeFiles/ni26_runtime.dir/src/training_fbc.cpp.o'),no_AdamW_change=True,no_new_reference_fit=True),indent=2))


if __name__=='__main__':main()
