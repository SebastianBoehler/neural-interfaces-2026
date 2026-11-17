"""Compile three observation-only graph TUs against qualified fixed libraries."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

BASE=Path('/mnt/lustre/work/gehler/gfh527')
OLD=BASE/'ni26_native353_parity_v3_20261117/build'
RUNTIME=BASE/'ni26_native353_ft_repair_v1_20261117/build'
INIT=BASE/'ni26_native353_initialization_v1_20261117/build/initialization.o'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path.cwd();build=root/'build';build.mkdir(exist_ok=False)
    flags={line.split(' = ',1)[0]:shlex.split(line.split(' = ',1)[1]) for line in (OLD/'CMakeFiles/ni26_neural_source.dir/flags.make').read_text().splitlines() if ' = ' in line}
    original=shlex.split((OLD/'CMakeFiles/ni26_neural_source.dir/link.txt').read_text())
    sources=['src/training_fbc_observed.cpp','src/session_bn_observed.cpp','tests/neural_source_stages.cpp']
    commands=[[original[0],*flags['CXX_DEFINES'],'-I'+str(root/'native/include'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(root/'native'/source),'-o',str(build/(Path(source).stem+'.o'))] for source in sources]
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda command:subprocess.run(command,check=True),commands))
    command=original.copy();command[command.index('-o')+1]=str(build/'source_stages');replaced=0
    for i,arg in enumerate(command):
        if arg.endswith('/tests/neural_source.cpp.o'):
            command[i:i+1]=[str(build/(Path(source).stem+'.o')) for source in sources]+[str(INIT)];replaced+=1;break
    if replaced!=1:raise ValueError('original test link object differs')
    for i,arg in enumerate(command):
        if arg in ('libni26_runtime.so','libni26_base.so'):command[i]=str(RUNTIME/arg)
        elif arg.startswith('-Wl,-rpath,'):
            paths=[p for p in arg[len('-Wl,-rpath,'):].split(':') if p not in (str(OLD),str(RUNTIME))];command[i]='-Wl,-rpath,'+':'.join([str(RUNTIME),*paths])
    subprocess.run(command,cwd=OLD,check=True)
    loaded=subprocess.run(['ldd',str(build/'source_stages')],capture_output=True,text=True,check=True).stdout
    if 'not found' in loaded:raise ValueError('missing source-stage dependency')
    for library in ('libni26_runtime.so','libni26_base.so'):
        if [line.split('=>',1)[1].split()[0] for line in loaded.splitlines() if line.strip().startswith(library+' =>')]!=[str(RUNTIME/library)]:raise ValueError('wrong source-stage actual library:'+library)
    symbols=subprocess.run(['nm','--defined-only','-C',str(build/'source_stages')],capture_output=True,text=True,check=True).stdout
    active=[line for line in symbols.splitlines() if ' T ' in line and any(name in line for name in ('ni26::neural::SessionBN::forward','ni26::neural::RawFBC::forward'))]
    if len(active)!=2:raise ValueError('observed native source graph is not actual consumer')
    (root/'worker_ldd.txt').write_text(loaded)
    (root/'build_proof.json').write_text(json.dumps(dict(compile_commands=commands,link_command=command,ldd=loaded,direct_consumer_symbols=active,executable_sha256=sha(build/'source_stages'),initializer_object_sha256=sha(INIT),optimizer_updates=0),indent=2))


if __name__=='__main__':main()
