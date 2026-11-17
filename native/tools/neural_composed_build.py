"""Compose qualified source/init objects with native raw-FBC training entry."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

BASE=Path('/mnt/lustre/work/gehler/gfh527')
OLD=BASE/'ni26_native353_parity_v3_20261117/build'
RUNTIME=BASE/'ni26_native353_ft_repair_v1_20261117/build'
PREP=BASE/'ni26_native353_preparation_v1_20261117/build'
INIT=BASE/'ni26_native353_initialization_v1_20261117/build'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path.cwd();build=root/'build';build.mkdir(exist_ok=False)
    flags={line.split(' = ',1)[0]:shlex.split(line.split(' = ',1)[1]) for line in (OLD/'CMakeFiles/ni26_neural_source.dir/flags.make').read_text().splitlines() if ' = ' in line}
    original=shlex.split((OLD/'CMakeFiles/ni26_neural_source.dir/link.txt').read_text());compiler=original[0]
    sources=['src/training.cpp','src/training_source.cpp','tests/neural_training_source.cpp',
             'src/sampling.cpp','src/sampling_rng.cpp','src/sampling_choice.cpp']
    commands=[[compiler,*flags['CXX_DEFINES'],'-I'+str(root/'native/include'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(root/'native'/source),'-o',str(build/(Path(source).stem+'.o'))] for source in sources]
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda command:subprocess.run(command,check=True),commands))
    reused=[PREP/'source.o',PREP/'preparation.o',INIT/'initialization.o']
    objects=[build/(Path(source).stem+'.o') for source in sources]+reused
    command=original.copy();command[command.index('-o')+1]=str(build/'neural_training_source');replaced=0
    for i,arg in enumerate(command):
        if arg.endswith('/tests/neural_source.cpp.o'):command[i:i+1]=map(str,objects);replaced+=1;break
    if replaced!=1:raise ValueError('original test link object differs')
    for i,arg in enumerate(command):
        if arg in ('libni26_runtime.so','libni26_base.so'):command[i]=str(RUNTIME/arg)
        elif arg.startswith('-Wl,-rpath,'):
            paths=[p for p in arg[len('-Wl,-rpath,'):].split(':') if p not in (str(OLD),str(RUNTIME))]
            command[i]='-Wl,-rpath,'+':'.join([str(RUNTIME),*paths])
    subprocess.run(command,cwd=OLD,check=True)
    loaded=subprocess.run(['ldd',str(build/'neural_training_source')],capture_output=True,text=True,check=True).stdout
    if 'not found' in loaded:raise ValueError('missing source-test dependency')
    for library in ('libni26_runtime.so','libni26_base.so'):
        values=[line.split('=>',1)[1].split()[0] for line in loaded.splitlines() if line.strip().startswith(library+' =>')]
        if values!=[str(RUNTIME/library)]:raise ValueError('wrong actual library:'+str(values))
    (root/'worker_ldd.txt').write_text(loaded)
    (root/'build_proof.json').write_text(json.dumps(dict(compile_commands=commands,link_command=command,ldd=loaded,reused_objects={str(p):sha(p) for p in reused},runtime_sha256=sha(RUNTIME/'libni26_runtime.so'),base_sha256=sha(RUNTIME/'libni26_base.so'),executable_sha256=sha(build/'neural_training_source')),indent=2))


if __name__=='__main__':main()
