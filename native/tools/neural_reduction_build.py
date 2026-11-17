"""Relink the demonstrated row-mean repair; reuse every compatible unchanged object."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


ROOT=Path.cwd()
OLD=Path('/mnt/lustre/work/gehler/gfh527/ni26_native353_parity_v3_20261117/build')
CAR=Path('/mnt/lustre/work/gehler/gfh527/ni26_core_car_repair_v1_20261117/build/libni26_base.so')
BASE_SHA='70d149780e714f2763beb631804a14b5b77ed7fa2c8d974393b9f36bffe41cb5'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def flags(target):
    lines=(OLD/f'CMakeFiles/{target}.dir/flags.make').read_text().splitlines()
    result=[]
    for key in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'):
        matches=[line.split('=',1)[1].strip() for line in lines if line.startswith(key+' =')]
        if len(matches)!=1:raise ValueError('ambiguous original compile flags:'+key)
        result+=shlex.split(matches[0])
    return result


def main():
    if sha(CAR)!=BASE_SHA:raise ValueError('qualified CAR base identity differs')
    build=ROOT/'build';build.mkdir(exist_ok=False);shutil.copy2(CAR,build/'libni26_base.so')
    compiler='/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++'
    plans=[('ni26_runtime',ROOT/'src/networks_input.cpp',build/'networks_input.o'),
           ('ni26_neural_source',ROOT/'tests/neural_source.cpp',build/'neural_source.o')]
    commands=[]
    for target,source,output in plans:commands.append([compiler,*flags(target),'-c',str(source),'-o',str(output)])
    def compile_one(command):subprocess.run(command,check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(compile_one,commands))
    used={};link_commands=[]
    for target,output,replace_object in [('ni26_runtime',build/'libni26_runtime.so','src/networks_input.cpp.o'),
                                          ('ni26_neural_source',build/'ni26_neural_source','tests/neural_source.cpp.o')]:
        command=shlex.split((OLD/f'CMakeFiles/{target}.dir/link.txt').read_text());changed=0
        if command.count('-o')!=1:raise ValueError('original linker output differs')
        command[command.index('-o')+1]=str(output)
        for i,arg in enumerate(command):
            if arg.endswith('.o'):
                original=Path(arg) if Path(arg).is_absolute() else OLD/arg
                if arg.endswith(replace_object):command[i]=str(plans[0 if target=='ni26_runtime' else 1][2]);changed+=1
                else:command[i]=str(original);used[str(original)]=sha(original)
            elif arg=='libni26_base.so':command[i]=str(build/'libni26_base.so')
            elif arg=='libni26_runtime.so':command[i]=str(build/'libni26_runtime.so')
            elif arg.startswith('-Wl,-rpath,'):
                paths=[p for p in arg[len('-Wl,-rpath,'):].split(':') if p not in (str(OLD),str(build))]
                command[i]='-Wl,-rpath,'+':'.join([str(build),*paths])
        if changed!=1:raise ValueError('expected exactly one replaced translation unit per target')
        subprocess.run(command,cwd=OLD,check=True);link_commands.append(command)
    loaded=subprocess.run(['ldd',str(build/'ni26_neural_source')],capture_output=True,text=True,check=True).stdout
    (ROOT/'worker_ldd.txt').write_text(loaded)
    if 'not found' in loaded:raise ValueError('missing executable dependency')
    for library in ('libni26_base.so','libni26_runtime.so'):
        lines=[line.split('=>',1)[1].split()[0] for line in loaded.splitlines() if line.strip().startswith(library+' =>')]
        if lines!=[str(build/library)]:raise ValueError('actual dynamic loader did not select fresh '+library+':'+str(lines))
    if sha(build/'libni26_base.so')!=BASE_SHA:raise ValueError('copied qualified base changed')
    for path,pin in used.items():
        if sha(Path(path))!=pin:raise ValueError('unchanged object changed:'+path)
    (ROOT/'build_proof.json').write_text(json.dumps(dict(compile_commands=commands,link_commands=link_commands,
        unchanged_objects=used,qualified_base_sha256=BASE_SHA,runtime_sha256=sha(build/'libni26_runtime.so'),
        executable_sha256=sha(build/'ni26_neural_source'),loader_paths_verified=True),indent=2))


if __name__=='__main__':main()
