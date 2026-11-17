"""Compile only two diagnostic TUs against the unchanged current neural runtime."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path.cwd();build=root/'build';build.mkdir(exist_ok=False)
    old=Path('/mnt/lustre/work/gehler/gfh527/ni26_native353_parity_v3_20261117/build')
    current=Path('/mnt/lustre/work/gehler/gfh527/ni26_native353_row_mean_repair_v1_20261117/build')
    lines=(old/'CMakeFiles/ni26_neural_source.dir/flags.make').read_text().splitlines();flags=[]
    for key in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'):
        values=[line.split('=',1)[1].strip() for line in lines if line.startswith(key+' =')]
        if len(values)!=1:raise ValueError('original compile flags changed')
        flags+=shlex.split(values[0])
    compiler='/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++'
    targets=[('neural_collect','neural_source.cpp'),('raw_refit_trace','finetune_first_refit.cpp')]
    commands=[[compiler,*flags,'-c',str(root/'tests'/source),'-o',str(build/(name+'.o'))] for name,source in targets]
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda command:subprocess.run(command,check=True),commands))
    links=[];loaded={}
    for name,_ in targets:
        command=shlex.split((old/'CMakeFiles/ni26_neural_source.dir/link.txt').read_text());count=0
        command[command.index('-o')+1]=str(build/name)
        for i,arg in enumerate(command):
            if arg.endswith('ni26_neural_source.dir/tests/neural_source.cpp.o'):command[i]=str(build/(name+'.o'));count+=1
            elif arg in ('libni26_runtime.so','libni26_base.so'):command[i]=str(current/arg)
            elif arg.startswith('-Wl,-rpath,'):
                paths=[p for p in arg[len('-Wl,-rpath,'):].split(':') if p not in (str(old),str(current))]
                command[i]='-Wl,-rpath,'+':'.join([str(current),*paths])
        if count!=1:raise ValueError('original test object substitution differs')
        subprocess.run(command,cwd=old,check=True);links.append(command)
        result=subprocess.run(['ldd',str(build/name)],capture_output=True,text=True,check=True).stdout
        if 'not found' in result:raise ValueError('missing diagnostic dependency')
        for library in ('libni26_runtime.so','libni26_base.so'):
            values=[line.split('=>',1)[1].split()[0] for line in result.splitlines() if line.strip().startswith(library+' =>')]
            if values!=[str(current/library)]:raise ValueError('wrong actual diagnostic library:'+str(values))
        loaded[name]=result
    (root/'worker_ldd.txt').write_text(loaded['neural_collect'])
    (root/'build_proof.json').write_text(json.dumps(dict(compile_commands=commands,link_commands=links,ldd=loaded,
        runtime_sha256=sha(current/'libni26_runtime.so'),base_sha256=sha(current/'libni26_base.so'),no_library_rebuild=True),indent=2))


if __name__=='__main__':main()
