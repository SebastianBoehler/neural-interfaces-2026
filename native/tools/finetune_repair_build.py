"""Relink only the FT arithmetic repair; retain qualified input/base and other objects."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


BASE=Path('/mnt/lustre/work/gehler/gfh527')
OLD=BASE/'ni26_native353_parity_v3_20261117/build'
CURRENT=BASE/'ni26_native353_row_mean_repair_v1_20261117/build'
TRACE=BASE/'ni26_native353_ft_diagnostic_v1_20261117/build'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def flags(target):
    lines=(OLD/f'CMakeFiles/{target}.dir/flags.make').read_text().splitlines();result=[]
    for key in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'):
        values=[line.split('=',1)[1].strip() for line in lines if line.startswith(key+' =')]
        if len(values)!=1:raise ValueError('original compile flags differ')
        result+=shlex.split(values[0])
    return result


def main():
    root=Path.cwd();build=root/'build';build.mkdir(exist_ok=False)
    if sha(CURRENT/'libni26_base.so')!='70d149780e714f2763beb631804a14b5b77ed7fa2c8d974393b9f36bffe41cb5':raise ValueError('qualified CAR base differs')
    shutil.copy2(CURRENT/'libni26_base.so',build/'libni26_base.so')
    compiler=BASE/'flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++'
    commands=[[str(compiler),*flags(target),'-c',str(root/source),'-o',str(build/output)] for target,source,output in
        [('ni26_runtime','src/finetune.cpp','finetune.o'),('ni26_neural_source','tests/finetune_first_refit.cpp','raw_refit_trace.o')]]
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda command:subprocess.run(command,check=True),commands))
    links=[];used={};ldds={}
    for target,name in [('ni26_runtime','libni26_runtime.so'),('ni26_neural_source','neural_collect'),('ni26_neural_source','raw_refit_trace')]:
        command=shlex.split((OLD/f'CMakeFiles/{target}.dir/link.txt').read_text());command[command.index('-o')+1]=str(build/name)
        for i,arg in enumerate(command):
            if arg.endswith('.o'):
                original=Path(arg) if Path(arg).is_absolute() else OLD/arg
                if arg.endswith('/src/finetune.cpp.o'):replacement=build/'finetune.o'
                elif arg.endswith('/src/networks_input.cpp.o'):replacement=CURRENT/'networks_input.o'
                elif target=='ni26_neural_source':replacement=TRACE/'neural_collect.o' if name=='neural_collect' else build/'raw_refit_trace.o'
                else:replacement=original
                command[i]=str(replacement)
                if replacement.parent!=build:used[str(replacement)]=sha(replacement)
            elif arg in ('libni26_runtime.so','libni26_base.so'):command[i]=str(build/arg)
            elif arg.startswith('-Wl,-rpath,'):
                paths=[p for p in arg[len('-Wl,-rpath,'):].split(':') if p not in (str(OLD),str(CURRENT),str(build))]
                command[i]='-Wl,-rpath,'+':'.join([str(build),*paths])
        subprocess.run(command,cwd=OLD,check=True);links.append(command)
        if target=='ni26_neural_source':
            result=subprocess.run(['ldd',str(build/name)],capture_output=True,text=True,check=True).stdout
            if 'not found' in result:raise ValueError('missing native dependency')
            for library in ('libni26_runtime.so','libni26_base.so'):
                values=[line.split('=>',1)[1].split()[0] for line in result.splitlines() if line.strip().startswith(library+' =>')]
                if values!=[str(build/library)]:raise ValueError('wrong actual library:'+str(values))
            ldds[name]=result
    for path,pin in used.items():
        if sha(Path(path))!=pin:raise ValueError('reused object changed:'+path)
    (root/'worker_ldd.txt').write_text(ldds['neural_collect'])
    (root/'build_proof.json').write_text(json.dumps(dict(compile_commands=commands,link_commands=links,unchanged_objects=used,ldd=ldds,
        runtime_sha256=sha(build/'libni26_runtime.so'),base_sha256=sha(build/'libni26_base.so'),only_FT_runtime_object_rebuilt=True),indent=2))


if __name__=='__main__':main()
