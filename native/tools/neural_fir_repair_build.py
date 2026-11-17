"""Conditional one-TU repair only after existing-spectrum causal control passes."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

BASE=Path('/mnt/lustre/work/gehler/gfh527')
OLD=BASE/'ni26_native353_parity_v3_20261117/build'
GRAPH=BASE/'ni26_native353_training_graph_v1_20261117'
RUNTIME=BASE/'ni26_native353_ft_repair_v1_20261117/build'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path.cwd();control=json.loads((root/'arithmetic.json').read_text())
    if not all(control.get(key) for key in ('original_and_native_reproduced','same_shape_counterfactual_exact','layout_product_changes')):raise ValueError('causal arithmetic proof absent; no candidate compilation/update allowed')
    build=root/'build';build.mkdir(exist_ok=False)
    source=root/'native/src/training_fbc.cpp';text=source.read_text();old='spectra_.push_back(at::fft_rfft(f, 480 + f.numel() - 1));'
    if text.count(old)!=1:raise ValueError('actual production FIR spectrum anchor changed')
    candidate=source.with_name('training_fbc_candidate.cpp')
    candidate.write_text(text.replace(old,'spectra_.push_back(at::fft_rfft(f, 480 + f.numel() - 1).view({1, 1, -1}).repeat({1, 41, 1}));'))
    flags={line.split(' = ',1)[0]:shlex.split(line.split(' = ',1)[1]) for line in (OLD/'CMakeFiles/ni26_neural_source.dir/flags.make').read_text().splitlines() if ' = ' in line}
    proof=json.loads((GRAPH/'build_proof.json').read_text());compiler=proof['link_command'][0]
    compile_command=[compiler,*flags['CXX_DEFINES'],'-I'+str(root/'native/include'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(candidate),'-o',str(build/'training_fbc.o')]
    subprocess.run(compile_command,check=True)
    command=proof['link_command'].copy();command[command.index('-o')+1]=str(build/'neural_training_source');count=0
    for i,arg in enumerate(command):
        if arg==str(OLD/'CMakeFiles/ni26_runtime.dir/src/training_fbc.cpp.o'):command[i]=str(build/'training_fbc.o');count+=1
    if count!=1:raise ValueError('qualified original RawFBC object substitution differs')
    subprocess.run(command,cwd=OLD,check=True)
    loaded=subprocess.run(['ldd',str(build/'neural_training_source')],capture_output=True,text=True,check=True).stdout
    if 'not found' in loaded:raise ValueError('missing FIR repair dependency')
    for library in ('libni26_runtime.so','libni26_base.so'):
        if [line.split('=>',1)[1].split()[0] for line in loaded.splitlines() if line.strip().startswith(library+' =>')]!=[str(RUNTIME/library)]:raise ValueError('wrong actual library:'+library)
    symbols=subprocess.run(['nm','--defined-only','-C',str(build/'neural_training_source')],capture_output=True,text=True,check=True).stdout
    active=[line for line in symbols.splitlines() if ' T ' in line and any(name in line for name in ('ni26::neural::RawFBC::forward','ni26::neural::SessionBN::forward'))]
    if len(active)!=2:raise ValueError('candidate RawFBC/SBN not direct active consumers')
    (root/'worker_ldd.txt').write_text(loaded)
    (root/'candidate.sha256').write_text(f'{sha(candidate)}  {candidate}\n')
    (root/'build_proof.json').write_text(json.dumps(dict(compile_command=compile_command,link_command=command,ldd=loaded,direct_consumer_symbols=active,candidate_sha256=sha(candidate),production_base_sha256=sha(source),executable_sha256=sha(build/'neural_training_source'),only_RawFBC_translation_unit_rebuilt=True,arithmetic_guard_passed_before_compilation=True,no_AdamW_change=True),indent=2))


if __name__=='__main__':main()
