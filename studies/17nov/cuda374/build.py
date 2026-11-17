"""Rebuild only NetBank against the objects behind the qualified worker."""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

base = Path('/mnt/lustre/work/gehler/gfh527')
original = base / 'ni26_native360_package_v1_20261117'
qualified = base / 'ni26_native366_exact_ea_v1_20261117'
release = base / 'ni26_native366_exact_release_v1_20261117/package'
old = base / 'ni26_native366_copy_v1_20261117'
cfg = original / 'build/CMakeFiles/ni26_runtime.dir'
source = (original / 'native/src/networks.cpp').read_text()
for before, after in (
    ('      finite(p, "neural posterior");\n', ''),
    ('probs.push_back(p[0].cpu());', 'probs.push_back(p[0]);'),
    ('result.probabilities = torch::stack(probs);',
     'result.probabilities = torch::stack(probs).cpu();\n'
     '  finite(result.probabilities, "neural posterior");'),
):
    if source.count(before) != 1:
        raise ValueError(f'NetBank base changed: {before}')
    source = source.replace(before, after)
Path('networks.cpp').write_text(source)
settings = dict(line.split(' = ', 1) for line in (cfg / 'flags.make').read_text().splitlines()
                if ' = ' in line)
flags = sum((shlex.split(settings[k]) for k in ('CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS')), [])
link = shlex.split((cfg / 'link.txt').read_text())
obj = Path.cwd() / 'networks.o'
subprocess.run([link[0], '-I' + str(original / 'native/src'), *flags, '-c', 'networks.cpp',
                '-o', str(obj)], check=True)
for name, new in {'networks': obj, 'finetune': old / 'finetune_probe.o',
                  'networks_extras': old / 'extras.o', 'networks_input': qualified / 'networks_input.o'}.items():
    token = f'CMakeFiles/ni26_runtime.dir/src/{name}.cpp.o'
    if link.count(token) != 1:
        raise ValueError(f'missing runtime object {name}')
    link[link.index(token)] = str(new)
runtime = 'libni26_runtime.so'
out = Path.cwd() / 'candidate' / runtime
shutil.copytree(release, out.parent, copy_function=os.link, ignore=shutil.ignore_patterns(runtime))
link[link.index('-o') + 1] = str(out)
link = ['-Wl,-rpath,$ORIGIN' if v.startswith('-Wl,-rpath,') else v for v in link]
subprocess.run([*link, '-L' + str(out.parent), '-lea_probe'], cwd=original / 'build', check=True)
def deps(path):
    elf = subprocess.check_output(['readelf', '-d', path], text=True)
    return re.findall(r'\((?:RPATH|RUNPATH)\).*\[(.*?)\]', elf), set(re.findall(r'\(NEEDED\).*\[(.*?)\]', elf))

rpath, _ = deps(out)
if rpath == ['$ORIGIN:' + str(base / 'flightrl-puffer5.pmbvox/toolchain/lib')]:
    subprocess.run(['cmake', f'-DLIBRARY={out}', f'-DOLD_RPATH={rpath[0]}', '-P',
                    str(original / 'normalize_rpath.cmake')], check=True)
if deps(out) != deps(release / runtime):
    raise ValueError('worker ELF dependency contract changed')
shutil.copytree(qualified / 'stress_native', 'candidate_stress', copy_function=os.link,
                ignore=shutil.ignore_patterns(runtime))
os.link(out, Path('candidate_stress') / runtime)
