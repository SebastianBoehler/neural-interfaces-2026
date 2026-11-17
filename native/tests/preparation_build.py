"""Narrow build against immutable qualified CAR and source-mean libraries."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
OLD = ROOT / 'ni26_native353_parity_v3_20261117/build'
BASE = ROOT / 'ni26_core_car_repair_v1_20261117/build/libni26_base.so'
RUNTIME = ROOT / 'ni26_native353_row_mean_repair_v1_20261117/build/libni26_runtime.so'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    build = root / 'build'
    build.mkdir()
    original = shlex.split((OLD / 'CMakeFiles/ni26_neural_source.dir/link.txt').read_text())
    flags = {}
    for line in (OLD / 'CMakeFiles/ni26_base.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1)
            flags[key] = shlex.split(value)
    objects, commands = [], []
    for relative in ('src/source.cpp', 'src/preparation.cpp', 'tests/preparation_source.cpp'):
        obj = build / (Path(relative).stem + '.o')
        command = [original[0], *flags['CXX_DEFINES'], '-I' + str(root / 'native/include'),
                   *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'], '-c',
                   str(root / 'native' / relative), '-o', str(obj)]
        subprocess.run(command, check=True)
        objects.append(str(obj))
        commands.append(command)
    link, replaced = [], 0
    for token in original:
        if token.endswith('/neural_source.cpp.o'):
            link.extend(objects)
            replaced += 1
        elif token == 'libni26_base.so':
            link.append(str(BASE))
        elif token == 'libni26_runtime.so':
            link.append(str(RUNTIME))
        else:
            link.append(token.replace(str(OLD), str(BASE.parent) + ':' + str(RUNTIME.parent)))
    if replaced != 1 or link.count('-o') != 1:
        raise ValueError('unexpected production link command')
    executable = build / 'ni26_preparation_source'
    link[link.index('-o') + 1] = str(executable)
    subprocess.run(link, cwd=build, check=True)
    dependencies = subprocess.check_output(['ldd', str(executable)], text=True)
    print(dependencies, flush=True)
    if 'not found' in dependencies or str(BASE) not in dependencies or str(RUNTIME) not in dependencies:
        raise ValueError('source preparation does not resolve exact qualified libraries')
    (root / 'build_proof.json').write_text(json.dumps(dict(compile_commands=commands, link_command=link,
        base_sha256=sha(BASE), runtime_sha256=sha(RUNTIME), executable_sha256=sha(executable)), indent=2))


if __name__ == '__main__':
    main()
