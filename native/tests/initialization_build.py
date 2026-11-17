"""Build two small initializer TUs; require the qualified base at actual load time."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
OLD = ROOT / 'ni26_native353_parity_v3_20261117/build'
BASE = ROOT / 'ni26_core_car_repair_v1_20261117/build/libni26_base.so'
BASE_SHA = '70d149780e714f2763beb631804a14b5b77ed7fa2c8d974393b9f36bffe41cb5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    build = root / 'build'
    build.mkdir()
    if sha(BASE) != BASE_SHA:
        raise ValueError('qualified base changed')
    original = shlex.split((OLD / 'CMakeFiles/ni26_probe.dir/link.txt').read_text())
    flags = {}
    for line in (OLD / 'CMakeFiles/ni26_base.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1)
            flags[key] = shlex.split(value)
    objects, commands = [], []
    for relative in ('src/initialization.cpp', 'tests/initialization_source.cpp'):
        obj = build / (Path(relative).stem + '.o')
        command = [original[0], *flags['CXX_DEFINES'], '-I' + str(root / 'native/include'),
                   *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'], '-c',
                   str(root / 'native' / relative), '-o', str(obj)]
        subprocess.run(command, check=True)
        objects.append(str(obj))
        commands.append(command)
    link, replaced = [], 0
    for token in original:
        if token.endswith('/probe.cpp.o'):
            link.extend(objects)
            replaced += 1
        elif token == 'libni26_base.so':
            link.append(str(BASE))
        else:
            link.append(token.replace(str(OLD), str(BASE.parent)))
    if replaced != 1 or link.count('-o') != 1:
        raise ValueError('unexpected production link command')
    executable = build / 'ni26_initialization_source'
    link[link.index('-o') + 1] = str(executable)
    subprocess.run(link, cwd=build, check=True)
    dependencies = subprocess.check_output(['ldd', str(executable)], text=True)
    print(dependencies, flush=True)
    if 'not found' in dependencies or str(BASE) not in dependencies:
        raise ValueError('initializer does not resolve the exact qualified base')
    proof = dict(compile_commands=commands, link_command=link, base_sha256=sha(BASE),
                 objects={path: sha(path) for path in objects}, executable_sha256=sha(executable))
    (root / 'build_proof.json').write_text(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
