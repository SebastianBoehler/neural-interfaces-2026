"""Compile one source-test TU; reuse the qualified CAR base and production link ABI."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

PARENT = Path('/mnt/lustre/work/gehler/gfh527/ni26_native353_parity_v3_20261117')
BASE = Path('/mnt/lustre/work/gehler/gfh527/ni26_core_car_repair_v1_20261117/build/libni26_base.so')
BASE_SHA = '70d149780e714f2763beb631804a14b5b77ed7fa2c8d974393b9f36bffe41cb5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd(); build = root / 'build'; build.mkdir()
    if sha(BASE) != BASE_SHA:
        raise ValueError('qualified CAR base changed')
    old = PARENT / 'build'
    command = shlex.split((old / 'CMakeFiles/ni26_probe.dir/link.txt').read_text())
    flags = {}
    for line in (old / 'CMakeFiles/ni26_base.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1); flags[key] = shlex.split(value)
    obj = build / 'core_training_source.o'
    compile_command = [command[0], *flags['CXX_DEFINES'], '-I' + str(root / 'native/include'),
        *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'], '-o', str(obj), '-c', str(root / 'native/tests/core_training_source.cpp')]
    subprocess.run(compile_command, check=True)
    link = [str(obj) if token.endswith('/probe.cpp.o') else
            str(BASE) if token == 'libni26_base.so' else
            token.replace(str(old), str(BASE.parent)) for token in command]
    link[link.index('-o') + 1] = 'ni26_core_training_source'
    subprocess.run(link, cwd=build, check=True)
    dependencies = subprocess.check_output(['ldd', str(build / 'ni26_core_training_source')], text=True)
    print(dependencies, flush=True)
    if 'not found' in dependencies or str(BASE) not in dependencies:
        raise ValueError('test does not resolve the exact qualified CAR base')
    proof = dict(qualified_base_sha256=sha(BASE), source_sha256=sha(root / 'native/tests/core_training_source.cpp'),
        object_sha256=sha(obj), executable_sha256=sha(build / 'ni26_core_training_source'),
        compile_command=compile_command, link_command=link, base_rebuilt=False)
    (root / 'build_proof.json').write_text(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
