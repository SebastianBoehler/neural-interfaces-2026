"""Replace only the diagnostic executable's test object after FIR parity."""

import json
from pathlib import Path
import shlex
import subprocess


BASE = Path('/mnt/lustre/work/gehler/gfh527')
OLD = BASE / 'ni26_native353_parity_v3_20261117/build'
FIR = BASE / 'ni26_native353_fir_layout_v1_20261117'
RUNTIME = BASE / 'ni26_native353_ft_repair_v1_20261117/build'


def main():
    root = Path.cwd()
    build = root / 'build'
    build.mkdir(exist_ok=False)
    proof = json.loads((FIR / 'build_proof.json').read_text())
    if not proof['arithmetic_guard_passed_before_compilation']:
        raise ValueError('validated FIR link required')
    flags = OLD / 'CMakeFiles/ni26_neural_source.dir/flags.make'
    options = dict(line.split(' = ', 1) for line in flags.read_text().splitlines()
                   if ' = ' in line)
    command = proof['link_command']
    test_object = build / 'neural_training_source.o'
    compile_command = [command[0], *shlex.split(options['CXX_DEFINES']),
                       '-I' + str(root / 'native/include'),
                       *shlex.split(options['CXX_INCLUDES']), *shlex.split(options['CXX_FLAGS']),
                       '-c', str(root / 'native/tests/neural_training_source.cpp'),
                       '-o', str(test_object)]
    previous = [i for i, arg in enumerate(command) if arg.endswith('/neural_training_source.o')]
    if len(previous) != 1 or str(FIR / 'build/training_fbc.o') not in command:
        raise ValueError('qualified FIR proof has unexpected test or training objects')
    binary = build / 'neural_training_source'
    command[previous[0]] = str(test_object)
    command[command.index('-o') + 1] = str(binary)
    subprocess.run(compile_command, check=True)
    subprocess.run(command, cwd=OLD, check=True)

    loaded = subprocess.check_output(['ldd', str(binary)], text=True)
    for library in ('libni26_runtime.so', 'libni26_base.so'):
        paths = [line.split('=>', 1)[1].split()[0] for line in loaded.splitlines()
                 if line.strip().startswith(library + ' =>')]
        if 'not found' in loaded or paths != [str(RUNTIME / library)]:
            raise ValueError('wrong library in actual loader: ' + library)
    symbols = subprocess.check_output(['nm', '--defined-only', '-C', str(binary)], text=True)
    for name in ('neural::RawFBC::forward', 'neural::SessionBN::forward',
                 'neural::prepare_raw_training_source', 'neural::raw_source_update'):
        if not any(' T ' in line and name in line for line in symbols.splitlines()):
            raise ValueError('missing direct native source consumer: ' + name)
    (root / 'worker_ldd.txt').write_text(loaded)
    (root / 'build_proof.json').write_text(json.dumps(dict(
        compile=compile_command, link=command, only_test_object_rebuilt=True), indent=2))


if __name__ == '__main__':
    main()
