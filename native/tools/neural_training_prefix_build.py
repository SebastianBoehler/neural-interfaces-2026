"""Link the shared Adam source update with the already qualified FIR objects."""

from concurrent.futures import ThreadPoolExecutor
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
    if str(FIR / 'build/training_fbc.o') not in command:
        raise ValueError('qualified FIR proof has unexpected training objects')
    compile_commands = []
    for source in ('src/training.cpp', 'src/training_source.cpp', 'tests/neural_training_source.cpp'):
        object_name = Path(source).stem + '.o'
        previous = [i for i, arg in enumerate(command) if arg.endswith('/' + object_name)]
        if len(previous) != 1:
            raise ValueError('qualified link has unexpected object ' + object_name)
        output = str(build / object_name)
        command[previous[0]] = output
        compile_commands.append([command[0], *shlex.split(options['CXX_DEFINES']),
            '-I' + str(root / 'native/include'), *shlex.split(options['CXX_INCLUDES']),
            *shlex.split(options['CXX_FLAGS']), '-c', str(root / 'native' / source), '-o', output])
    binary = build / 'neural_training_source'
    command[command.index('-o') + 1] = str(binary)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda args: subprocess.run(args, check=True), compile_commands))
    subprocess.run(command, cwd=OLD, check=True)

    loaded = subprocess.check_output(['ldd', str(binary)], text=True)
    for library in ('libni26_runtime.so', 'libni26_base.so'):
        paths = [line.split('=>', 1)[1].split()[0] for line in loaded.splitlines()
                 if line.strip().startswith(library + ' =>')]
        if 'not found' in loaded or paths != [str(RUNTIME / library)]:
            raise ValueError('wrong library in actual loader: ' + library)
    symbols = subprocess.check_output(['nm', '--defined-only', '-C', str(binary)], text=True)
    for name in ('neural::RawFBC::forward', 'neural::SessionBN::forward',
                 'neural::prepare_raw_training_source',
                 'neural::raw_source_update(ni26::neural::RawFBC&, ni26::neural::Adam&',
                 'ni26::train_raw_fbc'):
        if not any(' T ' in line and name in line for line in symbols.splitlines()):
            raise ValueError('missing direct native source consumer: ' + name)
    (root / 'worker_ldd.txt').write_text(loaded)
    (root / 'build_proof.json').write_text(json.dumps(dict(
        compile=compile_commands, link=command, reused_FIR_object=True), indent=2))


if __name__ == '__main__':
    main()
