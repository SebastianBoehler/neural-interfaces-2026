"""Create an explicitly CPU-only producer route without editing frozen GPU sources."""
import hashlib
import json
import shutil
from pathlib import Path

SOURCE = Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2')
TARGET = Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2')
SETUP = Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_setup_20261117_v2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    guard = json.loads((SOURCE / 'guard_cpu.json').read_text())
    if not guard['passed'] or guard['device'] != 'cpu':
        raise ValueError('passed explicit CPU source guard required')
    TARGET.mkdir()
    for name in ('experiments', 'studies'):
        shutil.copytree(SOURCE / name, TARGET / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('data', 'runs', '2026-competition', '.venv'):
        (TARGET / name).symlink_to((SOURCE / name).resolve(), target_is_directory=True)
    for path in SOURCE.glob('*.py'):
        shutil.copy2(path, TARGET / path.name)
    (TARGET / 'run.py').chmod(0o644)
    run = (TARGET / 'run.py').read_text()
    old = "    if not torch.cuda.is_available():\n        raise ValueError('explicit CUDA route unavailable')"
    if run.count(old) != 1:
        raise ValueError('explicit GPU route source anchor changed')
    run = run.replace(old, "    if os.environ['NI26_DEVICE'] != 'cpu':\n        raise ValueError('this independent route requires explicit CPU')")
    run = run.replace("'cuda'", "'cpu'")
    run = run.replace("meta.update(arm=arm,", "meta.update(device='cpu', device_rng='CPU-native;not CUDA trajectory parity', arm=arm,")
    compile(run, str(TARGET / 'run.py'), 'exec')
    (TARGET / 'run.py').write_text(run)
    for path in SETUP.iterdir():
        if path.suffix in ('.py', '.sbatch') and path.name != 'prepare_cpu.py':
            shutil.copy2(path, TARGET / path.name)
    (TARGET / 'route.json').write_text(json.dumps(dict(device='cpu', independent_complete_matrix=True,
        original_source_manifest=sha(SOURCE / 'source.sha256'), cpu_guard_sha256=sha(SOURCE / 'guard_cpu.json'),
        excluded_devices=['cuda'], no_mixed_folds=True), indent=2))
    paths = list(TARGET.glob('*.py')) + list(TARGET.glob('*.sbatch')) + [TARGET / 'route.json']
    paths += list((TARGET / 'experiments').rglob('*.py')) + list((TARGET / 'studies').rglob('*.py'))
    with (TARGET / 'source.sha256').open('x') as f:
        for path in sorted(paths):
            f.write(f'{sha(path)}  {path.relative_to(TARGET)}\n')
    for path in paths + [TARGET / 'source.sha256']:
        path.chmod(0o444)
    print(json.dumps(dict(workspace=str(TARGET), source_sha256=sha(TARGET / 'source.sha256'))), flush=True)


if __name__ == '__main__':
    main()
