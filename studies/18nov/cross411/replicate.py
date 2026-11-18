"""Prepare a fresh two-order run from the checksum-pinned §385 native worker."""
from pathlib import Path
import hashlib
import shutil
import subprocess

original = Path('../ni26_qualify385_v1_20261118')
subprocess.run(['sha256sum', '-c', 'sources.sha256', '--status'], cwd=original, check=True)
for name in ('old', 'cap'):
    shutil.copytree(original / name, name)
for name in ('members.sha256', 'data.py', 'replay.py', 'inputs.sha256'):
    shutil.copyfile(original / name, name)

# Keep true permutation seeds in the emitted keys. Map keys to array indices in the scorer.
changes = {
    'worker.py': [('for seed in (0, 1):', 'for seed in (411, 1411):')],
    'collect.py': [('for s in (0, 1)]', 'for s in (411, 1411)]'),
                   ('s = int(s)', 's = (411, 1411).index(int(s))')],
    'launch.sbatch': [('ni26-qual385', 'ni26-cross411b')],
    'collect.sbatch': [('ni26-qual385-score', 'ni26-cross411bs')],
}
for name, replacements in changes.items():
    source = (original / name).read_text()
    for before, after in replacements:
        assert source.count(before) == 1, (name, before)
        source = source.replace(before, after)
    Path(name).write_text(source)
files = sorted(changes) + ['members.sha256', 'inputs.sha256', 'data.py', 'replay.py', 'score.py']
Path('sources.sha256').write_text(''.join(
    f'{hashlib.sha256(Path(name).read_bytes()).hexdigest()}  {name}\n' for name in files))
