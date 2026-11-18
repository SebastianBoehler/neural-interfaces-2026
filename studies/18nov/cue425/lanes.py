"""Prepare two isolated native-worker directories from the verified §411b runner."""
from pathlib import Path
import hashlib
import shutil
import subprocess

original = Path('../ni26_cross411b_20261118').resolve()
for manifest in ('sources.sha256', 'members.sha256', 'inputs.sha256'):
    subprocess.run(['sha256sum','-c','--status',manifest],cwd=original,check=True)
for offset in (30,60):
    lane = Path(f'offset{offset}')
    lane.mkdir()
    dataset = Path(f'shift{offset}.npz').resolve()
    for arm in ('old','cap'):
        (lane/arm).symlink_to(original/arm, target_is_directory=True)
    for name in ('worker.py','data.py','replay.py','collect.py','collect.sbatch','members.sha256'):
        shutil.copyfile(original/name,lane/name)
    old = (original/'launch.sbatch').read_text()
    token = '$repo/data/scherer2015/windows.npz:/external.npz'
    assert old.count(token)==1
    (lane/'launch.sbatch').write_text(old.replace(token, f'{dataset}:/external.npz'))
    # Worker assets stay bound at the original paths, whereas the external input is the released crop.
    checks = []
    for line in (original/'inputs.sha256').read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        if path.endswith('scherer2015/windows.npz'):
            path = str(dataset)
            with dataset.open('rb') as stream:
                digest = hashlib.file_digest(stream,'sha256').hexdigest()
        else:
            path = str((original/path).resolve())
        checks.append(f'{digest}  {path}\n')
    (lane/'inputs.sha256').write_text(''.join(checks))
    files = sorted(f for f in lane.iterdir() if f.is_file())
    (lane/'sources.sha256').write_text(''.join(
        f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n' for f in files))
