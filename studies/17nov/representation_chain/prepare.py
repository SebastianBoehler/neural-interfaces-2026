"""Create a private chain snapshot with shared immutable inputs and copied code."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from config import sha
from patch import main as patch

TEMPLATE = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain')
ORIGINAL = Path('/mnt/lustre/work/gehler/gfh527/ni26_s315_setup_20261117')
REL = Path('studies/17nov/representation_chain')


def main(group, destination):
    root = Path(destination)
    if root.exists():
        raise FileExistsError(root)
    subprocess.run(['cp', '-as', str(TEMPLATE), str(root)], check=True)
    # Copy every source tree, so no patched or executing source depends on the old owner.
    for name in ('studies', 'replay', 'experiments', 'scripts', 'lean'):
        shutil.rmtree(root / name)
        shutil.copytree(TEMPLATE / name, root / name, symlinks=False,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ORIGINAL / 'representation_chain', root / REL)
    # Use the exact original §281 files as anchors for the existing §292 patcher.
    folder = root / 'studies/14nov/newmember'
    for name in ('s281_provenance.py', 'run_s.py', 's281_cp.py', 'score_s_v66p.py', 'run_s_v66.py'):
        shutil.copy2(ORIGINAL / 'original' / name, folder / name)
    for name in ('chainout', 'newmember', 's281_v66_cp', 'representation_streams', 'representation_inputs'):
        p = root / 'runs' / name
        if p.exists():
            shutil.rmtree(p)
        if name not in ('representation_streams', 'representation_inputs'):
            p.mkdir()
    shutil.rmtree(root / 'logs'); (root / 'logs').mkdir()
    patch(group, root)
    records = {}
    for name in ('studies', 'replay', 'experiments', 'scripts', 'lean'):
        for path in sorted((root / name).rglob('*.py')):
            if path.is_symlink():
                raise ValueError(f'executing source cannot be linked: {path}')
            records[str(path.relative_to(root))] = sha(path)
    for path in sorted((root / REL).glob('*.sbatch')):
        records[str(path.relative_to(root))] = sha(path)
    records['representation_config.json'] = sha(root / 'representation_config.json')
    manifest = root / 'representation-source.sha256'
    with manifest.open('x') as f:
        for name, digest in sorted(records.items()):
            f.write(f'{digest}  {name}\n')
    for name in records:
        (root / name).chmod(0o444)
    manifest.chmod(0o444)
    print(json.dumps(dict(group=group, workspace=str(root), manifest_sha256=sha(manifest))), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
