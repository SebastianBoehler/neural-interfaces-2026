"""Build the original full-system evaluator in a private physical checkout."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

WORK = Path('/mnt/lustre/work/gehler/gfh527')
HERE = Path(__file__).resolve().parent
ROOT = HERE/'chain'
ORIGINAL = WORK/'ni26_s315_setup_20261117'
REL = Path('studies/17nov/representation_chain')
ARMS = ('RF','O','E','C','S')
FAMILIES = tuple('n376'+a.lower() for a in ARMS)
CONTRASTS = [('E','O'),('E','C'),('E','S'),('E','RF'),('O','RF')]


def main():
    import hashlib
    expected = '35675981adeac5fdacea89e7bc024f1e23e0bdf3ca47ff0d21f6de0fff8bf274'
    assert hashlib.sha256((ORIGINAL/'setup.sha256').read_bytes()).hexdigest() == expected
    subprocess.run(['sha256sum','-c','setup.sha256'],cwd=ORIGINAL,check=True)
    if ROOT.exists():
        raise FileExistsError(ROOT)
    source = WORK/'ni26_lanes292_20261116/lane1/chain'
    subprocess.run(['cp','-a',str(source),str(ROOT)],check=True)
    shutil.copytree(ORIGINAL/'representation_chain',ROOT/REL)
    folder = ROOT/'studies/14nov/newmember'
    for p in (ORIGINAL/'original').glob('*.py'):
        dest = folder/p.name
        dest.chmod(0o644)
        shutil.copyfile(p,dest)
    for name in ('runs/chainout','runs/newmember','runs/s281_v66_cp','logs'):
        path = ROOT/name
        if path.is_symlink():
            raise ValueError(f'linked private output: {name}')
        shutil.rmtree(path)
        path.mkdir()
    sys.path.insert(0,str(ROOT/REL))
    from config import GROUPS, sha
    from patch import main as patch
    GROUPS['376'] = dict(families=FAMILIES,arms=ARMS,contrasts=CONTRASTS)
    patch('376',ROOT)
    manifest = {str(p.relative_to(ROOT)):sha(p) for subtree in ('studies','replay','experiments','scripts','lean')
                for p in (ROOT/subtree).rglob('*.py')}
    manifest['representation_config.json'] = sha(ROOT/'representation_config.json')
    for name in manifest:
        if (ROOT/name).is_symlink():
            raise ValueError(f'linked source {name}')
        (ROOT/name).chmod(0o444)
    lock = HERE/'runtime.sha256'
    lock.write_text(''.join(f'{h}  {name}\n' for name,h in sorted(manifest.items())))
    print(json.dumps(dict(source=sha(lock),root=str(ROOT))))


if __name__ == '__main__':
    main()
