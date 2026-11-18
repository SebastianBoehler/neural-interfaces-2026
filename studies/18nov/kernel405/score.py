"""Use the original guarded whole-decoder scorer for each registered seed."""
import os
import sys
from pathlib import Path

import score379 as score

seed = int(sys.argv[1])
assert seed in (405, 1405)
root = Path.cwd()
score.source_manifest(root/'source.sha256')
os.chdir(str(seed))
for name in ['source.sha256'] + [line.split(maxsplit=1)[1] for line in (root/'source.sha256').read_text().splitlines()]:
    path = Path(name)
    if not path.exists():
        path.symlink_to(root/name)
score.ARMS = ('U', 'D', 'L')
score.CONTRASTS = [('L', arm) for arm in ('U', 'D', 'frozen', 'JS')] + [('U','frozen')]
score.main(Path.cwd())
