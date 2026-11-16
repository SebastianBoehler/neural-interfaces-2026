"""Require an unchanged real-data smoke before any production fold."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import sha

root = Path(os.environ['EXPERT_ROOT']).resolve(strict=True)
lock = root / 'runs/input306/inputs.json'
smoke = root / 'runs/expert306_smoke/P01.npz'
marker = root / 'runs/expert306_smoke/passed.json'
check = json.loads(marker.read_text())
if (sha(lock) != lock.with_suffix('.sha256').read_text().split()[0] or
        sha(lock) != check['input_lock_sha256'] or sha(smoke) != check['smoke_sha256']):
    raise ValueError('smoke, input lock or marker changed')
print('real-data smoke and input lock match', flush=True)
