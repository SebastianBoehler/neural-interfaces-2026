"""Verify all fold checkpoints and §379 episodes before using the original scorer."""
import json
from pathlib import Path

from pilot import ARMS, OWNER, verify

import score379 as score

verify()
for p in score.PEOPLE:
    row = json.loads(Path(p+'.json').read_text())
    original = json.loads((OWNER/(p+'.json')).read_text())
    score.require(row['source_episodes']==original['source_episodes'], 'source episode mismatch '+p)
    score.require(set(row['checkpoints'])==set(ARMS), 'checkpoint inventory '+p)
    for arm, h in row['checkpoints'].items():
        score.require(score.sha(Path(f'{p}_{arm}.pt')) == h, 'checkpoint mismatch '+p+arm)
score.ARMS = ARMS
score.CONTRASTS = [('spec',b) for b in ('raw','perm','frozen','JS')] + [('raw','JS'),('perm','JS')]
score.main(Path.cwd())
Path('before_score379.json').rename('before_score380.json')
Path('result379.json').rename('result380.json')
