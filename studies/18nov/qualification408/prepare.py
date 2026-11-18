"""Build a private sibling without changing the pinned package."""
import hashlib
import os
from pathlib import Path

stock = Path('../ni26_qualify385_v1_20261118/cap')
dest = Path('pruned')
dest.mkdir()
for path in sorted(stock.iterdir()):
    if path.name != 'submission.py':
        os.link(path, dest / path.name)
        continue
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == '37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f'
    anchor = b'            self.V.append(self.view(x))'
    assert data.count(anchor) == 1
    edit = b'''            if self.latch is None and t >= max(self.thr):
                self.t += 1
                return p
            if self.latch is None:
                self.V.append(self.view(x))'''
    (dest / path.name).write_bytes(data.replace(anchor, edit))
assert len(list(dest.iterdir())) == 39
