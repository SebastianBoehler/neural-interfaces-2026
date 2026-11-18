"""Produce the frozen §387b worker by modifying only its stream order."""
from pathlib import Path

source = Path('../ni26_runtime387_v1_20261118/worker.py').read_text()
for old, new in [
    ('for name, bank, rows in jobs:', 'for name, bank, rows in (jobs[0], jobs[2], jobs[1]):'),
    ("if name.endswith('/2'):", "if name.endswith('/1'):"),
]:
    assert source.count(old) == 1
    source = source.replace(old, new)
Path('worker.py').write_text(source)
