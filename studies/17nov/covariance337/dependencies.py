"""Pinned immutable334 Native implementation; no copied scientific implementation."""
import hashlib
import sys
from pathlib import Path

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_gen334_v7_20261117')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

if sha(ROOT/'source.sha256')!='6fcdcd6cc7898979d605b9bbb3c22a2fb0f4116b44718dd75c7b82ac8864a736':raise ValueError('334 immutable source manifest')
for line in (ROOT/'source.sha256').read_text().splitlines():
    pin,name=line.split(maxsplit=1)
    if sha(ROOT/name)!=pin:raise ValueError('334 immutable dependency changed: '+name)
sys.path.insert(0,str(ROOT))
from data import metadata,PRODUCER,BASE,digest,PRODUCER_SHA
from static_features import native
