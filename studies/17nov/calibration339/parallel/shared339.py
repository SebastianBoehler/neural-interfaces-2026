"""Source-locked339science; orchestration alone separates independent seeds."""
import hashlib
import sys
from pathlib import Path
BASE=Path('/mnt/lustre/work/gehler/gfh527/ni26_calibration339_v2_20261117')
BASE_SHA='c6f199c747dbd2c38e86165735bc86553bb53a1ed30cd348f1792ccdb17ff22f'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(BASE/'source.sha256')!=BASE_SHA:raise ValueError('339scientificbasechanged')
for line in (BASE/'source.sha256').read_text().splitlines():
    h,n=line.split(maxsplit=1)
    if sha(BASE/n)!=h:raise ValueError('339scientificmodulechanged')
sys.path.insert(0,str(BASE))
import inputs339,temperature339
for mod in (inputs339,temperature339):
    if Path(mod.__file__).resolve()!=(BASE/(mod.__name__+'.py')).resolve():raise ValueError('wrong339scienceimport')
from inputs339 import src,inputs,logits,fingerprint,OUTER,CAL,VIEWS
from temperature339 import ensemble,apply,weights,fit


def digest(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
