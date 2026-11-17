"""Verified original factories and separate branch for §376c consumer."""
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/lustre/work/gehler/gfh527')
PARENT = WORK/'ni26_consumer376_v1_20261117'
PROD = WORK/'ni26_complement376c_v1_20261118'
P0 = WORK/'ni26_complement376_v2_20261117'
CHAIN = HERE/'chain'
ARMS = ('A','G')
ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def require(ok,detail):
    if not ok:
        raise ValueError(detail)


def pin(path,expected):
    require(sha(path)==expected, str(path))


def setup():
    pin(PARENT/'setup_chain.py','28e82b975b66ae8e608c3e99572820e87c22f1a93219ea654b4a9a8d1d61ba4f')
    sys.path.insert(0,str(PARENT))
    import setup_chain
    setup_chain.HERE, setup_chain.ROOT = HERE,CHAIN
    setup_chain.ARMS, setup_chain.FAMILIES = ARMS, tuple('n376c'+a.lower() for a in ARMS)
    setup_chain.CONTRASTS = [('G','A')]
    setup_chain.main()


if __name__ == '__main__':
    setup()
