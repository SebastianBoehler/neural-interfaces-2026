"""Use the verified §376 physical setup with the §378 family declaration."""
import hashlib
from pathlib import Path

WORK = Path('/mnt/lustre/work/gehler/gfh527')
HERE = Path(__file__).resolve().parent
ROOT = HERE/'chain'
PREVIOUS = WORK/'ni26_consumer376_v1_20261117'
ARMS = ('RF','RV','G','K','P')
CONTRASTS = [('K','RV'),('K','G'),('K','P'),('G','RV'),('RV','RF')]


def owned(filename, expected):
    source = (PREVIOUS/filename).read_bytes()
    assert hashlib.sha256(source).hexdigest() == expected
    return source.decode()


def main():
    text = owned('setup_chain.py','28e82b975b66ae8e608c3e99572820e87c22f1a93219ea654b4a9a8d1d61ba4f')
    for a,b,n in [
        ("('RF','O','E','C','S')",repr(ARMS),1),
        ("[('E','O'),('E','C'),('E','S'),('E','RF'),('O','RF')]",repr(CONTRASTS),1),
        ("'n376'","'n378'",1),("'376'","'378'",2)]:
        assert text.count(a) == n
        text = text.replace(a,b)
    ns = dict(__file__=str(__file__), __name__='private_setup378')
    exec(compile(text,str(PREVIOUS/'setup_chain.py'),'exec'),ns)
    assert ns['ROOT'] == ROOT and ns['ARMS'] == ARMS and ns['CONTRASTS'] == CONTRASTS
    ns['main']()


if __name__ == '__main__':
    main()
