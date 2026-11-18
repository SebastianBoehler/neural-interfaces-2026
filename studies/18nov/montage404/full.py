"""Bind fixed §401 replay and scorer to the separately source-sealed interface-metric members."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path('/mnt/lustre/work/gehler/gfh527')
FULL=ROOT/'ni26_priv401_full_v1_20261118'
SOURCE=ROOT/'ni26_align404_v1_20261118'

SOURCE_PIN='23762a6bbea74582a1553bdbc26f28c1cdcf21a4b5c07b9ac8ba1ce429f92309'


def pin(root,expected):
    raw=(root/'source.sha256').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected
    for line in raw.decode().splitlines():
        h,n=line.split(maxsplit=1)
        assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h


def members():
    pin(SOURCE,SOURCE_PIN)
    card=json.loads((SOURCE/'source.json').read_text())
    assert not card['query_labels_opened'] and card['eligible']==1898 and card['held']==757
    assert card['source']==hashlib.sha256((SOURCE/'source.sha256').read_bytes()).hexdigest()
    assert card['members']==hashlib.sha256((SOURCE/'members.npz').read_bytes()).hexdigest()
    return card


def main():
    pin(FULL,'0fc98e30f5b6f2cec8a30ca2f56480545f2619eae2d56c9d5a6176a18535c732')
    sys.path.insert(0,str(FULL))
    import full401
    # Both frozen entry points resolve these two member-bank bindings at invocation.
    full401.SRC=SOURCE
    full401.members=members
    if sys.argv[1]=='score':
        import score401
        assert score401.SRC==SOURCE and score401.members is members
        score401.main()
    else:
        full401.main(sys.argv[1]=='smoke',int(sys.argv[2]))


if __name__=='__main__':
    main()
