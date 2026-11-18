"""Bind fixed §401 replay and scorer to the separately source-sealed projection members."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path('/mnt/lustre/work/gehler/gfh527')
FULL=ROOT/'ni26_priv401_full_v1_20261118'
SOURCE=ROOT/'ni26_proj401c_source_v1_20261118'


def pin(root,expected):
    raw=(root/'source.sha256').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected
    for line in raw.decode().splitlines():
        h,n=line.split(maxsplit=1)
        assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h


def members():
    pin(SOURCE,'8c559dd410e94b19244a1786f5909e5142e42fc4c6eefa5f05d0e9a9c19abaca')
    card=json.loads((SOURCE/'source.json').read_text())
    assert not card['query_labels_opened'] and card['eligible']==1894 and card['source_dimensions']==96
    assert card['removed_variance']==8 and card['source']==hashlib.sha256((SOURCE/'source.sha256').read_bytes()).hexdigest()
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
