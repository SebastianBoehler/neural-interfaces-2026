"""Use prior immutable implementations with new independent random seeds."""
import importlib.util
from pathlib import Path
import hashlib
ROOT=Path('/mnt/lustre/work/gehler/gfh527')


def load(folder,manifest,name):
    path=ROOT/folder
    data=(path/'source.sha256').read_bytes()
    assert hashlib.sha256(data).hexdigest()==manifest
    record=dict((k,h) for h,k in (line.split(maxsplit=1) for line in data.decode().splitlines()))
    assert hashlib.sha256((path/'selector.py').read_bytes()).hexdigest()==record['selector.py']
    spec=importlib.util.spec_from_file_location(name,path/'selector.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.exchange


near=load('ni26_band390_v1_20261118','c6f9f3a964c928a68fb6506081a2e4d0d5541466ee41d5e981d50fab84a5a35f','near390')
broad=load('ni26_turn386_v1_20261118','5fe10977cd9e510da3106442a0ffa6ddd94a08f87c137a7cc5654b9197f471b7','broad386')
CORE=('Z','N2390','N3390','F2390','F3390')
GEOM=()


def exchange(y,c,original,arm,feature):
    return (broad if arm.startswith('F') else near)(y,c,original,arm,feature)
