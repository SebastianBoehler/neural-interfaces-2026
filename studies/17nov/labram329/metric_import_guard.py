"""Reproduce baseline's cached-protocol collision and prove named official scorer ownership."""
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sys.path[:0]=[str(BASE/'experiments'),str(BASE/'2026-competition')]
    import protocol as cached
    if Path(cached.__file__).resolve()!=BASE/'experiments/protocol.py':raise ValueError('collisionreproducerwrongpath')
    sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'2026-competition')]
    import protocol as repeated
    if repeated is not cached or Path(repeated.__file__).resolve()==ROOT/'experiments/protocol.py':raise ValueError('cachedprotocolcollisionnotreproduced')
    name='ni26_atom329_named_metric_guard';spec=importlib.util.spec_from_file_location(name,ROOT/'experiments/protocol.py')
    own=importlib.util.module_from_spec(spec);sys.modules[name]=own;spec.loader.exec_module(own)
    actual=Path(inspect.getsourcefile(own.score)).resolve();metric=Path(inspect.getsourcefile(own.balanced_accuracy)).resolve()
    if actual!=ROOT/'experiments/protocol.py' or sha(metric)!=sha(ROOT/'2026-competition/benchmark_utils/metrics.py'):raise ValueError('actualofficialmetricidentityfailed')
    result=dict(passed=True,labels_opened=False,models_fit=False,collision_reproduced=True,cached_protocol=str(cached.__file__),
        named_protocol=str(own.__file__),score_function_path=str(actual),score_file_sha256=sha(actual),
        actual_metric_path=str(metric),actual_metric_sha256=sha(metric),official_snapshot_metric_sha256=sha(ROOT/'2026-competition/benchmark_utils/metrics.py'))
    with (ROOT/'metric_import_guard.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
