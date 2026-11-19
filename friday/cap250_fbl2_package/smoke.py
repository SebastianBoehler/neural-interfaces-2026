"""Compare the assembled Solver with the measured native hook on two real window prefixes."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from build import ZIP_HASH, digest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'2026-competition'),str(ROOT/'native/tools'),str(ROOT/'friday/cap250c21e')]
from benchmark_utils.streaming import fresh_model
from fbl2 import FBL2Hook
from runtime_reference import load_solver

KNOWN_SHA='4da9b1b595175a7134c28b23fc42d6642d5666461f740abb9d644b20e1643d32'
UNKNOWN_SHA='3c09ad8948328945ad54f7e955d51e8b349d447f7350dc058b8e1989e7a59b37'
EXTERNAL_SHA='a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91'
PUBLIC_SHA='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'


def pinned(path,sha):
    if digest(path)!=sha:
        raise ValueError(f'pin failed for {path.name}')
    return path


def load_paths(stock,trial,scratch,device):
    paths=[]
    with ZipFile(pinned(stock,ZIP_HASH)) as original,ZipFile(trial) as packed:
        if original.namelist()!=packed.namelist() or any(
            original.read(name)!=packed.read(name) for name in original.namelist() if name!='submission.py'
        ):
            raise ValueError('assembled package changed frozen assets')
        for name,archive in (('stock',original),('trial',packed)):
            directory=Path(scratch)/name
            archive.extractall(directory)
            paths.append(load_solver(directory,device))
    return tuple(paths)


def frames(public,external,known,unknown):
    stored=json.loads(pinned(known,KNOWN_SHA).read_text())['shared']['recognized']
    assert stored['session']=='P10-3' and stored['activity']['recognized']==10
    other=json.loads(pinned(unknown,UNKNOWN_SHA).read_text())['sessions']['Scherer2015/G|1|0']
    for name,path,rows,who,visit,sha in (
        ('recognized',public,stored['rows'],'Dreyer2026Proteus/P10','3',PUBLIC_SHA),
        ('unrecognized',external,other['rows'][:31],'Scherer2015/G','1',EXTERNAL_SHA),
    ):
        with np.load(pinned(path,sha),allow_pickle=False) as z:
            assert set(z['subject'][rows])=={who} and set(z['session'][rows])=={visit}
            x=torch.from_numpy(z['X'][rows].copy())
        assert x.shape==(31,41,480) and x.dtype==torch.float32
        yield name,x


def compare(stock_template,trial_template,name,x,tolerance):
    original,assembled=fresh_model(stock_template),fresh_model(trial_template)
    reference=FBL2Hook(original,'fbl2')
    assert assembled.routeB.a==1/17 and original.routeB.a==1/17
    expected_route=10 if name=='recognized' else -1
    keys = ('out', 'mix', 'shadow', 'out_v5', 'mix6')
    errors = dict.fromkeys(keys, 0.)
    labels=[]
    for row in x:
        first=original.predict(row[None])
        reference.check(original.trace['v6'])
        second=assembled.predict(row[None])
        assert torch.equal(first,second) and second.shape==(1,) and second.dtype==torch.int64
        assert second.device.type == 'cpu'
        observed, expected = (dict(model.trace, mix6=model.trace['v6']['mix6'])
                              for model in (assembled, original))
        for key in keys:
            np.testing.assert_allclose(observed[key], expected[key], rtol=0, atol=tolerance, equal_nan=False)
            errors[key] = max(errors[key], float(np.abs(observed[key]-expected[key]).max()))
        labels.append(int(second.item()))
    assert not original.dead and not assembled.dead
    for model in (original,assembled):
        assert model.log6['n']==model.log6['v6']==31 and model.log6['rec']==expected_route and not model.log6['err']
        assert model.ft.steps==(0 if name=='recognized' else 40)
    assert reference.calls==31 and reference.moved==31
    assert reference.warm==(0 if name=='recognized' else 11)
    repeated=fresh_model(trial_template).predict(x[:1])
    assert int(repeated.item())==labels[0]
    return dict(max_posterior_error=errors,rows=31,recognized=expected_route,first=labels[0],optimizer_steps=assembled.ft.steps,**reference.summary())


def main(args):
    torch.set_num_threads(1)
    gpu = args.device == 'cuda:0'
    if gpu:
        assert torch.__version__ == '2.14.1+cu130' and 'RTX 2080 Ti' in torch.cuda.get_device_name()
        assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    with TemporaryDirectory() as scratch:
        stock, trial = load_paths(args.stock, args.trial, scratch, args.device)
        proof = {}
        for name, x in frames(args.public, args.external, args.known, args.unknown):
            x = x.to(args.device)
            if gpu:
                warmed = fresh_model(stock)
                FBL2Hook(warmed, 'fbl2')
                for row in x:
                    warmed.predict(row[None])
                del warmed
            proof[name] = compare(stock, trial, name, x, 5e-5 if gpu else 0.)
    receipt = dict(scope='package assembly smoke', device=args.device, archive=digest(args.trial),
                   matches_hook=True, labels_opened=False, streams=proof)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('stock','trial','public','external','known','unknown'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    main(parser.parse_args())
