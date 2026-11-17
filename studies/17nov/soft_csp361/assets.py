"""Pinned package/control restoration and returning-activation audit."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
PREV = ROOT/'ni26_external_binary356_v2_20261117'
sys.path.insert(0, str(PREV))
from inputs import CHAIN, chain_guard, pin, require, sha

ORIGINAL = ROOT/'ni26_lanes292_20261116/lane1/chain'
RET = ROOT/'ni26_js321_20261117_returning_pair_cpu1'
RET_REPORT = ROOT/'ni26_js321_20261117_returning_pair_cpu_complete1/conditional_results.json'
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
SOURCE_356 = '5dc07bddb5d03a13133f05b17cb966fc8162f981791700de35b7a1f5085d4d3f'


def verify_sources():
    pin(PREV/'source.sha256', SOURCE_356)
    for line in (PREV/'source.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        pin(PREV/name, digest)
    pin(CHAIN/'data/proteus_windows.npz', DATA_SHA)
    pin(CHAIN/'runs/shuffle_perms.npz', '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f')
    pin(ORIGINAL/'studies/14nov/newmember/s281_cp.py', '49292fce9afbb505d5e39a9a27d4c2686c7140f445d1680fc3085017ee498f73')
    pin(ORIGINAL/'lean/src66/cpers.py', '2aae3dc07624b7786d26020aa99ce8488e28a0484582e228fddf20afe756e53e')


def get_stage():
    path = ORIGINAL/'studies/14nov/newmember/s281_cp.py'
    spec = importlib.util.spec_from_file_location('original_pers361', path)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    stage.verify_package_config()
    return stage.CP.CausalPers, stage.CONFIG


def get_count_filter():
    # Import the captured control's owning evaluator with its own cache paths.
    root = ORIGINAL/'studies/14nov/rundetect'
    pins = {
        'package.py': '867838bae7c3bb7d29bd6eddb5a86c88d24e9a353c87f24721cbe2be29177b0f',
        'jointcount.py': '0237451d7f21738bf8e4261075ba5b2e16a8befb3a43afe46089709f9097c37c',
        'common.py': '667b5fec87048c7bd85dd20ceabb55fc14b859bee53a8c7e6f0186af2a037299',
    }
    for name, digest in pins.items():
        pin(root/name, digest)
    data_root = ORIGINAL/'runs/rundetect'
    pin(data_root/'decisions.npz', '3e216552c8dd5acef2655008982a94768052a3871585881e3e270e8bf9e36e39')
    pin(data_root/'proj_d6_r1.npz', '0fe448b6e08676026e700daa8d4eadcec771d0641a29394864297dcd28cbceaa')
    require('PROTO' not in os.environ and 'SRC' not in os.environ, 'unexpected protocol override')
    sys.path.insert(0, str(root))
    import package
    require(Path(package.__file__).resolve() == (root/'package.py').resolve()
            and Path(package.J.__file__).resolve() == (root/'jointcount.py').resolve()
            and Path(package.C.__file__).resolve() == (root/'common.py').resolve()
            and package.PROTO == 'all_loso', 'wrong count-filter owner or protocol')
    return package


def load_controls(dtype=np.float64):
    with np.load(CHAIN/'data/proteus_windows.npz', allow_pickle=False) as z:
        metadata = {k: z[k] for k in ('subject', 'session', 'interface')}
        raw_x = z['X'].astype(dtype)
    metadata['person'] = np.array([p.rsplit('/', 1)[1] for p in metadata['subject']])
    metadata['stream'] = np.char.add(np.char.add(metadata['subject'], '|'), metadata['session'])
    scorer, original, people, cells = chain_guard(metadata)
    return scorer, original, people, cells, metadata, raw_x


def returning_guard(orders, first_fusion):
    pin(RET_REPORT, 'd0532701c0b88a4acd8fd8bd3d6b7b98469ef1d6bae0758b34ce690d50313f04')
    parent = json.loads(RET_REPORT.read_text())
    require(len(parent['complete_cells']) == 16, 'returning archive scope changed')
    stopped = {}
    for index in range(16):
        folder = RET/f'cell_{index}'
        owned = parent['complete_cells'][index]
        require(owned['index'] == index, 'returning proof index changed')
        pin(folder/'proof.json', owned['proof_sha256'])
        proof = json.loads((folder/'proof.json').read_text())
        fold = ('P04_s23', 'P10_s23')[index // 8]
        require(proof['index'] == index and proof['order'] == orders[index % 8]
                and proof['fold'] == fold and proof['query_labels_read'] is False,
                'returning-person proof does not match the registered cell')
        require(proof['predictions_sha256'] == owned['predictions_sha256'], 'returning prediction source differs')
        pin(folder/'predictions.npz', owned['predictions_sha256'])
        with np.load(folder/'predictions.npz', allow_pickle=False) as z:
            require(np.array_equal(z['base_scores'], z['off_scores'])
                    and np.array_equal(z['base_labels'], z['off_labels']), 'returning zero guard failed')
        require(len(proof['records']) == 2, 'returning session count changed')
        for r in proof['records']:
            a = r['activation']['base']
            require(a['rec'] == a['cp_stopped'] and 0 <= a['cp_stopped'] < first_fusion
                    and a['cp_fused'] == 0, 'CP affected a returning control output')
        stopped[f'{fold}|{proof["order"]}'] = [r['activation']['base']['cp_stopped'] for r in proof['records']]
    return stopped
