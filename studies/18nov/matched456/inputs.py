"""Read authenticated historical and three-seed stock, with canonical stream ages."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import score379 as ref
from fusion_inputs import probability, restore

STAGES = ref.WORK / 'ni26_match456_stages_20261118'
MEMORY = ref.WORK / 'ni26_memory454_20261118'
CONSENSUS = ref.WORK / 'ni26_cap455_v1_20261118'
NATIVE = ref.WORK / 'ni26_native455_v1_20261118'
REPORT = STAGES / 'aux430_result.json'
REPORT_SHA = '249acc834ea4c0c087cec55be3d155a96fbd15580c0a0f0b277fa6224d9afa4d'


def module(folder, name, digest):
    path = folder / (name + '.py')
    assert ref.sha(path) == digest, path
    spec = importlib.util.spec_from_file_location('matched_' + name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def inputs():
    assert ref.sha(REPORT) == REPORT_SHA
    receipt = json.loads(REPORT.read_text())
    sc, scores, meta, ages = restore()
    controls, reference_ages = ref.controls_and_age(meta)
    assert tuple(sc.ORDERS) == ref.ORDERS == tuple(receipt['orders'])
    stock, old = {}, {}
    for i, order in enumerate(ref.ORDERS):
        archive = STAGES / 'stock' / f'{order}.npz'
        assert ref.sha(archive) == receipt['three_sha256'][order]
        assert np.array_equal(ages[order], reference_ages[i])
        with np.load(archive) as z:
            stock[order] = z['scores']
        old[order] = scores['control', order, 'B2_pool']
        assert np.array_equal(old[order].argmax(1), controls['frozen'][i])
        assert np.count_nonzero(old[order].argmax(1) != stock[order].argmax(1)) == receipt['changed_labels_by_order'][order]
    return meta, ages, old, stock


def log_probability(scores, ages):
    with np.errstate(divide='ignore'):
        return np.log(probability(scores, ages))
