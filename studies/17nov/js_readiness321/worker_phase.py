"""Standard production worker probe after a separate controlled-executor semantic proof."""
import json
import sys
from pathlib import Path

import worker


def main(variant,device):
    if variant not in ('fir','reuse'):raise ValueError('registered variant required')
    r=json.loads(Path('/semantics/phase_'+variant+'.json').read_text())
    if r['status']!='phase_matched_optimization_guard_passed' or not r['deterministic_algorithms'] or r['complete_sha256']!=worker.sha('/experiment/complete.json'):
        raise ValueError('controlled semantic/artifact proof missing')
    if r['source_sha256']!=worker.sha('/semantics/source.sha256'):raise ValueError('semantic source changed')
    if worker.torch.are_deterministic_algorithms_enabled():raise ValueError('production worker must retain standard executor')
    original=worker.load_model
    def load(path,*a,**k):
        if str(path)=='/active/draft_off':path='/experiment/draft_off'
        return original(path,*a,**k)
    worker.load_model=load;worker.main(device)


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
