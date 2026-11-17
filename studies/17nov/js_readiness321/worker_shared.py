"""Reuse unchanged paired worker probe after actual optimized source guard passes."""
import json
import sys
from pathlib import Path

import worker


def main(device):
    g=json.loads(Path('/experiment/optimization_guard.json').read_text())
    if g['status']!='same_weight_sharedFIR_raw_active_reset_guards_passed':raise ValueError('optimized source guard missing')
    if g['complete_sha256']!=worker.sha('/experiment/complete.json'):raise ValueError('optimized artifact provenance changed')
    original=worker.load_model
    def load(path,*args,**kwargs):
        if str(path)=='/active/draft_off':path='/experiment/draft_off'
        return original(path,*args,**kwargs)
    worker.load_model=load;worker.main(device)


if __name__=='__main__':main(sys.argv[1])
