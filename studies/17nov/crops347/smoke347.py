"""Real source one-epoch no-op, paired exposure/activity and measured-cost checks."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from training347 import source,native,state_sha,run,sha,digest


def main():
    root=Path(os.environ['TRAIN347_ROOT']);start=time.monotonic();torch.set_num_threads(6)
    d,fit,_,_,_=source();reference=native(d,epochs=1);reference.fit(fit);reference_hash=state_sha(reference.net)
    records={};models={};total=0.
    for arm in ('A','B','C'):
        _,_,_,_,_,model,trace,card,seconds=run(arm,epochs=1);records[arm]=dict(trace=trace,seconds=seconds);models[arm]=model
        if arm=='A' and (trace['final_state_sha256']!=reference_hash or reference.scale!=model.scale or not np.array_equal(reference.G,model.G)):raise ValueError('disabled wrapper real native one-epoch BITWISE failure')
        if arm!='A':total+=float(np.median(trace['step_seconds'][2:]))*2480
    b,c=records['B']['trace'],records['C']['trace']
    for key in ('initial_state_sha256','scale','source_order_sha256','original_input_sha256','extra_rows_sha256','original_labels_sha256','extra_labels_sha256','base_batches','extra_batches','torch_rng_before','torch_rng_after','augment_rng_before','augment_rng_after'):
        if b[key]!=c[key]:raise ValueError('unmatched B/C source recipe:'+key)
    if not b['extra_is_original_copy'] or c['extra_is_original_copy'] or b['final_state_sha256']==c['final_state_sha256']:raise ValueError('copy/late actual route no-op')
    projection=1.5*total+180;elapsed=time.monotonic()-start
    # Reserve both full15min allocations plus all20min qualification bounds and this smoke.
    conservative_coreseconds=2*1200+26+240+6*(elapsed+15)+6*projection
    report=dict(passed=True,budget_passed=bool(projection<1800 and conservative_coreseconds<=28800),
        source_sha256=sha(root/'source.sha256'),records=records,original_one_epoch_state_sha256=reference_hash,
        original_zero_BITWISE=True,source_only=True,held_transformed=False,held_labels_opened=False,
        projected_two_full_fits_seconds=projection,projected_total_CPUcoreseconds=conservative_coreseconds,seconds=elapsed)
    (root/'smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    if not report['budget_passed']:raise ValueError('eight-CPUh measured projection fails')


if __name__=='__main__':main()
