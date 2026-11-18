"""Reflect each §449 subtraction across its original write axis, at the fixed angle."""
import sys

import torch
from intervene import variants


def amplified(items, posterior, prototypes):
    original, size = variants(items,posterior,prototypes)
    w = items.flatten(1).double()
    unit = torch.nn.functional.normalize(w,dim=1)
    out = {'unchanged':items}
    for k in ('semantic','wrong'):
        v = original[k].flatten(1).double()
        radius = (v*unit).sum(1,keepdim=True)*unit
        reflected = 2*radius-v
        torch.testing.assert_close(reflected.norm(dim=1),w.norm(dim=1),atol=2e-6,rtol=1e-6)
        torch.testing.assert_close((reflected-w).norm(dim=1),(v-w).norm(dim=1),atol=1e-9,rtol=1e-8)
        torch.testing.assert_close((reflected-radius),-(v-radius),atol=1e-12,rtol=1e-9)
        out[k] = reflected.to(items).reshape_as(items)
    return out,size


if __name__=='__main__':
    if sys.argv[1]=='source':
        import diagnostic
        diagnostic.main(sys.argv[2]=='smoke',amplified)
    else:
        from actor import run
        run(sys.argv[1],amplified,keep_deltas=True)
