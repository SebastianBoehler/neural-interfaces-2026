"""Extract the exact frozen FIR bytes for the independent native hash test."""
import argparse
import hashlib
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('module',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('refusing to replace a hash fixture')
    module=torch.jit.load(str(args.module),map_location='cpu')
    coefficients=[p.detach().numpy() for _,p in module.spectral_filtering.fir_list.named_parameters()]
    if [len(p) for p in coefficients]!=[199,199,133,99,81,67,57,51,45]:raise ValueError('FIR shape changed')
    if any(p.dtype.kind!='f' or p.dtype.itemsize!=4 for p in coefficients):raise ValueError('FIR must remain float32')
    raw=b''.join(p.astype('<f4',copy=False).tobytes() for p in coefficients)
    if hashlib.sha256(raw).hexdigest()!='607a44ee25576803092c03d77feb67411fe7a2266475fc730ae7ae28c81506c6':raise ValueError('frozen FIR identity changed')
    args.output.write_bytes(raw)


if __name__=='__main__':main()
