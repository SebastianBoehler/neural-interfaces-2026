"""Existing SOURCE spectra only: reconstruct both consumers before conditional repair."""
import argparse
import json
from pathlib import Path

import torch
from torch.jit._pickle import restore_type_tag


def error(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('FIR arithmetic shape/dtype differs')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('nonfinite FIR arithmetic')
    kind=torch.complex128 if a.is_complex() else torch.float64
    delta=(a.to(kind)-b.to(kind)).abs()
    return dict(max_abs=float(delta.max()),different=int(torch.count_nonzero(a!=b)))


def main():
    p=argparse.ArgumentParser()
    for key in ('native','original','out'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    with torch.serialization.safe_globals([restore_type_tag]):native=torch.load(args.native,weights_only=True,map_location='cpu')
    original=torch.load(args.original,weights_only=True,map_location='cpu');rows=[]
    for i,length in enumerate([199,199,133,99,81,67,57,51,45]):
        key='input_spectrum_'+str(i);fx=original[key];native_fx=native[key];fy=original['fir_spectrum_'+str(i)]
        if not torch.equal(fx,native_fx) or not torch.equal(fy,native['fir_spectrum_'+str(i)]):raise ValueError('existing spectra do not match')
        if list(fx.stride())!=original[key+'_stride'].tolist() or list(fy.stride())!=original['fir_spectrum_'+str(i)+'_stride'].tolist():raise ValueError('captured original FFT layout not reproduced')
        if list(native_fx.stride())!=native[key+'_stride'].tolist():raise ValueError('captured native input FFT layout not reproduced')
        vector=native['fir_spectrum_'+str(i)][0,0].contiguous()
        retained=vector[None,None].expand(1,41,-1).contiguous()
        if not torch.equal(retained,fy) or retained.stride()!=fy.stride():raise ValueError('counterfactual FY values/layout differ from actual original')
        old_product=native_fx*vector;actual_product=fx*fy;fixed_product=native_fx*retained
        n=480+length-1;start=(length-1)//2
        reconstruct=lambda product:torch.fft.irfft(product,n=n)[...,start:start+480]
        old=reconstruct(old_product);actual=reconstruct(actual_product);fixed=reconstruct(fixed_product)
        row=dict(band=i,filter_length=length,fft_length=n,native_FX_stride=list(native_fx.stride()),original_FX_stride=list(fx.stride()),native_FY_shape=list(vector.shape),native_FY_stride=list(vector.stride()),original_FY_shape=list(fy.shape),original_FY_stride=list(fy.stride()),
            original_reproduction=error(actual,original['fir'][:,i]),native_reproduction=error(old,native['fir'][:,i]),
            product_layout_difference=error(old_product,actual_product),counterfactual_product=error(fixed_product,actual_product),counterfactual_FIR=error(fixed,original['fir'][:,i]))
        rows.append(row)
    result=dict(bands=rows,optimizer_updates=0,forward_NN_calls=0,new_reference_fits=0,only_existing_source_spectra=True,
        original_and_native_reproduced=all(r[k]['different']==0 for r in rows for k in ('original_reproduction','native_reproduction')),
        same_shape_counterfactual_exact=all(r[k]['different']==0 for r in rows for k in ('counterfactual_product','counterfactual_FIR')),
        layout_product_changes=any(r['product_layout_difference']['different'] for r in rows))
    args.out.write_text(json.dumps(result,indent=2))
    if not result['original_and_native_reproduced'] or not result['same_shape_counterfactual_exact'] or not result['layout_product_changes']:raise ValueError('FIR layout causal control failed; STOP before repair/update')
    print('FIR_LAYOUT_ARITHMETIC_CAUSAL_CONTROL_PASS; original/native/counterfactual BYTE_EXACT')


if __name__=='__main__':main()
