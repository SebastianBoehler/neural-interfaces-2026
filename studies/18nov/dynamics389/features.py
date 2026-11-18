"""Stateless, two-half lagged EEG sufficient statistics for ridge next-sample prediction."""
import json
from pathlib import Path
import sys
import numpy as np
from common import DATA,DATA_SHA,SENSORS,metadata,sha

BANDS=((4,8),(8,14),(14,30))


def transform(x,channel_index):
    assert x.ndim==3 and x.shape[1:]==(41,480)
    x=x[:,channel_index]-x.mean(1,keepdims=True)
    spec=np.fft.rfft(x,axis=-1)
    hz=np.fft.rfftfreq(480,1/120)
    n=len(x)
    gram=np.empty((n,2,3,60,60),np.float32)
    perm_xy=np.empty((n,2,3,48,12),np.float32)
    for b,(low,high) in enumerate(BANDS):
        waveform=np.fft.irfft(spec*((hz>=low)&(hz<high)),n=480,axis=-1).astype(np.float32)
        for h in range(2):
            t=waveform[:,:,h*240:(h+1)*240]
            z=np.concatenate([t[:,:,8-lag:240-lag] for lag in (1,2,4,8,0)],axis=1)
            g=z@z.transpose(0,2,1)/232
            perm=np.random.default_rng(38900+h).permutation(232)
            xy=z[:,:48,perm]@z[:,48:].transpose(0,2,1)/232
            gram[:,h,b]=g
            perm_xy[:,h,b]=xy
    assert np.isfinite(gram).all() and np.isfinite(perm_xy).all()
    return gram,perm_xy


def main(smoke):
    metadata()
    with np.load(DATA) as z:
        names=z['ch_names'].tolist()
        idx=[names.index(c) for c in SENSORS]
        x=z['X']
    start=transform(x[:4],idx)
    one=transform(x[1:2],idx)
    for a,b in zip(start,one):
        np.testing.assert_allclose(a[1],b[0],rtol=1e-5,atol=1e-18)
    assert np.linalg.norm(start[0][...,:48,48:]-start[1])>0
    if smoke:
        Path('input_smoke.json').write_text(json.dumps(dict(source=sha('source.sha256'),data=DATA_SHA)))
        return
    assert json.loads(Path('input_smoke.json').read_text())['source']==sha('source.sha256')
    dest=[np.lib.format.open_memmap(name+'.npy',mode='w+',dtype=np.float32,shape=(len(x),*a.shape[1:]))
          for name,a in zip(('gram','perm_xy'),start)]
    for i in range(0,len(x),32):
        for dst,values in zip(dest,transform(x[i:i+32],idx)):
            dst[i:i+len(values)]=values
    for dst in dest:
        dst.flush()
    Path('inputs.json').write_text(json.dumps(dict(source=sha('source.sha256'),data=DATA_SHA,
                 gram=sha('gram.npy'),perm_xy=sha('perm_xy.npy')),indent=2))


if __name__=='__main__':
    main(sys.argv[1]=='smoke')
