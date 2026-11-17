"""Fixed within-current-window source interventions; exact exposed source rows preserved."""
import numpy as np


def variants(x):
    frequency=np.fft.rfftfreq(480,1/120);spectrum=np.fft.rfft(x,axis=-1)
    masks={'remove_0_1':~(frequency<=1),'remove_1_4':~((frequency>=1)&(frequency<4)),
        'remove_4_40':~((frequency>=4)&(frequency<40)),'remove_40_58':~((frequency>=40)&(frequency<=58)),
        'keep_4_40':(frequency>=4)&(frequency<40)}
    result={'identity':x.copy()}
    result.update({k:np.fft.irfft(spectrum*m,n=480,axis=-1).astype(np.float32) for k,m in masks.items()})
    result['CAR']=(x-x.mean(1,keepdims=True)).astype(np.float32)
    return result
