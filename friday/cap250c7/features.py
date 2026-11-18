"""Build the same probability/log features on historical and exact-worker streams."""
import numpy as np

from arbitrate import COUNT, FLOOR, PRIOR, prior_scales

PAIRS = np.array([1, 2, 3, 5, 6, 7])


def columns(old, cap, old_scale, cap_scale):
    a, b = old.argmax(1), cap.argmax(1)
    row = np.arange(len(a))
    m0, m1 = old[row,a]-old[row,b], cap[row,b]-cap[row,a]
    u, v = np.sort(old,axis=1), np.sort(cap,axis=1)
    g0, g1 = u[:,-1]-u[:,-2], v[:,-1]-v[:,-2]
    pair = 3*a+b
    x = np.column_stack([m0,m1,m0/old_scale,m1/cap_scale,g0/old_scale,g1/cap_scale,
                         pair[:,None]==PAIRS])
    return x, a, b, g0, g1


def batch(old, cap):
    x,a,b,_,_ = columns(old,cap,prior_scales(old),prior_scales(cap))
    return x,a,b


class CausalFeatures:
    def __init__(self):
        self.count = COUNT
        self.old_sum = COUNT*PRIOR
        self.cap_sum = COUNT*PRIOR

    def step(self, old, cap):
        x,a,b,g0,g1 = columns(old[None],cap[None],max(FLOOR,self.old_sum/self.count),
                               max(FLOOR,self.cap_sum/self.count))
        self.old_sum += g0[0]
        self.cap_sum += g1[0]
        self.count += 1
        return x[0],int(a[0]),int(b[0])


def select(x, a, b, parameters):
    features = (x-parameters['mean'])/parameters['scale']
    utility = features@parameters['beta']+parameters['bias']
    return np.where((a!=b)&(utility>.05),a,b)
