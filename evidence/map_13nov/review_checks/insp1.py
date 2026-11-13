import numpy as np, sys
D="runs/replay_data/"
for fn in ["inputs.npz","bases.npz","bases_fresh.npz","bases_fresh_mix.npz","post_cache.npz","base_cache.npz"]:
    z=np.load(D+fn,allow_pickle=True)
    print("==",fn)
    for k in z.files:
        a=z[k]; print("  ",k,a.shape,a.dtype)
