"""All mutable inference buffers and the compiled graph live inside one session."""
from pathlib import Path
import torch

SEEDS=(454,1454)


class SessionGraph:
    def __init__(self):
        torch.ops.load_library('/native/decoder.so')
        params=[torch.load(Path('/weights')/f'all_{s}.pt',map_location='cuda',weights_only=True) for s in SEEDS]
        keys=('encoder.0','encoder.1','head','query','key','value')
        self.weights=[torch.stack([p['model'][key+'.'+v] for p in params]) for key in keys for v in ('weight','bias')]
        self.centroid=torch.stack([p['centroid'].reshape(48) for p in params])
        self.mean=params[0]['mean']
        self.scale=params[0]['scale']
        for field in ('mean','scale'):
            assert torch.equal(params[0][field],params[1][field])
        self.frame=torch.zeros((41,480),device='cuda')
        self.hann=torch.hann_window(480,device='cuda')
        self.state=torch.zeros((2,2,48),device='cuda')
        self._predict()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out=self._predict()
        self.state.zero_()

    def _predict(self):
        x=torch.ops.ni458.feature(self.frame,self.hann,self.mean,self.scale)
        logits=torch.ops.ni458.step(x,self.weights,self.state,self.centroid)
        diff=logits[:,0].double()-logits[:,1].double()
        return .1*(diff-diff.mean(1,keepdim=True)).clamp(-1,1)

    def step(self,window):
        self.frame.copy_(window[0])
        self.graph.replay()
        return self.out.cpu().numpy()
