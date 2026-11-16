import sys
import unittest
from pathlib import Path
import torch
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parent.parent/'meta_adaptation')]
import adapt
import flatten


class EntropySignTest(unittest.TestCase):
    def test_only_per_window_entropy_changes_sign(self):
        torch.manual_seed(13)
        anchor=torch.randn(3,adapt.FEATURES,dtype=torch.float64)*.001
        head=(anchor+0.00001).requires_grad_(True)
        bias=torch.tensor([.1,0.,-.1],dtype=torch.float64)
        x=torch.randn(4,adapt.FEATURES,dtype=torch.float64)
        logp=torch.log_softmax(x@head.T+bias,dim=1); entropy=-(logp.exp()*logp).sum(1).mean()
        difference=flatten.loss(head,anchor,bias,x)-adapt.loss(head,anchor,bias,x)
        self.assertTrue(torch.allclose(difference,-2*entropy,atol=1e-12,rtol=0))
        gradient=torch.autograd.grad(difference+2*entropy,head)[0]
        self.assertLess(float(gradient.abs().max()),1e-12)


if __name__=='__main__':unittest.main()
