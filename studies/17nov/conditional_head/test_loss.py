"""Small deterministic tests for the conditional penalty's invariants."""
import numpy as np
import torch
from loss import objective, penalty


def test_conditional_penalty_gradient_and_gauge():
    generator = torch.Generator().manual_seed(53)
    theta = torch.randn((3, 9), generator=generator, dtype=torch.float64, requires_grad=True)
    delta = torch.randn((21, 8), generator=generator, dtype=torch.float64)
    value = penalty(theta, delta)
    grad, = torch.autograd.grad(value, theta)
    centered = theta[:, :-1]-theta[:, :-1].mean(0, keepdim=True)
    expected = 2*(delta@centered.T).T@delta/(21*3)
    expected -= expected.mean(0, keepdim=True)
    torch.testing.assert_close(grad[:, :-1],expected,rtol=1e-12,atol=1e-12)
    assert torch.count_nonzero(grad[:, -1]) == 0
    direction = grad / torch.linalg.vector_norm(grad)
    eps = 1e-5
    fd = (penalty(theta+eps*direction,delta)-penalty(theta-eps*direction,delta))/(2*eps)
    torch.testing.assert_close(fd,(grad*direction).sum(),rtol=1e-8,atol=1e-8)
    shift = torch.randn((1, 8), generator=generator, dtype=torch.float64)
    shifted = torch.cat((theta[:,:-1]+shift,theta[:,-1:]),dim=1)
    torch.testing.assert_close(value,penalty(shifted,delta),rtol=1e-12,atol=1e-12)
    zero = penalty(theta,torch.zeros_like(delta)); zg, = torch.autograd.grad(zero,theta)
    assert zero.item() == 0 and torch.count_nonzero(zg) == 0


def test_zero_lambda_same_objective():
    generator = torch.Generator().manual_seed(99)
    x = torch.randn((12,8),generator=generator,dtype=torch.float64)
    y = torch.arange(12) % 3
    theta = torch.randn((3,9),generator=generator,dtype=torch.float64)
    d1 = torch.randn((21,8),generator=generator,dtype=torch.float64)
    d2 = torch.zeros_like(d1)
    np.testing.assert_equal(objective(x,y,theta,d1,0,1).item(),objective(x,y,theta,d2,0,1).item())
