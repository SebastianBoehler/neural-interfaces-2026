"""Apply equally sized semantic and wrong-task edits to earlier memory writes."""
import torch


def variants(items, posterior, prototypes):
    assert posterior.shape == (len(items),3) and prototypes.shape == (3,*items.shape[1:])
    assert torch.isfinite(items).all() and torch.isfinite(posterior).all()
    torch.testing.assert_close(posterior.sum(1), torch.ones_like(posterior[:,0]))
    w = items.flatten(1).double()
    norm = w.norm(dim=1,keepdim=True)
    assert norm.min() > 0
    unit = w/norm
    prior = prototypes.flatten(1).double()
    semantic = torch.nn.functional.normalize(w-posterior.double()@prior,dim=1)
    wrong = -(posterior.double()@prior.roll(1,0))
    wrong -= (wrong*unit).sum(1,keepdim=True)*unit
    assert wrong.norm(dim=1).min() > 1e-12
    cosine = (semantic*unit).sum(1,keepdim=True)
    sine = (semantic-cosine*unit).norm(dim=1,keepdim=True)
    assert cosine.min() > 0 and sine.max() > 1e-5
    cyclic = cosine*unit+sine*torch.nn.functional.normalize(wrong,dim=1)
    semantic, cyclic = semantic*norm, cyclic*norm
    for edited in (semantic, cyclic):
        torch.testing.assert_close(edited.norm(dim=1), norm[:,0],atol=1e-10,rtol=1e-9)
    torch.testing.assert_close((semantic-w).norm(dim=1), (cyclic-w).norm(dim=1),atol=1e-10,rtol=1e-9)
    assert (semantic-cyclic).abs().max() > 1e-5
    return {'unchanged':items, 'semantic':semantic.to(items).reshape_as(items),
            'wrong':cyclic.to(items).reshape_as(items)}, float(((semantic-w).norm(dim=1)/norm[:,0]).mean())
