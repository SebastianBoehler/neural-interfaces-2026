"""Official frozen visual encoders, strict local checkpoint loading and no network."""
import hashlib
import os
import socket
import sys
from contextlib import contextmanager

import torch
from assets import verify


@contextmanager
def offline():
    original = socket.socket.connect
    def reject(*args, **kwargs):
        raise RuntimeError('visual encoder attempted network access during offline execution')
    socket.socket.connect = reject
    try:
        yield
    finally:
        socket.socket.connect = original


def build(root, variant, device):
    if variant not in ('pretrained', 'random') or device not in ('cpu', 'cuda'):
        raise ValueError('explicit registered encoder/device required')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('requested A100 CUDA route unavailable')
    verify(root)
    os.environ['XFORMERS_DISABLED'] = '1'
    sys.path.insert(0, str(root / 'assets/dinov2'))
    with offline():
        from dinov2.hub.backbones import dinov2_vits14
        torch.manual_seed(319)
        model = dinov2_vits14(pretrained=False)
        if variant == 'pretrained':
            weights = torch.load(root / 'assets/dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=True)
            model.load_state_dict(weights, strict=True)
    model.requires_grad_(False); model.eval(); model.to(device)
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('backbone unexpectedly trainable')
    return model


def state_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def features(model, image, tokens):
    z = model.forward_features(image)
    if z['x_norm_patchtokens'].shape[1:] != (tokens, 384) or z['x_norm_clstoken'].shape[1:] != (384,):
        raise ValueError('official backbone patch layout or dimension changed')
    result = torch.cat([z['x_norm_clstoken'], z['x_norm_patchtokens'].mean(1)], 1)
    if result.shape != (len(image), 768) or not torch.isfinite(result).all():
        raise ValueError('invalid frozen visual features')
    return result
