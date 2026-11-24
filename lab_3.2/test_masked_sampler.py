"""Проверка логики маски на настоящем SpacedSampler из DiffBIR без весов и GPU.

Вместо ControlLDM — «модель», которая всегда предсказывает чистый латент target
(в v-параметризации, как у DiffBIR v2.1). Запуск: python test_masked_sampler.py
"""
import os
import sys

import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, 'DiffBIR')]

import masked_sampler  # noqa: E402,F401  (совместимость torch.Tuple до импорта DiffBIR)
from diffbir.sampler import SpacedSampler  # noqa: E402
from diffbir.utils.common import instantiate_from_config  # noqa: E402

from masked_sampler import (MaskedSpacedSampler, feather_mask, mask_to_latent, to_mask_tensor,  # noqa: E402
                            upscale_mask)

diffusion = instantiate_from_config(OmegaConf.load(os.path.join(HERE, 'DiffBIR', 'configs', 'inference',
                                                                'diffusion_v2.1.yaml')))
abar = torch.tensor(diffusion.alphas_cumprod if hasattr(diffusion, 'alphas_cumprod') else
                    __import__('numpy').cumprod(1 - diffusion.betas), dtype=torch.float32)
H = W = 128  # пиксели выхода -> латент 16x16
target = torch.full((1, 4, H // 8, W // 8), 2.0)   # что «рисует» диффузия
known = torch.full((1, 4, H // 8, W // 8), -1.0)   # что должно остаться вне маски


def fake_model(x, model_t, cond):
    a = abar[model_t].view(-1, 1, 1, 1)
    return (a.sqrt() * x - target) / (1 - a).sqrt()  # v такое, что предсказанный x0 = target


def run(cls, **kw):
    torch.manual_seed(0)
    x_T = torch.randn_like(target)
    sampler = cls(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    return sampler.sample(fake_model, 'cpu', 20, tuple(target.shape), {}, None, 1.0, x_T=x_T.clone(),
                          progress=False, **kw)


def test_full_mask_equals_original():
    m = mask_to_latent(torch.ones(1, 1, H, W))
    assert torch.equal(run(MaskedSpacedSampler, mask=m, x_known=known), run(SpacedSampler))


def test_empty_mask_returns_known():
    m = mask_to_latent(torch.zeros(1, 1, H, W))
    assert torch.allclose(run(MaskedSpacedSampler, mask=m, x_known=known), known)


def test_combined_mask():
    px = torch.zeros(1, 1, H, W)
    px[..., :, : W // 2] = 1  # левая половина
    out = run(MaskedSpacedSampler, mask=mask_to_latent(px, dilate_latent=1), x_known=known)
    lat_w = W // 8
    assert torch.allclose(out[..., :, lat_w // 2 + 1:], known[..., :, lat_w // 2 + 1:])
    assert torch.allclose(out[..., :, : lat_w // 2], target[..., :, : lat_w // 2], atol=1e-3)


def test_mask_helpers():
    m = to_mask_tensor(torch.tensor([[0, 255], [255, 0]], dtype=torch.uint8))
    assert m.shape == (1, 1, 2, 2) and set(m.unique().tolist()) == {0.0, 1.0}
    up = upscale_mask(m, (8, 8))
    assert up[0, 0, 0, 7] == 1 and up[0, 0, 0, 0] == 0

    px = torch.zeros(1, 1, 256, 256)
    px[..., 96:160, 96:160] = 1
    lat = mask_to_latent(px, dilate_latent=1)
    assert lat.shape == (1, 1, 32, 32)
    assert lat[0, 0, 11:21, 11:21].all() and lat.sum() == 10 * 10

    f = feather_mask(px, dilate_px=8, sigma=4)
    assert f[0, 0, 128, 128] > 0.9999                   # центр маски — чистый SR
    assert f[0, 0, 0, 0] == 0.0 and f[0, 0, 96 - 21, 128] == 0  # далеко от маски — чистый bicubic
    assert 0 < f[0, 0, 96 - 8, 128] < 1                # переходная зона


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
