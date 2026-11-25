import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

LATENT_FACTOR = 8

def to_mask_tensor(mask, device=None):
    m = torch.as_tensor(mask, device=device).float()
    while m.ndim < 4:
        m = m.unsqueeze(0)
    return (m > 0.5 * m.max().clamp(min=1e-8)).float() if m.max() > 0 else m


def upscale_mask(mask, size):
    return F.interpolate(mask, size=size, mode='nearest')


def dilate(mask, radius):
    if radius <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)


def mask_to_latent(mask_px, dilate_latent=1):
    m = F.max_pool2d(mask_px, kernel_size=LATENT_FACTOR, stride=LATENT_FACTOR)
    return dilate(m, dilate_latent)


def gaussian_blur(mask, sigma):
    if sigma <= 0:
        return mask
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, device=mask.device, dtype=mask.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    m = F.pad(mask, (r, r, r, r), mode='replicate')
    m = F.conv2d(m, k.view(1, 1, 1, -1))
    m = F.conv2d(m, k.view(1, 1, -1, 1))
    return m


def feather_mask(mask_px, dilate_px=8, sigma=4.0):
    m = gaussian_blur(dilate(mask_px, dilate_px), sigma)
    return m.clamp(0, 1)


def blend(fg, bg, alpha):
    return alpha * fg + (1 - alpha) * bg


# ---------------- сэмплер ----------------

class MaskedSamplingMixin:
    @torch.no_grad()
    def sample(self, model, device, steps, x_size, cond, uncond, cfg_scale, tiled=False, tile_size=-1,
               tile_stride=-1, x_T=None, progress=True, mask=None, x_known=None):
        if mask is None or bool((mask > 0.5).all()):
            return super().sample(model, device, steps, x_size, cond, uncond, cfg_scale, tiled, tile_size,
                                  tile_stride, x_T, progress)
        assert x_known is not None, 'x_known is required when mask is given'

        self.make_schedule(steps)
        self.to(device)
        if tiled:
            from diffbir.utils.common import make_tiled_fn
            forward = model.forward
            model.forward = make_tiled_fn(
                lambda x_tile, t, c, hi, hi_end, wi, wi_end: forward(
                    x_tile, t, {'c_txt': c['c_txt'], 'c_img': c['c_img'][..., hi:hi_end, wi:wi_end]}),
                tile_size, tile_stride)
        x = torch.randn(x_size, device=device, dtype=torch.float32) if x_T is None else x_T
        mask = mask.to(device=device, dtype=x.dtype)
        x_known = x_known.to(device=device, dtype=x.dtype)

        total_steps = len(self.timesteps)
        x = self.replace_known(x, x_known, mask, total_steps - 1)
        timesteps = np.flip(self.timesteps)
        bs = x_size[0]
        for i, step in enumerate(tqdm(timesteps, total=total_steps, disable=not progress)):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            x = self.p_sample(model, x, model_t, t, cond, uncond, self.get_cfg_scale(cfg_scale, step))
            # x теперь соответствует шагу total_steps - i - 2 (или чистому латенту, если шаг был последним)
            x = self.replace_known(x, x_known, mask, total_steps - i - 2)
        if tiled:
            model.forward = forward
        return x

    def replace_known(self, x, x_known, mask, idx):
        if idx < 0:
            noised = x_known
        else:
            a = self.sqrt_alphas_cumprod[idx]
            s = self.sqrt_one_minus_alphas_cumprod[idx]
            noised = a * x_known + s * torch.randn_like(x_known)
        return mask * x + (1 - mask) * noised


def make_masked_sampler_class(base):
    return type('Masked' + base.__name__, (MaskedSamplingMixin, base), {})


if not hasattr(torch, 'Tuple'):
    import typing
    torch.Tuple = typing.Tuple

try:
    from diffbir.sampler import SpacedSampler

    MaskedSpacedSampler = make_masked_sampler_class(SpacedSampler)
except ImportError:
    MaskedSpacedSampler = None
