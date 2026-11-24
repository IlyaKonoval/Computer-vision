import torch

from masked_sampler import (feather_mask, make_masked_sampler_class, mask_to_latent, to_mask_tensor,
                            upscale_mask)


class StubRestoreEDMSampler:
    s_churn, s_tmin, s_tmax, s_noise = 5.0, 0.0, float('inf'), 1.01

    def __init__(self, num_steps=20):
        self.num_steps = num_steps

    def prepare_sampling_loop(self, x, cond, uc, num_steps):
        sigmas = torch.cat([torch.linspace(14.6, 0.03, self.num_steps), torch.zeros(1)])
        x = x * torch.sqrt(1.0 + sigmas[0] ** 2)
        return x, x.new_ones([x.shape[0]]), sigmas, len(sigmas), cond, uc

    def get_sigma_gen(self, num_sigmas):
        return range(num_sigmas - 1)

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0, x_center=None, **kw):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            x = x + torch.randn_like(x) * self.s_noise * (sigma_hat ** 2 - sigma ** 2).view(-1, 1, 1, 1) ** 0.5
        denoised = denoiser(x, sigma_hat, cond, 1.0)
        d = (x - denoised) / sigma_hat.view(-1, 1, 1, 1)
        return x + d * (next_sigma - sigma_hat).view(-1, 1, 1, 1)

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, x_center=None, control_scale=1.0,
                 use_linear_control_scale=False, control_scale_start=0.0):
        x, s_in, sigmas, n, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)
        for i in self.get_sigma_gen(n):
            gamma = min(self.s_churn / (n - 1), 2 ** 0.5 - 1)
            x = self.sampler_step(s_in * sigmas[i], s_in * sigmas[i + 1], denoiser, x, cond, uc, gamma, x_center)
        return x


Masked = make_masked_sampler_class(StubRestoreEDMSampler)
H = W = 128  # пиксели выхода -> латент 16x16
target = torch.full((1, 4, H // 8, W // 8), 2.0)   # что «рисует» диффузия
known = torch.full((1, 4, H // 8, W // 8), -1.0)   # что должно остаться вне маски
denoiser = lambda x, sigma, c, cs: target.expand_as(x)


def run(mask_lat, seed=0):
    torch.manual_seed(seed)
    noise = torch.randn_like(target)
    return Masked()(denoiser, noise, cond={}, mask=mask_lat, x_known=known)


def run_base(seed=0):
    torch.manual_seed(seed)
    noise = torch.randn_like(target)
    return StubRestoreEDMSampler()(denoiser, noise, cond={})


def test_full_mask_equals_original():
    m = mask_to_latent(torch.ones(1, 1, H, W))
    assert torch.equal(run(m), run_base()), 'mask=1 must reproduce the original sampler bit-exactly'


def test_empty_mask_returns_known():
    m = mask_to_latent(torch.zeros(1, 1, H, W))
    assert torch.allclose(run(m), known), 'mask=0 must return x_known'


def test_combined_mask():
    px = torch.zeros(1, 1, H, W)
    px[..., :, : W // 2] = 1  # левая половина
    m = mask_to_latent(px, dilate_latent=1)
    out = run(m)
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
