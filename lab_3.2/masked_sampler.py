import torch
import torch.nn.functional as F

LATENT_FACTOR = 8  # VAE SDXL уменьшает изображение в 8 раз

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
    """Плавная маска для финальной склейки в пикселях.

    Сначала расширяем маску на dilate_px (там SR ещё корректен, т.к. латентная маска шире),
    затем размываем. Пиксели дальше ~dilate_px + 3*sigma от маски остаются ровно bicubic.
    """
    m = gaussian_blur(dilate(mask_px, dilate_px), sigma)
    return m.clamp(0, 1)


def blend(fg, bg, alpha):
    return alpha * fg + (1 - alpha) * bg


# ---------------- сэмплер ----------------

class MaskedSamplingMixin:
    """Переопределяет __call__ у RestoreEDMSampler: добавляет параметры mask и x_known.

    mask    — латентная маска (B или 1, 1, h, w), 1 = восстанавливать, 0 = оставить как есть;
    x_known — латент изображения, которое должно остаться вне маски (B, 4, h, w).
    При mask=None или маске из одних единиц поведение совпадает с исходным сэмплером
    (в т.ч. по случайным числам, т.е. результат идентичен при том же seed).
    """

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, x_center=None, control_scale=1.0,
                 use_linear_control_scale=False, control_scale_start=0.0, mask=None, x_known=None):
        if mask is None or bool((mask > 0.5).all()):
            return super().__call__(denoiser, x, cond, uc, num_steps, x_center, control_scale,
                                    use_linear_control_scale, control_scale_start)
        assert x_known is not None, 'x_known is required when mask is given'

        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)
        mask = mask.to(device=x.device, dtype=x.dtype)
        x_known = x_known.to(device=x.device, dtype=x.dtype)

        x = self.replace_known(x, x_known, mask, sigmas[0])
        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i], s_in * sigmas[i + 1], denoiser, x, cond, uc, gamma, x_center,
                control_scale=control_scale,
                use_linear_control_scale=use_linear_control_scale,
                control_scale_start=control_scale_start,
            )
            x = self.replace_known(x, x_known, mask, sigmas[i + 1])
        return x

    @staticmethod
    def replace_known(x, x_known, mask, sigma):
        """Вне маски подставить известный латент с шумом уровня sigma (forward-процесс EDM)."""
        noised_known = x_known + sigma * torch.randn_like(x_known)
        return mask * x + (1 - mask) * noised_known


def make_masked_sampler_class(base):
    return type('Masked' + base.__name__, (MaskedSamplingMixin, base), {})


try:  # внутри репозитория SUPIR — готовый класс для instantiate_from_config
    from sgm.modules.diffusionmodules.sampling import RestoreEDMSampler

    MaskedRestoreEDMSampler = make_masked_sampler_class(RestoreEDMSampler)
except ImportError:
    MaskedRestoreEDMSampler = None
