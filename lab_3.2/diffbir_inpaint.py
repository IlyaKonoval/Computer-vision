"""DiffBIR v2.1 + маска: восстановление (SR x4) только выбранной области изображения.

Код DiffBIR не изменяется: используется его модель (SwinIR + ControlLDM на Stable Diffusion 2.1),
а сэмплер SpacedSampler заменяется на MaskedSpacedSampler из masked_sampler.py.

Пример:
    python diffbir_inpaint.py --lq data/lq/realesrgan/0801.png --mask data/masks/combined.png --out out.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DIFFBIR_ROOT = os.path.join(HERE, 'DiffBIR')
WEIGHTS_DIR = os.path.join(HERE, 'weights')
WEIGHTS = {
    'swinir': 'realesrgan_s4_swinir_100k.pth',      # stage 1: SwinIR, обучен на деградациях Real-ESRGAN
    'sd': 'sd2.1-base-zsnr-laionaes5.ckpt',          # Stable Diffusion 2.1 base (zero-SNR, v-prediction)
    'control': 'DiffBIR_v2.1.pt',                    # stage 2: IRControlNet DiffBIR v2.1
}
WEIGHTS_URL = 'https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/'

POS_PROMPT = ('Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - '
              'realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore '
              'detailing, hyper sharpness, perfect without deformations.')
NEG_PROMPT = ('painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, '
              'unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, '
              'jpeg artifacts, deformed, lowres, over-smooth.')

DEFAULTS = dict(steps=50, cfg_scale=6.0, strength=1.0, dilate_latent=1, feather_dilate=8, feather_sigma=4.0,
                vae_tile=512, cldm_tile=512, cldm_stride=256)

if HERE not in sys.path:
    sys.path.insert(0, HERE)
if DIFFBIR_ROOT not in sys.path:
    sys.path.insert(0, DIFFBIR_ROOT)

from masked_sampler import (MaskedSpacedSampler, blend, feather_mask, mask_to_latent,  # noqa: E402
                            to_mask_tensor, upscale_mask)


def _load_ckpt(name):
    path = os.path.join(WEIGHTS_DIR, WEIGHTS[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found, download it from {WEIGHTS_URL}{WEIGHTS[name]}')
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    if next(iter(sd)).startswith('module.'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    return sd


class DiffBIR:
    """Модели DiffBIR v2.1: SwinIR (очистка) -> ControlLDM (генерация деталей) -> VAE."""

    def __init__(self, device='cuda', dtype=torch.float16):
        from omegaconf import OmegaConf
        from diffbir.utils.common import instantiate_from_config

        cfg = lambda name: OmegaConf.load(os.path.join(DIFFBIR_ROOT, 'configs', 'inference', name))
        self.device, self.dtype = device, dtype

        self.cleaner = instantiate_from_config(cfg('swinir.yaml'))
        self.cleaner.load_state_dict(_load_ckpt('swinir'), strict=True)
        self.cleaner.eval().to(device)

        self.cldm = instantiate_from_config(cfg('cldm.yaml'))
        sd = _load_ckpt('sd')
        _, missing = self.cldm.load_pretrained_sd(sd)
        del sd
        # в чекпойнте SD 2.x нет последнего слоя CLIP и проекций: берётся предпоследний слой (penultimate)
        unused_clip = ('cond_stage_model.model.transformer.resblocks.23.', 'cond_stage_model.model.logit_scale',
                       'cond_stage_model.model.text_projection')
        missing = [k for k in missing if not k.startswith(unused_clip)]
        assert not missing, f'missing SD weights: {sorted(missing)[:5]}'
        self.cldm.load_controlnet_from_ckpt(_load_ckpt('control'))
        self.cldm.eval().to(device)
        self.cldm.cast_dtype(dtype)

        self.diffusion = instantiate_from_config(cfg('diffusion_v2.1.yaml')).to(device)
        self.text_cache = {}
        self.cldm.clip.to('cpu')  # текстовый энкодер (1.4 ГБ) нужен только для промптов -> держим на CPU
        for prompt in (POS_PROMPT, NEG_PROMPT):
            self.text(prompt)
        if device == 'cuda':
            # На Windows при нехватке видеопамяти драйвер молча выносит её в RAM и всё замедляется
            # в разы; лимит превращает это в явную ошибку OOM.
            torch.cuda.set_per_process_memory_fraction(0.92)
            torch.cuda.empty_cache()

    @torch.no_grad()
    def text(self, prompt):
        if prompt not in self.text_cache:
            self.cldm.clip.to(self.device)
            with torch.autocast('cuda', dtype=self.dtype, enabled=self.device == 'cuda'):
                self.text_cache[prompt] = self.cldm.clip.encode([prompt])
            self.cldm.clip.to('cpu')
            torch.cuda.empty_cache()
        return self.text_cache[prompt]

    @torch.no_grad()
    def clean(self, img):
        """Stage 1: SwinIR убирает шум/размытие/JPEG; вход и выход (1,3,H,W) в [0,1]."""
        h, w = img.shape[2:]
        ph, pw = (-h) % 64, (-w) % 64
        out = self.cleaner(F.pad(img, (0, pw, 0, ph)))
        return out[:, :, :h, :w].clamp(0, 1)

    def vae_encode(self, img01, tile):
        tiled = min(img01.shape[2:]) > tile
        return self.cldm.vae_encode(img01 * 2 - 1, sample=False, tiled=tiled, tile_size=tile)

    def vae_decode(self, z, tile):
        tiled = min(z.shape[2:]) * 8 > tile
        return self.cldm.vae_decode(z, tiled=tiled, tile_size=tile // 8)


@torch.no_grad()
def masked_restore(model, lq_up, mask_px, seed=231, pos_prompt=POS_PROMPT, neg_prompt=NEG_PROMPT, **kw):
    """SR-inpainting.

    lq_up   — LQ, увеличенный bicubic до выходного размера, (1,3,H,W) в [0,1], H и W кратны 64
    mask_px — маска выходного разрешения (1,1,H,W), 1 = восстанавливать
    Возвращает (out, sr): out — склейка, sr — выход диффузии до склейки (None при пустой маске).
    """
    from diffbir.utils.common import wavelet_reconstruction

    p = {**DEFAULTS, **kw}
    mask_px = mask_px.to(lq_up)
    if not bool((mask_px > 0.5).any()):
        return lq_up.clone(), None  # пустая маска: улучшать нечего, диффузию не запускаем
    full = bool((mask_px > 0.5).all())

    torch.manual_seed(seed)
    with torch.autocast('cuda', dtype=model.dtype):
        cond_img = model.clean(lq_up)
        c_img = model.vae_encode(cond_img, p['vae_tile'])
        cond = {'c_txt': model.text(pos_prompt), 'c_img': c_img}
        uncond = {'c_txt': model.text(neg_prompt), 'c_img': c_img}
        m_lat = z_known = None
        if not full:
            m_lat = mask_to_latent(mask_px, p['dilate_latent'])
            z_known = model.vae_encode(lq_up, p['vae_tile'])  # латент того, что остаётся вне маски
        torch.cuda.empty_cache()

        model.cldm.control_scales = [p['strength']] * 13
        sampler = MaskedSpacedSampler(model.diffusion.betas, model.diffusion.parameterization, rescale_cfg=False)
        x_T = torch.randn(c_img.shape, device=c_img.device, dtype=torch.float32)
        # SD 2.1-base обучена на 512x512: U-Net считается тайлами 512 px с перекрытием (как в DiffBIR),
        # иначе attention на латенте 128x128 не помещается в 8 ГБ
        tiled = min(c_img.shape[2:]) * 8 > p['cldm_tile']
        z = sampler.sample(model.cldm, model.device, p['steps'], tuple(c_img.shape), cond, uncond, p['cfg_scale'],
                           tiled=tiled, tile_size=p['cldm_tile'] // 8, tile_stride=p['cldm_stride'] // 8,
                           x_T=x_T, progress=False, mask=m_lat, x_known=z_known)
        torch.cuda.empty_cache()
        x = model.vae_decode(z, p['vae_tile'])
    sr = wavelet_reconstruction((x.float() + 1) / 2, cond_img.float()).clamp(0, 1)

    if full:
        return sr, sr
    alpha = feather_mask(mask_px, p['feather_dilate'], p['feather_sigma'])
    return blend(sr, lq_up, alpha), sr


# ---------------- ввод/вывод ----------------

def load_lq(path, scale=4, device='cuda'):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    W, H = int(round(w * scale / 64)) * 64, int(round(h * scale / 64)) * 64
    up = np.asarray(img.resize((W, H), Image.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(up).permute(2, 0, 1)[None].to(device)


def load_mask(path, size, device='cuda'):
    m = to_mask_tensor(np.asarray(Image.open(path).convert('L')), device)
    return upscale_mask(m, size)


def save_img(x, path):
    arr = (x[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lq', required=True)
    ap.add_argument('--mask', required=True, help='бинарная маска с разрешением входа (png)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=DEFAULTS['steps'])
    ap.add_argument('--cfg_scale', type=float, default=DEFAULTS['cfg_scale'])
    ap.add_argument('--seed', type=int, default=231)
    args = ap.parse_args()

    model = DiffBIR()
    lq_up = load_lq(args.lq)
    mask = load_mask(args.mask, lq_up.shape[-2:])
    out, _ = masked_restore(model, lq_up, mask, seed=args.seed, steps=args.steps, cfg_scale=args.cfg_scale)
    save_img(out, args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
