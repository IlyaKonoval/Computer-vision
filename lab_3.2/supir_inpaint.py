import argparse
import copy
import gc
import os
import sys
import types

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from masked_sampler import blend, feather_mask, mask_to_latent, to_mask_tensor, upscale_mask  # noqa: E402

A_PROMPT = ('Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - '
            'realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore '
            'detailing, hyper sharpness, perfect without deformations.')
N_PROMPT = ('painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, '
            'unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, '
            'jpeg artifacts, deformed, lowres, over-smooth')

DEFAULTS = dict(num_steps=50, restoration_scale=-1.0, s_churn=5.0, s_noise=1.01, cfg_scale=4.0,
                cfg_scale_start=1.0, use_linear_cfg=True, control_scale=1.0, color_fix='Wavelet',
                dilate_latent=1, feather_dilate=8, feather_sigma=4.0)


def _has_xformers():
    try:
        import xformers.ops  # noqa: F401
        return True
    except Exception:
        return False


def _load_sd(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        try:  # mmap не держит весь чекпойнт в RAM
            sd = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        except RuntimeError:
            sd = torch.load(path, map_location='cpu', weights_only=False)
    return sd.get('state_dict', sd)


def load_supir(supir_root, sdxl_ckpt, supir_ckpt, clip1_path=None, clip2_path=None, device='cuda'):
    supir_root = os.path.abspath(supir_root)
    if supir_root not in sys.path:
        sys.path.insert(0, supir_root)
    ckpt_mod = types.ModuleType('CKPT_PTH')
    ckpt_mod.SDXL_CLIP1_PATH = clip1_path
    ckpt_mod.SDXL_CLIP2_CKPT_PTH = clip2_path
    ckpt_mod.LLAVA_CLIP_PATH = None
    ckpt_mod.LLAVA_MODEL_PATH = None
    sys.modules['CKPT_PTH'] = ckpt_mod

    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config
    import SUPIR.utils.tilevae as tilevae

    config = OmegaConf.load(os.path.join(supir_root, 'options', 'SUPIR_v0.yaml'))
    if not _has_xformers():
        config.model.params.first_stage_config.params.ddconfig.attn_type = 'vanilla'
        tilevae.is_xformers_available = False
        print('xformers not found -> using PyTorch attention')

    model = instantiate_from_config(config.model).cpu().eval()
    model.model.half()
    for path in (sdxl_ckpt, supir_ckpt):
        sd = _load_sd(path)
        model.load_state_dict(sd, strict=False)
        print(f'loaded {path}')
        del sd
        gc.collect()

    model.ae_dtype = torch.float32
    model.model.dtype = torch.float16
    model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
    model.to(device)
    return model


@torch.no_grad()
def encode_prompts(model, prompt='', a_prompt=A_PROMPT, n_prompt=N_PROMPT, offload=True):
    device = next(model.model.parameters()).device
    dummy = torch.zeros(1, 4, 128, 128, device=device)
    c, uc = model.prepare_condition(dummy, [prompt], a_prompt, n_prompt, 1)
    if offload:
        model.conditioner.to('cpu')
        torch.cuda.empty_cache()
    return {'c': c, 'uc': uc}


def build_sampler(model, num_steps, restoration_scale, s_churn, s_noise, cfg_scale, cfg_scale_start,
                  use_linear_cfg):
    from sgm.util import instantiate_from_config
    cfg = copy.deepcopy(model.sampler_config)
    cfg.target = 'masked_sampler.MaskedRestoreEDMSampler'
    cfg.params.num_steps = num_steps
    cfg.params.guider_config.params.scale_min = cfg_scale
    cfg.params.guider_config.params.scale = cfg_scale_start if use_linear_cfg else cfg_scale
    cfg.params.restore_cfg = restoration_scale
    cfg.params.s_churn = s_churn
    cfg.params.s_noise = s_noise
    return instantiate_from_config(cfg)


@torch.no_grad()
def masked_sample(model, cond, lq_up, mask_px, seed=1234, **kw):
    from pytorch_lightning import seed_everything
    from SUPIR.utils.colorfix import adaptive_instance_normalization, wavelet_reconstruction

    p = {**DEFAULTS, **kw}
    mask_px = mask_px.to(lq_up)
    if not bool((mask_px > 0.5).any()):
        return lq_up.clone(), None
    full = bool((mask_px > 0.5).all())

    sampler = build_sampler(model, p['num_steps'], p['restoration_scale'], p['s_churn'], p['s_noise'],
                            p['cfg_scale'], p['cfg_scale_start'], p['use_linear_cfg'])
    seed_everything(seed)

    # шаги 1-2 как в batchify_sample: denoise-encoder SUPIR -> латент-условие и «центр» для restore_cfg
    _z = model.encode_first_stage_with_denoise(lq_up, use_sample=False)
    x_stage1 = model.decode_first_stage(_z)
    z_stage1 = model.encode_first_stage(x_stage1)
    c = dict(cond['c'], control=_z)
    uc = dict(cond['uc'], control=_z)
    noised_z = torch.randn_like(_z)

    if full:
        m_lat, z_known = None, None
    else:
        m_lat = mask_to_latent(mask_px, p['dilate_latent'])
        z_known = model.encode_first_stage(lq_up)

    denoiser = lambda x, sigma, cc, control_scale: model.denoiser(model.model, x, sigma, cc, control_scale)
    z = sampler(denoiser, noised_z, cond=c, uc=uc, x_center=z_stage1, control_scale=p['control_scale'],
                mask=m_lat, x_known=z_known)
    sr = model.decode_first_stage(z)
    if p['color_fix'] == 'Wavelet':
        sr = wavelet_reconstruction(sr, x_stage1)
    elif p['color_fix'] == 'AdaIn':
        sr = adaptive_instance_normalization(sr, x_stage1)

    if full:
        return sr, sr
    alpha = feather_mask(mask_px, p['feather_dilate'], p['feather_sigma'])
    return blend(sr, lq_up, alpha), sr


def load_lq(path, scale=4, device='cuda'):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    W, H = int(round(w * scale / 64)) * 64, int(round(h * scale / 64)) * 64
    up = np.asarray(img.resize((W, H), Image.BICUBIC), dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(up).permute(2, 0, 1)[None].to(device)


def load_mask(path, size, device='cuda'):
    m = to_mask_tensor(np.asarray(Image.open(path).convert('L')), device)
    return upscale_mask(m, size)


def save_img(x, path):
    arr = ((x[0].clamp(-1, 1).permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(arr).save(path)


def add_model_args(ap):
    ap.add_argument('--supir_root', required=True)
    ap.add_argument('--sdxl_ckpt', required=True)
    ap.add_argument('--supir_ckpt', required=True)
    ap.add_argument('--clip1_path', default=None)
    ap.add_argument('--clip2_path', default=None)
    ap.add_argument('--steps', type=int, default=DEFAULTS['num_steps'])
    ap.add_argument('--seed', type=int, default=1234)


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument('--lq', required=True)
    ap.add_argument('--mask', required=True, help='бинарная маска с разрешением входа (png)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    model = load_supir(args.supir_root, args.sdxl_ckpt, args.supir_ckpt, args.clip1_path, args.clip2_path)
    cond = encode_prompts(model)
    lq_up = load_lq(args.lq)
    mask = load_mask(args.mask, lq_up.shape[-2:])
    out, _ = masked_sample(model, cond, lq_up, mask, seed=args.seed, num_steps=args.steps)
    save_img(out, args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
