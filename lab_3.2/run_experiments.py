import argparse
import csv
import glob
import os
import time

import torch

from diffbir_inpaint import DEFAULTS, DiffBIR, load_lq, load_mask, masked_restore, save_img

HERE = os.path.dirname(os.path.abspath(__file__))
DEGRADATIONS = ['bicubic', 'unknown', 'realesrgan']
MASKS = ['ones', 'zeros', 'combined']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=DEFAULTS['steps'])
    ap.add_argument('--cfg_scale', type=float, default=DEFAULTS['cfg_scale'])
    ap.add_argument('--seed', type=int, default=231)
    ap.add_argument('--data', default=os.path.join(HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    ap.add_argument('--limit', type=int, default=0, help='обработать только первые N изображений (для пробы)')
    args = ap.parse_args()

    ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.data, 'hr', '*.png')))
    if args.limit:
        ids = ids[:args.limit]

    model = DiffBIR()

    timing_path = os.path.join(args.out, 'timing.csv')
    os.makedirs(args.out, exist_ok=True)
    new_file = not os.path.exists(timing_path)
    with open(timing_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['image', 'degradation', 'mask', 'seconds', 'max_mem_gb'])
        for deg in DEGRADATIONS:
            for img_id in ids:
                lq_up = load_lq(os.path.join(args.data, 'lq', deg, f'{img_id}.png'))
                for mask_name in MASKS:
                    out_path = os.path.join(args.out, 'sr', deg, mask_name, f'{img_id}.png')
                    if os.path.exists(out_path):
                        continue
                    mask = load_mask(os.path.join(args.data, 'masks', f'{mask_name}.png'), lq_up.shape[-2:])
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    t0 = time.time()
                    out, raw = masked_restore(model, lq_up, mask, seed=args.seed, steps=args.steps,
                                              cfg_scale=args.cfg_scale)
                    torch.cuda.synchronize()
                    dt = time.time() - t0
                    save_img(out, out_path)
                    if raw is not None and mask_name == 'combined':
                        save_img(raw, out_path.replace('.png', '_raw.png'))
                    mem = torch.cuda.max_memory_allocated() / 2 ** 30
                    writer.writerow([img_id, deg, mask_name, f'{dt:.2f}', f'{mem:.2f}'])
                    f.flush()
                    print(f'{deg:10s} {img_id} {mask_name:8s} {dt:6.1f}s  {mem:.1f} GB')


if __name__ == '__main__':
    main()
