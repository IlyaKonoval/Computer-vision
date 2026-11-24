
import argparse
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEGRADATIONS = ['bicubic', 'unknown', 'realesrgan']
MASKS = ['ones', 'combined', 'zeros']
MASK_LABELS = {'ones': 'маска = 1 (DiffBIR везде)', 'combined': 'комбинированная', 'zeros': 'маска = 0 (bicubic)'}
COLORS = {'ones': '#2a78d6', 'combined': '#1baf7a', 'zeros': '#eb6834'}  # фиксированные категориальные слоты
TRANSITION = 24  # px: зона плавной склейки (dilate 8 + 3*sigma 4 + запас) — исключается из «вне маски»


def load(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


def psnr(a, b, region=None):
    se = (a - b) ** 2
    mse = se[region].mean() if region is not None else se.mean()
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)


def to_t(x, device):
    return torch.from_numpy(x).permute(2, 0, 1)[None].to(device)


class Metrics:
    def __init__(self, device):
        import lpips
        import pyiqa
        self.device = device
        self.lpips = lpips.LPIPS(net='alex', spatial=True, verbose=False).to(device).eval()
        self.clipiqa = pyiqa.create_metric('clipiqa', device=device)

    @torch.no_grad()
    def lpips_map(self, a, b):
        return self.lpips(to_t(a, self.device) * 2 - 1, to_t(b, self.device) * 2 - 1)[0, 0].cpu().numpy()

    @torch.no_grad()
    def clip(self, a):
        return float(self.clipiqa(to_t(a, self.device)).item())


def regions(mask_path, size):
    m = np.asarray(Image.open(mask_path).convert('L')) > 127
    m = np.asarray(Image.fromarray(m.astype(np.uint8)).resize(size, Image.NEAREST)) > 0
    mt = torch.from_numpy(m.astype(np.float32))[None, None]
    near = F.max_pool2d(mt, 2 * TRANSITION + 1, 1, TRANSITION)[0, 0].numpy() > 0
    return m, ~near  # внутри маски; вне маски с отступом от зоны склейки


def crops(h, w):
    """Кропы для CLIP-IQA: центральный 512 целиком внутри круга, 4 угла 192 — целиком вне."""
    c = 512
    inside = [(slice((h - c) // 2, (h + c) // 2), slice((w - c) // 2, (w + c) // 2))]
    s = 192
    outside = [(slice(y, y + s), slice(x, x + s)) for y in (0, h - s) for x in (0, w - s)]
    return inside, outside


def compute(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    met = Metrics(device)
    rows = []
    ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.data, 'hr', '*.png')))
    for img_id in ids:
        hr = load(os.path.join(args.data, 'hr', f'{img_id}.png'))
        h, w = hr.shape[:2]
        inside, outside = regions(os.path.join(args.data, 'masks', 'combined.png'), (w, h))
        c_in, c_out = crops(h, w)
        for deg in DEGRADATIONS:
            outs = {}
            for mask in MASKS:
                p = os.path.join(args.results, 'sr', deg, mask, f'{img_id}.png')
                if os.path.exists(p):
                    outs[mask] = load(p)
            for mask, sr in outs.items():
                lm = met.lpips_map(sr, hr)
                row = dict(image=img_id, degradation=deg, mask=mask,
                           psnr=psnr(sr, hr), lpips=float(lm.mean()), clipiqa=met.clip(sr),
                           psnr_in=psnr(sr, hr, inside), psnr_out=psnr(sr, hr, outside),
                           lpips_in=float(lm[inside].mean()), lpips_out=float(lm[outside].mean()),
                           clipiqa_in=np.mean([met.clip(sr[r]) for r in c_in]),
                           clipiqa_out=np.mean([met.clip(sr[r]) for r in c_out]))
                # проверка корректности склейки: вне маски combined == bicubic, внутри ~ full DiffBIR
                if mask == 'combined' and 'zeros' in outs:
                    row['psnr_out_vs_bicubic'] = psnr(sr, outs['zeros'], outside)
                if mask == 'combined' and 'ones' in outs:
                    row['psnr_in_vs_full'] = psnr(sr, outs['ones'], inside)
                rows.append(row)
                print(img_id, deg, mask, f"PSNR={row['psnr']:.2f} LPIPS={row['lpips']:.3f} "
                                         f"CLIPIQA={row['clipiqa']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results, 'metrics.csv'), index=False)
    return df


# ---------------- графики и таблицы ----------------

def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#888')
    ax.tick_params(colors='#444')
    ax.grid(axis='y', color='#e4e4e4', linewidth=0.8)
    ax.set_axisbelow(True)


def bar_groups(ax, df, metric, title):
    width = 0.26
    x = np.arange(len(DEGRADATIONS))
    for k, mask in enumerate(MASKS):
        vals = [df[(df.degradation == d) & (df['mask'] == mask)][metric].mean() for d in DEGRADATIONS]
        bars = ax.bar(x + (k - 1) * width, vals, width - 0.02, color=COLORS[mask], label=MASK_LABELS[mask])
        for b, v in zip(bars, vals):
            ax.annotate(f'{v:.2f}' if v < 1 else f'{v:.1f}', (b.get_x() + b.get_width() / 2, v), ha='center', va='bottom',
                        fontsize=7, color='#444', xytext=(0, 2), textcoords='offset points')
    ax.set_xticks(x, DEGRADATIONS)
    ax.set_title(title, fontsize=10, loc='left')
    style(ax)


def top_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))


def plot_overall(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    bar_groups(axes[0], df, 'psnr', 'PSNR, дБ (выше — лучше)')
    bar_groups(axes[1], df, 'lpips', 'LPIPS (ниже — лучше)')
    bar_groups(axes[2], df, 'clipiqa', 'CLIP-IQA (выше — лучше)')
    top_legend(fig, axes[0])
    fig.savefig(os.path.join(out_dir, 'metrics_overall.png'), dpi=150)
    plt.close(fig)


def plot_regions(df, out_dir):
    """Метрики отдельно внутри и вне круга: combined должен совпадать с ones внутри и с zeros снаружи."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, (metric, name) in enumerate([('psnr', 'PSNR, дБ'), ('lpips', 'LPIPS'), ('clipiqa', 'CLIP-IQA')]):
        bar_groups(axes[0, col], df, f'{metric}_in', f'{name} — внутри маски')
        bar_groups(axes[1, col], df, f'{metric}_out', f'{name} — вне маски')
    top_legend(fig, axes[0, 0])
    fig.savefig(os.path.join(out_dir, 'metrics_regions.png'), dpi=150)
    plt.close(fig)


def summary_table(df, path):
    cols = ['psnr', 'lpips', 'clipiqa', 'psnr_in', 'psnr_out', 'lpips_in', 'lpips_out', 'clipiqa_in', 'clipiqa_out']
    g = df.groupby(['degradation', 'mask'])[cols].mean().reindex(
        pd.MultiIndex.from_product([DEGRADATIONS, MASKS])).dropna(how='all')
    lines = ['| Деградация | Маска | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 2)]
    for (d, m), r in g.iterrows():
        lines.append(f'| {d} | {m} | ' + ' | '.join(f'{r[c]:.3f}' for c in cols) + ' |')
    if 'psnr_out_vs_bicubic' in df:
        c = df[df['mask'] == 'combined']
        lines += ['', f"Склейка: PSNR(combined, bicubic) вне маски = {c['psnr_out_vs_bicubic'].min():.1f} дБ (мин.), "
                      f"PSNR(combined, full DiffBIR) внутри маски = {c['psnr_in_vs_full'].mean():.1f} дБ (сред.)"]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def contour(img, mask_px, color=(1.0, 0.85, 0.0)):
    m = torch.from_numpy(mask_px.astype(np.float32))[None, None]
    edge = (F.max_pool2d(m, 5, 1, 2) - (-F.max_pool2d(-m, 5, 1, 2)))[0, 0].numpy() > 0
    out = img.copy()
    out[edge] = color
    return out


def comparison_grids(args, out_dir):
    """Для каждого изображения: LQ | bicubic | DiffBIR (маска 1) | маска combined | HR, + увеличенный фрагмент."""
    ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.data, 'hr', '*.png')))
    titles = ['LQ (x4 меньше)', 'маска = 0 (bicubic)', 'маска = 1 (DiffBIR)', 'комбинированная', 'HR (эталон)']
    for deg in DEGRADATIONS:
        rows = []
        for img_id in ids:
            paths = [os.path.join(args.results, 'sr', deg, m, f'{img_id}.png') for m in ('zeros', 'ones', 'combined')]
            if not all(os.path.exists(p) for p in paths):
                continue
            hr = load(os.path.join(args.data, 'hr', f'{img_id}.png'))
            lq = load(os.path.join(args.data, 'lq', deg, f'{img_id}.png'))
            lq = np.asarray(Image.fromarray((lq * 255).astype(np.uint8)).resize(hr.shape[1::-1], Image.NEAREST),
                            dtype=np.float32) / 255
            inside, _ = regions(os.path.join(args.data, 'masks', 'combined.png'), hr.shape[1::-1])
            zeros, ones, comb = (load(p) for p in paths)
            rows.append([lq, zeros, ones, contour(comb, inside), hr])
        if not rows:
            continue
        fig, axes = plt.subplots(len(rows), 5, figsize=(15, 3 * len(rows)))
        axes = np.atleast_2d(axes)
        for i, row in enumerate(rows):
            for j, im in enumerate(row):
                axes[i, j].imshow(im)
                axes[i, j].axis('off')
                if i == 0:
                    axes[i, j].set_title(titles[j], fontsize=10)
        fig.suptitle(f'Деградация: {deg}', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'grid_{deg}.jpg'), dpi=110)
        plt.close(fig)

    # крупный план на границе маски: видно, что шва нет и снаружи — bicubic
    for img_id in ids[:2]:
        deg = 'realesrgan'
        paths = {m: os.path.join(args.results, 'sr', deg, m, f'{img_id}.png') for m in ('zeros', 'ones', 'combined')}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        hr = load(os.path.join(args.data, 'hr', f'{img_id}.png'))
        h, w = hr.shape[:2]
        sl = (slice(h // 2 - 192, h // 2 + 192), slice(w // 2 + 220, w // 2 + 604))  # правый край круга
        ims = [load(paths['zeros'])[sl], load(paths['ones'])[sl], load(paths['combined'])[sl], hr[sl]]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
        for ax, im, t in zip(axes, ims, ['bicubic', 'DiffBIR, маска = 1', 'комбинированная (граница круга)', 'HR']):
            ax.imshow(im)
            ax.set_title(t, fontsize=10)
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'zoom_{deg}_{img_id}.jpg'), dpi=120)
        plt.close(fig)


def plot_timing(args, out_dir):
    p = os.path.join(args.results, 'timing.csv')
    if not os.path.exists(p):
        return
    t = pd.read_csv(p)
    print('\nВремя, с (среднее) и пик памяти GPU, ГБ:')
    print(t.groupby('mask')[['seconds', 'max_mem_gb']].mean().round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'data'))
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--plots_only', action='store_true', help='не считать метрики, взять metrics.csv')
    args = ap.parse_args()

    fig_dir = os.path.join(args.results, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(args.results, 'metrics.csv')) if args.plots_only else compute(args)
    plot_overall(df, fig_dir)
    plot_regions(df, fig_dir)
    summary_table(df, os.path.join(args.results, 'summary.md'))
    comparison_grids(args, fig_dir)
    plot_timing(args, fig_dir)


if __name__ == '__main__':
    main()
