import argparse
import io
import os
import urllib.request
import zipfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DIV2K_URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/'
ARCHIVES = {
    'hr': ('DIV2K_valid_HR.zip', 'DIV2K_valid_HR/{id}.png'),
    'bicubic': ('DIV2K_valid_LR_bicubic_X4.zip', 'DIV2K_valid_LR_bicubic/X4/{id}x4.png'),
    'unknown': ('DIV2K_valid_LR_unknown_X4.zip', 'DIV2K_valid_LR_unknown/X4/{id}x4.png'),
}
IMAGE_IDS = ['0801', '0804', '0810', '0829', '0855']
HR_SIZE = 1024
SCALE = 4
LQ_SIZE = HR_SIZE // SCALE


def download(name, cache_dir):
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        print(f'download {name}')
        urllib.request.urlretrieve(DIV2K_URL + name, path + '.part')
        os.replace(path + '.part', path)
    return path


def read_from_zip(zf, member):
    data = np.frombuffer(zf.read(member), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def center_crop_box(h, w, size):
    y = (h - size) // 2 // SCALE * SCALE
    x = (w - size) // 2 // SCALE * SCALE
    return y, x


def random_blur(img, rng):
    k = int(rng.choice([7, 9, 11, 13, 15, 17, 19, 21]))
    sx, sy = rng.uniform(0.2, 3.0, size=2)
    theta = rng.uniform(-np.pi, np.pi)
    ax = np.arange(k) - k // 2
    xx, yy = np.meshgrid(ax, ax)
    c, s = np.cos(theta), np.sin(theta)
    xr, yr = c * xx + s * yy, -s * xx + c * yy
    kernel = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def random_noise(img, rng, sigma_range, poisson_range):
    if rng.random() < 0.6:
        sigma = rng.uniform(*sigma_range) / 255.0
        if rng.random() < 0.4:  # «серый» шум, одинаковый по каналам
            noise = rng.normal(0, sigma, img.shape[:2])[..., None]
        else:
            noise = rng.normal(0, sigma, img.shape)
        return img + noise
    scale = rng.uniform(*poisson_range)
    vals = 10 ** 2.5 / scale
    return rng.poisson(np.clip(img, 0, 1) * vals) / vals


def jpeg(img, rng, q_range):
    q = int(rng.uniform(*q_range))
    u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    _, buf = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) / 255.0


def realesrgan_degrade(hr, seed):
    rng = np.random.default_rng(seed)
    img = hr.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    interps = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

    # порядок 1
    img = random_blur(img, rng)
    s = rng.uniform(0.15, 1.5)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=int(rng.choice(interps)))
    img = random_noise(img, rng, (1, 30), (0.05, 3))
    img = jpeg(img, rng, (30, 95))
    # порядок 2
    if rng.random() < 0.8:
        img = random_blur(img, rng)
    s = rng.uniform(0.3, 1.2)
    img = cv2.resize(img, (int(w / SCALE * s), int(h / SCALE * s)), interpolation=int(rng.choice(interps)))
    img = random_noise(img, rng, (1, 25), (0.05, 2.5))
    # финальный resize к целевому размеру + JPEG
    img = cv2.resize(img, (w // SCALE, h // SCALE), interpolation=int(rng.choice(interps)))
    img = jpeg(img, rng, (30, 95))
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)



def make_masks(size):
    ones = np.ones((size, size), np.uint8)
    zeros = np.zeros((size, size), np.uint8)
    yy, xx = np.mgrid[:size, :size]
    r = size * np.sqrt(0.5 / np.pi)  # круг площадью 50% кадра
    combined = (((yy - size / 2) ** 2 + (xx - size / 2) ** 2) <= r ** 2).astype(np.uint8)
    return {'ones': ones, 'zeros': zeros, 'combined': combined}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--cache', default=os.path.join(ROOT, 'data', 'div2k_zip'))
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    zips = {k: zipfile.ZipFile(download(name, args.cache)) for k, (name, _) in ARCHIVES.items()}

    for d in ['hr', 'lq/bicubic', 'lq/unknown', 'lq/realesrgan', 'masks']:
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    for n, img_id in enumerate(IMAGE_IDS):
        hr = read_from_zip(zips['hr'], ARCHIVES['hr'][1].format(id=img_id))
        y, x = center_crop_box(*hr.shape[:2], HR_SIZE)
        hr_crop = hr[y:y + HR_SIZE, x:x + HR_SIZE]
        cv2.imwrite(os.path.join(args.out, 'hr', f'{img_id}.png'), hr_crop)

        for deg in ['bicubic', 'unknown']:
            lr = read_from_zip(zips[deg], ARCHIVES[deg][1].format(id=img_id))
            ly, lx = y // SCALE, x // SCALE
            lr_crop = lr[ly:ly + LQ_SIZE, lx:lx + LQ_SIZE]
            assert lr_crop.shape[:2] == (LQ_SIZE, LQ_SIZE), lr.shape
            cv2.imwrite(os.path.join(args.out, 'lq', deg, f'{img_id}.png'), lr_crop)

        lq = realesrgan_degrade(hr_crop, seed=1000 + n)
        cv2.imwrite(os.path.join(args.out, 'lq', 'realesrgan', f'{img_id}.png'), lq)
        print(f'{img_id}: crop at y={y}, x={x}')

    for name, m in make_masks(LQ_SIZE).items():
        cv2.imwrite(os.path.join(args.out, 'masks', f'{name}.png'), m * 255)
    print('done ->', args.out)


if __name__ == '__main__':
    main()
