
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
IMAGES = BASE / "images"
RESULTS = BASE / "results"


def to_uint8(g):
    m = g.max()
    if m == 0:
        return np.zeros(g.shape, np.uint8)
    return (g * (255.0 / m)).astype(np.uint8)


def sobel_opencv(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return to_uint8(cv2.magnitude(gx, gy))


def sobel_native(gray):
    h, w = gray.shape
    p = np.pad(gray, 1, mode="reflect").tolist()
    g = [[0.0] * w for _ in range(h)]
    for y in range(h):
        r0, r1, r2 = p[y], p[y + 1], p[y + 2]
        out = g[y]
        for x in range(w):
            gx = (r0[x + 2] - r0[x]) + 2 * (r1[x + 2] - r1[x]) + (r2[x + 2] - r2[x])
            gy = (r2[x] - r0[x]) + 2 * (r2[x + 1] - r0[x + 1]) + (r2[x + 2] - r0[x + 2])
            out[x] = (gx * gx + gy * gy) ** 0.5
    return to_uint8(np.array(g))


def synthetic(n=512):
    img = np.zeros((n, n), np.uint8)
    cv2.rectangle(img, (n // 8, n // 8), (n // 2, n // 2), 255, -1)
    cv2.circle(img, (int(n * 0.7), int(n * 0.65)), n // 5, 180, -1)
    cv2.line(img, (0, n - 1), (n - 1, n // 3), 120, 3)
    return img


def load_defaults():
    from skimage import data
    IMAGES.mkdir(exist_ok=True)
    imgs = {
        "camera": data.camera(),
        "astronaut": cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2GRAY),
        "synthetic": synthetic(),
    }
    for name, im in imgs.items():
        cv2.imwrite(str(IMAGES / f"{name}.png"), im)
    return imgs


def timeit(f, img, repeats):
    best = []
    for _ in range(repeats):
        t = time.perf_counter()
        f(img)
        best.append(time.perf_counter() - t)
    return sum(best) / len(best)


def main():
    RESULTS.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        imgs = {Path(a).stem: cv2.imread(a, cv2.IMREAD_GRAYSCALE) for a in sys.argv[1:]}
    else:
        imgs = load_defaults()

    # 1. Результаты и проверка совпадения
    for name, g in imgs.items():
        a, b = sobel_opencv(g), sobel_native(g)
        print(f"{name}: max|opencv-native|={np.abs(a.astype(int) - b).max()}")
        cv2.imwrite(str(RESULTS / f"{name}_opencv.png"), a)
        cv2.imwrite(str(RESULTS / f"{name}_native.png"), b)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
        for axis, im, t in zip(ax, (g, a, b), ("Оригинал", "OpenCV", "Нативный")):
            axis.imshow(im, cmap="gray")
            axis.set_title(t)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(RESULTS / f"{name}_compare.png", dpi=100)
        plt.close(fig)

    # 2. Быстродействие на разных размерах
    base = next(iter(imgs.values()))
    sizes = [128, 256, 512, 1024]
    t_cv, t_nat = [], []
    print(f"\n{'размер':>8} {'OpenCV, мс':>12} {'нативный, мс':>14} {'нат/cv':>9}")
    for s in sizes:
        im = cv2.resize(base, (s, s))
        cv_t = timeit(sobel_opencv, im, 10)
        nat_t = timeit(sobel_native, im, 1 if s >= 512 else 3)
        t_cv.append(cv_t); t_nat.append(nat_t)
        print(f"{s:>8} {cv_t * 1e3:>12.3f} {nat_t * 1e3:>14.1f} {nat_t / cv_t:>9.0f}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, [t * 1e3 for t in t_cv], "o-", label="OpenCV")
    plt.plot(sizes, [t * 1e3 for t in t_nat], "^-", label="Нативный (циклы)")
    plt.yscale("log")
    plt.xlabel("Сторона изображения, px")
    plt.ylabel("Время, мс (лог. шкала)")
    plt.title("Время работы фильтра Собеля")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "timing.png", dpi=110)


if __name__ == "__main__":
    main()
