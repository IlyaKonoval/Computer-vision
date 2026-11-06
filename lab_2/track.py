import argparse
import time
from pathlib import Path

import cv2
import numpy as np

MIN_INLIERS = 10
RATIO = 0.8
RANSAC_THRESH = 4.0
REF_SCALES = (1.0, 0.5, 0.25)
MIN_TRACKED = 8
FB_ERR = 1.5
REDETECT_EVERY = 15
SMOOTH = 0.7
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def make_detector(name):
    if name == "orb":
        return cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING
    return cv2.SIFT_create(), cv2.NORM_L2


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def parse_box(text, shape):
    h, w = shape[:2]
    if text is None:
        return np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    x0, y0, x1, y1 = (float(v) for v in text.split(","))
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def build_reference(det, ref):
    pts, descs = [], []
    for s in REF_SCALES:
        img = ref if s == 1.0 else cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        kp, des = det.detectAndCompute(preprocess(img), None)
        if des is None:
            continue
        pts += [(k.pt[0] / s, k.pt[1] / s) for k in kp]
        descs.append(des)
    return np.float32(pts), np.vstack(descs)


def is_valid_quad(quad, frame_shape):
    pts = quad.reshape(-1, 2).astype(np.float32)
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
        return False
    area = cv2.contourArea(pts)
    h, w = frame_shape[:2]
    return 0.001 * w * h < area < 4 * w * h


def fit_homography(src, dst, min_inliers):
    if len(src) < min_inliers:
        return None
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    if H is None or int(mask.sum()) < min_inliers:
        return None
    m = mask.ravel().astype(bool)
    return H, src[m], dst[m]


def locate(pts_ref, des_ref, kp, des, matcher):
    if des is None or len(kp) < 2:
        return None
    pairs = matcher.knnMatch(des_ref, des, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = pts_ref[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return fit_homography(src, dst, MIN_INLIERS)


def flow_track(prev_gray, gray, ref_pts, cur_pts):
    if len(cur_pts) < MIN_TRACKED:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, cur_pts, None, **LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **LK)
    err = np.linalg.norm(cur_pts - back, axis=2).ravel()
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (err < FB_ERR)
    return fit_homography(ref_pts[ok], nxt[ok], MIN_TRACKED)


def is_plausible(quad, prev_quad):
    a, b = cv2.contourArea(quad.reshape(-1, 2)), cv2.contourArea(prev_quad.reshape(-1, 2))
    if not 0.5 < a / max(b, 1e-6) < 2.0:
        return False
    shift = np.linalg.norm(quad.reshape(-1, 2) - prev_quad.reshape(-1, 2), axis=1).mean()
    return shift < 0.6 * np.sqrt(b)


def draw(frame, quad, label, inliers):
    pts = quad.reshape(-1, 2)
    cv2.polylines(frame, [pts.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    x, y = pts.min(axis=0).astype(int)
    cv2.putText(frame, f"{label} ({inliers})", (max(x, 5), max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def track(video, out_dir, detector, label, box, use_flow=True):
    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"не удалось открыть {video}")
    det, norm = make_detector(detector)
    matcher = cv2.BFMatcher(norm)
    corners = parse_box(box, first.shape)
    x0, y0 = corners.min(axis=0).astype(int)
    x1, y1 = corners.max(axis=0).astype(int)
    pts_ref, des_ref = build_reference(det, first[y0:y1, x0:x1])
    corners -= (x0, y0)  # углы в координатах вырезанного эталона

    h, w = first.shape[:2]
    name = Path(video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / f"{name}_{detector}{'' if use_flow else '_noflow'}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
    shots = {}
    found = n = inl_sum = 0
    t0 = time.perf_counter()
    frame = first
    prev_gray = None
    ref_pts = cur_pts = None   # пары (точка эталона, её положение на кадре) для потока
    prev_quad, last_ok, shown, tracked = None, -10**9, None, 0
    while True:
        n += 1
        gray = preprocess(frame)
        quad, inl, via_flow = None, 0, False
        if use_flow and cur_pts is not None and prev_gray is not None:
            res = flow_track(prev_gray, gray, ref_pts, cur_pts)
            if res is not None:
                q = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), res[0])
                if is_valid_quad(q, frame.shape) and is_plausible(q, prev_quad):
                    quad, inl, via_flow = q, len(res[2]), True
                    ref_pts, cur_pts = res[1], res[2]
        # SIFT: перезахват при потере и периодическая коррекция дрейфа
        if quad is None or n % REDETECT_EVERY == 0:
            kp, des = det.detectAndCompute(gray, None)
            res = locate(pts_ref, des_ref, kp, des, matcher)
            if res is not None:
                q = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), res[0])
                near = n - last_ok <= 5
                if is_valid_quad(q, frame.shape) and (not near or is_plausible(q, prev_quad)):
                    quad, inl, via_flow = q, len(res[2]), False
                    ref_pts, cur_pts = res[1], res[2]
        prev_gray = gray
        if quad is None:
            ref_pts = cur_pts = None
        if quad is not None:
            found += 1
            tracked += via_flow
            inl_sum += inl
            fresh = shown is None or n - last_ok > 5
            last_ok, prev_quad = n, quad
            shown = quad if fresh else SMOOTH * quad + (1 - SMOOTH) * shown
            draw(frame, shown, label, inl)
        else:
            cv2.putText(frame, "not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        if n % 60 == 1:
            shots[n] = frame.copy()
        ok, frame = cap.read()
        if not ok:
            break
    dt = time.perf_counter() - t0
    writer.release()
    for i, img in shots.items():
        cv2.imwrite(str(out_dir / f"{name}_{detector}{'' if use_flow else '_noflow'}_f{i:04d}.jpg"), img)
    stats = dict(video=name, detector=detector, frames=n, found=found,
                 pct=100 * found / n, mean_inliers=inl_sum / max(found, 1), fps=n / dt,
                 ref_kp=len(pts_ref), tracked=tracked)
    print("{video:28s} {detector:5s} kp={ref_kp:5d} frames={frames:4d} found={found:4d} "
          "({pct:5.1f}%) flow={tracked:4d} inliers={mean_inliers:6.1f} fps={fps:5.1f}".format(**stats))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="+")
    ap.add_argument("--detector", choices=["sift", "orb"], default="sift")
    ap.add_argument("--out", default="results")
    ap.add_argument("--label", default="object")
    ap.add_argument("--box", help="рамка объекта на первом кадре x0,y0,x1,y1 в пикселях (по умолчанию весь кадр)")
    ap.add_argument("--flow", action=argparse.BooleanOptionalAction, default=True,
                    help="оптический поток между кадрами (--no-flow: только детектор)")
    a = ap.parse_args()
    for v in a.video:
        track(v, Path(a.out), a.detector, a.label, a.box, a.flow)


if __name__ == "__main__":
    main()
