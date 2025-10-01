#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
270 사이즈(기본)의 컨트롤 포인트 → B-spline 곡선과
이미지(윤곽선) 간의 거리(mm)를 계산하고 요약 통계를 출력합니다.

입력:
- IMAGE_PATH: 270 인솔 윤곽선 이미지(흰 배경+검은 선)
- CSV_PATH  : 컨트롤 포인트 CSV (행: size_mm, x1, y1, x2, y2, ...)

출력:
- distances_<size>.csv  : B-spline 각 샘플점의 최근접 윤곽선까지 거리(mm)
- overlay_<size>.png    : mm로 맞춘 윤곽선 vs 스플라인 오버레이
- 터미널에 요약 통계(평균/중앙/90,95퍼/최대, trimmed 최대)

※ 스케일(픽셀→mm)은 이미지 윤곽선의 세로 길이를
   컨트롤 포인트로 만든 스플라인의 세로 길이에 맞춰 자동 산출합니다.
"""

import os, math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ===== 사용자 설정 =====
IMAGE_PATH   = "250L.jpg"            # 270 윤곽선 이미지 경로
CSV_PATH     = "pred_series_230_250_270.csv" # 컨트롤 포인트 CSV 경로
TARGET_SIZE  = 250                  # 사용할 사이즈(mm) !!! 반드시 해줄것...
THRESHOLD    = 40                     # 윤곽선 이진화 임계(0~255, 작을수록 검게 필터)
SPL_SAMPLES  = 1500                   # 스플라인 샘플 개수
TRIM_ENDS    = 10                     # 오픈 스플라인 끝단 트림 샘플 수(최대치 튀는 것 완화)
SHOW_PLOT    = True                   # 오버레이 창 표시 여부
# ======================


# ---------- CSV 파싱 ----------
def parse_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = [t for t in s.replace(",", " ").split() if t]
            try:
                vals = list(map(float, toks))
            except Exception:
                continue
            if len(vals) < 3:
                continue
            size = int(round(vals[0]))
            xy = np.array(vals[1:], dtype=float)
            if xy.size % 2 == 1:
                xy = xy[:-1]
            pts = xy.reshape(-1, 2)
            rows.append((size, pts))
    if not rows:
        raise RuntimeError(f"No valid rows in CSV: {path}")
    return rows

def pick_ctrl_for_size(rows, target):
    # 같은 사이즈가 여러 번이면 "마지막" 행이 최신이라 가정
    candidates = [(i,s,p) for i,(s,p) in enumerate(rows) if s == target]
    if candidates:
        _, s, p = candidates[-1]
        return s, p
    # 없으면 가장 가까운 사이즈를 사용
    i_best = min(range(len(rows)), key=lambda i: abs(rows[i][0]-target))
    return rows[i_best][0], rows[i_best][1].copy()

# ---------- B-spline ----------
def open_uniform_knot_vector(n_ctrl, degree):
    kv = np.concatenate([
        np.zeros(degree+1),
        np.arange(1, n_ctrl - degree),
        np.full(degree+1, n_ctrl - degree),
    ])
    return kv / kv[-1]

def bspline_basis(i, k, knots, t):
    t = np.asarray(t)
    if k == 0:
        last = (i+1 == len(knots)-1)
        return np.where(
            (knots[i] <= t) & ((t < knots[i+1]) | (last & np.isclose(t, knots[i+1]))),
            1.0, 0.0
        )
    left_den  = knots[i+k]   - knots[i]
    right_den = knots[i+k+1] - knots[i+1]
    left  = 0.0 if left_den  <= 0 else ((t - knots[i]) / left_den) * bspline_basis(i, k-1, knots, t)
    right = 0.0 if right_den <= 0 else ((knots[i+k+1] - t) / right_den) * bspline_basis(i+1, k-1, knots, t)
    return left + right

def bspline_curve(ctrl, degree=3, samples=1000):
    n = len(ctrl)
    knots = open_uniform_knot_vector(n, degree)
    t = np.linspace(0, 1, samples, endpoint=True)
    basis = np.stack([bspline_basis(i, degree, knots, t) for i in range(n)], axis=1)
    xy = basis @ ctrl
    return xy

# ---------- 윤곽선 추출 ----------
def extract_outline_points_mm(image_path, mm_per_px=None, threshold=50):
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    mask = arr < threshold  # 어두운 픽셀만
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("No outline pixels found. Try raising THRESHOLD.")
    # 좌상단 기준으로 원점 이동
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    h_px = maxy - miny
    w_px = maxx - minx
    if mm_per_px is None:
        mm_per_px = None  # 스케일은 바깥에서 결정
    out = {
        "px_points": np.column_stack([xs, ys]),
        "bbox": (minx, miny, maxx, maxy),
        "h_px": h_px, "w_px": w_px,
        "arr_shape": arr.shape,
        "img": img
    }
    return out

# ---------- 거리 계산 ----------
def nearest_distances(A, B, chunk=1500):
    n = A.shape[0]
    out = np.empty(n, dtype=float)
    s = 0
    while s < n:
        e = min(s + chunk, n)
        diff = A[s:e, None, :] - B[None, :, :]
        d2 = np.sum(diff*diff, axis=2)
        out[s:e] = np.sqrt(d2.min(axis=1))
        s = e
    return out

def summarize(d):
    return {
        "mean_mm": float(d.mean()),
        "median_mm": float(np.median(d)),
        "p90_mm": float(np.percentile(d, 90)),
        "p95_mm": float(np.percentile(d, 95)),
        "max_mm": float(d.max()),
    }

# ---------- 메인 ----------
def main():
    rows = parse_rows(CSV_PATH)
    size, ctrl = pick_ctrl_for_size(rows, TARGET_SIZE)

    # 스플라인(mm)
    curve_mm = bspline_curve(ctrl, degree=3, samples=SPL_SAMPLES)
    # 스플라인 세로 길이(mm) → 이후 이미지 스케일 결정에 사용
    curve_h_mm = curve_mm[:,1].max() - curve_mm[:,1].min()

    # 이미지 윤곽선
    outline = extract_outline_points_mm(IMAGE_PATH, threshold=THRESHOLD)
    minx, miny, maxx, maxy = outline["bbox"]
    h_px = outline["h_px"]

    # 픽셀→mm 스케일: 이미지 윤곽선 세로 길이 == 스플라인 세로 길이
    mm_per_px = curve_h_mm / float(h_px)

    # 좌상단 원점으로 정렬 & mm 변환
    xs, ys = outline["px_points"][:,0], outline["px_points"][:,1]
    outline_mm = np.column_stack([(xs - minx)*mm_per_px, (ys - miny)*mm_per_px])

    curve_mm_shift = curve_mm - curve_mm.min(axis=0)
    
    # 거리: 스플라인 각 샘플 → 최근접 윤곽선
    d_curve_to_outline = nearest_distances(curve_mm_shift, outline_mm, chunk=2000)

    # 끝단 트림(오픈 스플라인 끝에서 튈 수 있음)
    if TRIM_ENDS > 0 and TRIM_ENDS*2 < len(d_curve_to_outline):
        d_trim = d_curve_to_outline[TRIM_ENDS:-TRIM_ENDS]
    else:
        d_trim = d_curve_to_outline

    # 요약
    stat_full = summarize(d_curve_to_outline)
    stat_trim = summarize(d_trim)

    print(f"[SIZE] {size} mm  (rows in CSV may differ; using nearest if 1:1 not found)")
    print("[Curve→Outline distance, mm]")
    for k,v in stat_full.items():
        print(f"  {k:>9}: {v:6.3f}")
    print("[Trimmed ends] (exclude first/last %d samples)" % TRIM_ENDS)
    for k,v in stat_trim.items():
        print(f"  {k:>9}: {v:6.3f}")

    # 저장물
    base = f"{int(size)}"
    out_csv = f"distances_{base}.csv"
    np.savetxt(out_csv,
               np.column_stack([np.arange(len(d_curve_to_outline)), d_curve_to_outline]),
               fmt=["%d","%.6f"], delimiter=",",
               header="sample_index,distance_mm", comments="")
    print(f"[Saved] {out_csv}")

    # 오버레이
    plt.figure(figsize=(5,9))
    plt.scatter(outline_mm[:,0], outline_mm[:,1], s=1, label="Outline (px→mm)",color='red')

    
    plt.plot(curve_mm_shift[:,0], curve_mm_shift[:,1], lw=2, label="B-spline")
    
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", "box")
    plt.title(f"Outline vs B-spline (size={size}mm)")
    plt.legend()
    out_png = f"overlay_{base}.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()
    print(f"[Saved] {out_png}")

if __name__ == "__main__":
    main()
