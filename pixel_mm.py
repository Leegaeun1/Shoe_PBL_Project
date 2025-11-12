#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
컨트롤 포인트로 생성한 B-spline 곡선과
이미지(윤곽선) 간의 거리(mm)를 계산하고 요약 통계를 출력합니다.

※ 스케일(픽셀→mm)은 이미지 파일 이름에서 실제 사이즈를 읽어와 자동 산출합니다.
"""

'''윤곽선 사진이랑 컨트롤포인트 하면 비교해줌'''

import os
import re  #파일 이름 파싱을 위해 re 모듈을 가져옵니다.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ===== 사용자 설정 =====
IMAGE_PATH = "240L.jpg"          # 분석할 윤곽선 이미지 경로
CSV_PATH = "pred_series_krr.csv"     # 컨트롤 포인트 CSV 경로
TARGET_SIZE = 240                # CSV에서 사용할 사이즈(mm)
THRESHOLD = 40                   # 윤곽선 이진화 임계(0~255, 작을수록 검게 필터)
SPL_SAMPLES = 1500               # 스플라인 샘플 개수
TRIM_ENDS = 10                   # 오픈 스플라인 끝단 트림 샘플 수(최대치 튀는 것 완화)
SHOW_PLOT = True                 # 오버레이 창 표시 여부
# ======================


# ---------- CSV 파싱 ----------
def parse_rows(path): # CSV 파일 읽음
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

def pick_ctrl_for_size(rows, target):  # TARGET_SIZE에 해당하는 사이즈의 제어점 데이터 선택
    candidates = [(i,s,p) for i,(s,p) in enumerate(rows) if s == target]
    if candidates:
        _, s, p = candidates[-1]
        return s, p
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
    left_den = knots[i+k] - knots[i]
    right_den = knots[i+k+1] - knots[i+1]
    left = 0.0 if left_den <= 0 else ((t - knots[i]) / left_den) * bspline_basis(i, k-1, knots, t)
    right = 0.0 if right_den <= 0 else ((knots[i+k+1] - t) / right_den) * bspline_basis(i+1, k-1, knots, t)
    return left + right

def bspline_curve(ctrl, degree=3, samples=1000): # 제어점들을 이용해 부드러운 곡선 생성
    n = len(ctrl)
    knots = open_uniform_knot_vector(n, degree)
    t = np.linspace(0, 1, samples, endpoint=True)
    basis = np.stack([bspline_basis(i, degree, knots, t) for i in range(n)], axis=1)
    xy = basis @ ctrl
    return xy

# ---------- 윤곽선 추출 ----------
def extract_outline_points(image_path, threshold=50):
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    mask = arr < threshold  # THRESHOLD값보다 어두운 모든 픽셀들 찾아내 그 위치를 픽셀 좌표로 저장함.
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("No outline pixels found. Try raising THRESHOLD.")
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    h_px = maxy - miny # 전체 길이가 몇픽셀인가
    out = {
        "px_points": np.column_stack([xs, ys]),
        "bbox": (minx, miny, maxx, maxy),
        "h_px": h_px,
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
        "mean_mm": float(d.mean()),# 거리값들의 평균
        "median_mm": float(np.median(d)), # 중앙값
        "p90_mm": float(np.percentile(d, 90)),  # 상위 90%
        "p95_mm": float(np.percentile(d, 95)), # 상위 95%
        "max_mm": float(d.max()), # 최대값
    }

# 파일 이름에서 사이즈(mm)를 읽어오는 함수
def parse_size_mm_from_filename(path, fallback=250.0):
    """ 파일 경로에서 2~3자리 숫자를 찾아 사이즈(mm)로 반환합니다. """
    m = re.search(r'(\d{2,3})', os.path.basename(path))
    return float(m.group(1)) if m else float(fallback)

# ---------- 메인 ----------
def main():
    rows = parse_rows(CSV_PATH)
    size, ctrl = pick_ctrl_for_size(rows, TARGET_SIZE)

    # ****************** [핵심 수정: 제어점 복사로 곡선 닫기] ******************
    degree = 3 # bspline_curve의 기본 degree (degree=3 사용 가정)
    
    # 닫힌 곡선을 만들기 위해 시작점 3개를 끝에 복사하여 추가
    ctrl_closed = np.concatenate([ctrl, ctrl[:degree]], axis=0) 
    
    # 스플라인(mm) 생성
    # curve_mm = bspline_curve(ctrl, degree=3, samples=SPL_SAMPLES)
    curve_mm = bspline_curve(ctrl_closed, degree=degree, samples=SPL_SAMPLES)
    # ********************************************************************************
    
    # 이미지에서 윤곽선 픽셀 추출
    outline = extract_outline_points(IMAGE_PATH, threshold=THRESHOLD)
    minx, miny, _, _ = outline["bbox"]
    h_px = outline["h_px"]

    # 이미지 파일 이름에서 실제 사이즈를 읽어와 스케일을 자동 계산
    image_size_mm = parse_size_mm_from_filename(IMAGE_PATH, fallback=float(TARGET_SIZE))
    mm_per_px = image_size_mm / float(h_px) # 실제 크기/픽셀 높이 -> 1픽셀 당 몇 MM인가
    print(f"[INFO] Scale calculated for '{os.path.basename(IMAGE_PATH)}':")
    print(f"       - Real size from filename: {image_size_mm:.1f} mm")
    print(f"       - Outline height in pixels: {h_px} px")
    print(f"       - Resulting scale: {mm_per_px:.4f} mm/px")
    # -------------------------------------------------------------------

    # 윤곽선 픽셀 좌표를 mm 좌표로 변환 및 원점 정렬
    xs, ys = outline["px_points"][:,0], outline["px_points"][:,1]
    outline_mm = np.column_stack([(xs - minx) * mm_per_px, (ys - miny) * mm_per_px])

    # 스플라인 곡선 원점 정렬
    curve_mm_shift = curve_mm - curve_mm.min(axis=0)
    
    # 거리 계산: 스플라인 각 샘플 → 최근접 윤곽선
    d_curve_to_outline = nearest_distances(curve_mm_shift, outline_mm, chunk=2000)

    # 끝단 트림 (TRIM_ENDS가 적용됨)
    d_trim = d_curve_to_outline
    if TRIM_ENDS > 0 and TRIM_ENDS*2 < len(d_curve_to_outline):
        d_trim = d_curve_to_outline[TRIM_ENDS:-TRIM_ENDS]

    # 요약 통계 출력
    stat_full = summarize(d_curve_to_outline)
    stat_trim = summarize(d_trim)
    print(f"\n[SIZE] Comparing {size}mm spline from CSV to {image_size_mm:.0f}mm image")
    print("[Curve→Outline distance, mm]")
    for k,v in stat_full.items():
        print(f"   {k:>9}: {v:6.3f}")
    print("\n[Trimmed ends] (excluding first/last %d samples)" % TRIM_ENDS)
    for k,v in stat_trim.items():
        print(f"   {k:>9}: {v:6.3f}")

    # 거리 데이터 CSV로 저장
    base = f"{int(size)}"
    out_csv = f"distances_{base}.csv"
    np.savetxt(out_csv,
               np.column_stack([np.arange(len(d_curve_to_outline)), d_curve_to_outline]),
               fmt=["%d","%.6f"], delimiter=",",
               header="sample_index,distance_mm", comments="")
    print(f"\n[Saved] {out_csv}")

    # 오버레이 이미지 생성 및 저장
    plt.figure(figsize=(5,9))
    plt.scatter(outline_mm[:,0], outline_mm[:,1], s=1, label="Outline (from Image)", color='red', alpha=0.8)
    # ****************** [핵심 수정: 시각화 시 B-spline 끝단 트림 적용] ******************
    curve_to_plot = curve_mm_shift
    if TRIM_ENDS > 0 and TRIM_ENDS*2 < len(curve_mm_shift):
        # TRIM_ENDS만큼 앞뒤 샘플을 제외하고 그립니다. (불안정한 시작/끝점 제거)
        curve_to_plot = curve_mm_shift[TRIM_ENDS:-TRIM_ENDS]

    plt.plot(curve_to_plot[:,0], curve_to_plot[:,1], lw=2, label="B-spline (from CSV)", color='blue')
    # ********************************************************************************
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", "box")
    plt.title(f"Outline vs B-spline ({size}mm)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_png = f"overlay_{base}.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"[Saved] {out_png}")
    
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()