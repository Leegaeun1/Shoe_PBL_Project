#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
두 개의 좌표 CSV(예: 정답 vs 예측)를 사이즈별로 비교합니다.

허용 입력 형식
 - "size, x1,y1,..."
 - "type,side,size,x1,y1,..."
 - 헤더 유무 상관없음(비수치 행은 자동 스킵)

출력(타임스탬프 폴더 내부)
 - compare_<SIZE>.png : 좌표 오버레이 시각화
 - compare_summary_all_sizes.csv : 모든 사이즈 요약(단일 파일)

옵션으로 per-size 통계/거리 시퀀스 CSV를 저장할 수 있음(기본 False).
"""

import os, csv, re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import MultipleLocator


# ========= 사용자 설정 =========

Data_DIR = "control_points_master_test_Q" 

# 정답 파일 경로
FILE_A =os.path.join(
    Data_DIR,
    "control_points_master_test_Q.csv"
)
# 예측 파일 경로
FILE_B =os.path.join(
    Data_DIR,
    "pred_Data_230_280_ge_SVR.csv"
)

TARGET_SIZES = None  # None이면 공통 사이즈 전부, 예: [230,240]
USE_BSPLINE = True  # 제어점 듬성듬성이면 True
DEGREE = 3
SAMPLES = 1500
CLOSED = False        # 닫힌 윤곽(순환 정렬)
# 저장물 통제
SAVE_PLOTS = True
SHOW_PLOTS = False
SAVE_PER_SIZE_STATS = False     # ← 개별 compare_stats_<size>.csv 저장 여부
SAVE_DISTANCE_SERIES = False    # ← distances_* 저장 여부
OUT_ROOT = "."                  # 타임스탬프 폴더 상위
# =============================

# ---------- 공용 유틸 ----------
_NUM = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][\+\-]?\d+)?$')
def _is_num(x: str) -> bool:
    return bool(_NUM.match(x))

def chordlen_resample(P, n):
    P = np.asarray(P, float)
    if len(P) <= 1:
        return np.repeat(P[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.zeros(len(P)); u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-12:
        return np.repeat(P[:1], n, axis=0)
    u /= L
    s = np.linspace(0,1,n,endpoint=True)
    x = np.interp(s, u, P[:,0]); y = np.interp(s, u, P[:,1])
    return np.stack([x,y], axis=1)

def _align_score(P, Q):
    return float(np.sum((P-Q)**2))

def cyclic_align(P, Q):
    """P와 Q(동일 길이)를 순환 시프트/역방향 포함 최적 정렬."""
    n = len(P)
    best = (None, 1e30, 0, False)
    for rev in [False, True]:
        R = Q[::-1].copy() if rev else Q.copy()
        for k in range(n):
            Rk = np.roll(R, -k, axis=0)
            sc = _align_score(P, Rk)
            if sc < best[1]:
                best = (Rk, sc, k, rev)
    return best[0]

def nearest_distances(A, B, chunk=3000):
    """A 각 점에서 B까지의 최단거리(유클리드)"""
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

def summarize_dist(d_ab, d_ba):
    """양방향 요약 + Chamfer/Hausdorff"""
    d_ab = np.asarray(d_ab); d_ba = np.asarray(d_ba)
    res = {
        "A_to_B_mean": float(d_ab.mean()),
        "A_to_B_median": float(np.median(d_ab)),
        "A_to_B_p90": float(np.percentile(d_ab, 90)),
        "A_to_B_p95": float(np.percentile(d_ab, 95)),
        "A_to_B_max": float(d_ab.max()),
        "B_to_A_mean": float(d_ba.mean()),
        "B_to_A_median": float(np.median(d_ba)),
        "B_to_A_p90": float(np.percentile(d_ba, 90)),
        "B_to_A_p95": float(np.percentile(d_ba, 95)),
        "B_to_A_max": float(d_ba.max()),
    }
    res["Chamfer_mean"] = 0.5*(res["A_to_B_mean"] + res["B_to_A_mean"])
    res["Hausdorff_max"] = float(max(res["A_to_B_max"], res["B_to_A_max"]))
    return res


def pca_major_axis(P):
    """주축 v1(길이 방향)과 그에 수직인 v2를 반환, 뒤꿈치 index = v1 투영 최소값."""
    C = P - P.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    v1 = Vt[0]                     # 길이(Toe–Heel) 방향
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = np.array([-v1[1], v1[0]]) # v1에 수직(우수좌표계)
    z1 = P @ v1
    heel_idx = int(np.argmin(z1))
    return v1, v2, heel_idx

def to_heel_up_frame(P, v1=None, v2=None, y0_shift=None):
    """
    v1을 +y, v2를 +x로 두는 회전 적용 후, y의 최소값(뒤꿈치)을 0으로 이동.
    y0_shift를 넘기면 동일 기준으로 정렬(Ref와 Pred 일관성 보장).
    """
    if v1 is None or v2 is None:
        v1, v2, _ = pca_major_axis(P)
    R = np.stack([v2, v1], axis=1)   # (2x2), x'=v2, y'=v1
    Pp = P @ R                       # 회전
    if y0_shift is None:
        y0_shift = Pp[:,1].min()     # 뒤꿈치 y=0
    Pp[:,1] -= y0_shift              # 위로 이동
    return Pp, R, y0_shift



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

def bspline_curve(ctrl, degree=3, samples=1000, closed=False):
    # 형상을 부드럽게 만든다
    ctrl = np.asarray(ctrl, float)
    if closed:
        ctrl = np.concatenate([ctrl, ctrl[:degree]], axis=0)
    n = len(ctrl)
    knots = open_uniform_knot_vector(n, degree)
    t = np.linspace(0, 1, samples, endpoint=True)
    basis = np.stack([bspline_basis(i, degree, knots, t) for i in range(n)], axis=1)
    xy = basis @ ctrl
    return xy

# ---------- CSV 파서 ----------
def load_rows_any(path):
    """
    반환: dict[size:int] = np.ndarray[(L,2)]
    규칙:
      - 한 행에서 '첫 번째 수치'를 size로 간주
      - 그 뒤의 모든 수치를 좌표로 사용(홀수면 마지막 하나 버림)
      - 비수치 행/헤더는 자동 스킵
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row: 
                continue
            toks = [t.strip() for t in row if t.strip()]
            nums = []
            for t in toks:
                if _is_num(t): # 숫자만 추가
                    nums.append(float(t))
            if len(nums) < 3: # 사이즈, X1,Y1 이하이면 아래 해당 X
                continue
            size = int(round(nums[0]))
            xy = np.array(nums[1:], float) # 1차원으로 나타냄 
            if xy.size % 2 == 1:
                xy = xy[:-1]
            if xy.size < 4:
                continue
            P = xy.reshape(-1,2)
            out[size] = P # 순서대로 (x,y)좌표들을 N개의 컨트롤포인트 행으로 나타냄
    if not out:
        raise RuntimeError(f"No valid numeric rows in {path}")
    return out

# ---------- 비교 ----------
def compare_one_size(Pa, Pb, size, use_bspline=False, degree=3, samples=1500,
                     closed=True, out_dir=".", save_series=False, save_stats=False):
    # 1) 동일 샘플수로 보간/스플라인
    if use_bspline:
        Ca = bspline_curve(Pa, degree=degree, samples=samples, closed=closed)
        Cb = bspline_curve(Pb, degree=degree, samples=samples, closed=closed)
    else:
        Ca = chordlen_resample(Pa, samples)
        Cb = chordlen_resample(Pb, samples)

    # 2) (닫힘이면) 순환 정렬로 점 인덱스 맞춤
    if closed:
        Cb = cyclic_align(Ca, Cb)

    # 3) A의 주축 기준으로 둘 다 '뒤꿈치 y=0, +y=앞꿈치' 프레임으로 강체변환
    v1, v2, _ = pca_major_axis(Ca)
    Ca_h, R, y0 = to_heel_up_frame(Ca, v1=v1, v2=v2, y0_shift=None)
    Cb_h, _, _ = to_heel_up_frame(Cb, v1=v1, v2=v2, y0_shift=y0)  # 동일 y0 기준

    # 4) 거리 계산(강체변환이라 거리 보존)
    d_ab = nearest_distances(Ca_h, Cb_h, chunk=4000)
    d_ba = nearest_distances(Cb_h, Ca_h, chunk=4000)
    stats = summarize_dist(d_ab, d_ba)

    # 5) (선택) 저장물 ...
    # [save_stats, save_series 블록은 기존 그대로]

    # 6) 시각화 (y축 뒤집지 않습니다!)

    if SAVE_PLOTS:
        plt.figure(figsize=(5,9))
        
        # --- [수정] ---
        # 불안정한 시작/끝 10개 샘플을 잘라냅니다.
        TRIM_ENDS = 1
        Ca_h_plot = Ca_h[TRIM_ENDS:-TRIM_ENDS]
        Cb_h_plot = Cb_h[TRIM_ENDS:-TRIM_ENDS]
        # --------------

        # 잘라낸 데이터(plot)로 그림을 그립니다.
        plt.plot(Ca_h_plot[:,0], Ca_h_plot[:,1], lw=2, label="A (Ref)", alpha=0.9)
        plt.plot(Cb_h_plot[:,0], Cb_h_plot[:,1], lw=2, label="B (Pred)", alpha=0.9)



        ax = plt.gca()                     # ← 축 객체 얻기
        ax.set_aspect("equal", "box")      # ← 여기에 적용

        ax.yaxis.set_major_locator(MultipleLocator(10))   # y축 10단위
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.set_ylim(bottom=0)
        ax.grid(True, which='major', linestyle='--', alpha=0.5)


        ttl = (f"Compare A vs B  (size {int(size)})\n"
               f"Chamfer(mean)={stats['Chamfer_mean']:.3f} mm | "
               f"Hausdorff(max)={stats['Hausdorff_max']:.3f} mm")
        plt.title(ttl)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        out_png = os.path.join(out_dir, f"compare_{int(size)}.png")
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()

    return stats



def main():
    # 타임스탬프 출력 폴더 생성
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"compare_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    dataA = load_rows_any(FILE_A)
    dataB = load_rows_any(FILE_B)

    if TARGET_SIZES is None:
        sizes = sorted(set(dataA.keys()).intersection(set(dataB.keys())))
        if not sizes:
            raise RuntimeError("두 파일의 공통 사이즈가 없습니다. TARGET_SIZES를 지정하세요.")
    else:
        sizes = [s for s in TARGET_SIZES if (s in dataA and s in dataB)]
        if not sizes:
            raise RuntimeError("TARGET_SIZES에 해당하는 공통 사이즈가 없습니다.")

    print(f"[INFO] 공통 비교 사이즈: {sizes}")
    print(f"[INFO] Output dir: {os.path.abspath(out_dir)}")

    # 전체 요약 집계(단일 파일)
    summary_rows = []
    for s in sizes:
        Pa, Pb = dataA[s], dataB[s]
        stats = compare_one_size(
            Pa, Pb, s,
            use_bspline=USE_BSPLINE, degree=DEGREE, samples=SAMPLES, closed=CLOSED,
            out_dir=out_dir,
            save_series=SAVE_DISTANCE_SERIES,
            save_stats=SAVE_PER_SIZE_STATS
        )
        row = {"size": int(s)}
        row.update(stats)
        summary_rows.append(row)

    # 단 하나의 요약 CSV만 저장
    summary_csv = os.path.join(out_dir, "compare_summary_all_sizes.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["size",
                  "A_to_B_mean","A_to_B_median","A_to_B_p90","A_to_B_p95","A_to_B_max",
                  "B_to_A_mean","B_to_A_median","B_to_A_p90","B_to_A_p95","B_to_A_max",
                  "Chamfer_mean","Hausdorff_max"]
        w.writerow(header)
        for r in summary_rows:
            w.writerow([r.get(h, "") for h in header])

    print(f"[OK] Saved images and summary to: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
