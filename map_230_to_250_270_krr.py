#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_230_to_250_270_krr.py
- 5쌍 학습용: (230_i, 250_i, 270_i), i=1..5
- 새로운 230_new가 들어오면 → (250_pred, 270_pred) 추정.
- 입력 CSV 예시(각 5행):
    train_230.csv / train_250.csv / train_270.csv
    각 행: x1,y1,x2,y2,...  (같은 포인트 수)
- 옵션: 포인트 수가 다르면 자동 호장 재표본으로 통일.
"""

import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_distances


DEGREE = 3
SPLINE_SAMPLES = 400
N_CTRL = None  # None이면 자동 통일




def to_ctrl(v): return np.asarray(v, float).reshape(-1,2)

def bbox_h(vec):
    P = to_ctrl(vec)
    return (P[:,1].max() - P[:,1].min())

def median_height_ratio(X_rows, Y_rows):
    # 행별 높이비의 중앙값
    ratios = []
    for x, y in zip(X_rows, Y_rows):
        hx, hy = bbox_h(x), bbox_h(y)
        if hx > 1e-9:
            ratios.append(hy / hx)
    if not ratios:
        return 1.0
    ratios = np.array(ratios, float)
    return float(np.median(ratios))

def median_heuristic_gamma(X):
    D = pairwise_distances(X)
    D = D[np.triu_indices_from(D, k=1)]
    med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    return 1.0 / (2.0 * (med**2 + 1e-12))

def isotropic_rescale_about_centroid(base_vec, pred_vec, ratio):
    """pred_vec을 base_vec의 중심을 기준으로 등비(ratio) 스케일."""
    Pb = to_ctrl(base_vec)
    Pp = to_ctrl(pred_vec)
    c = Pb.mean(axis=0)
    Pp2 = (Pp - c) * ratio + c
    return Pp2.reshape(-1)



# ----- B-spline -----
def open_uniform_knot_vector(n_ctrl, degree):
    knots = np.concatenate([
        np.zeros(degree + 1),
        np.arange(1, n_ctrl - degree),
        np.full(degree + 1, n_ctrl - degree)
    ])
    return knots / np.max(knots)

def bspline_basis(i, degree, knots, t):
    if degree == 0:
        last = (i + 1 == len(knots) - 1)
        if (knots[i] <= t < knots[i+1]) or (last and np.isclose(t, knots[i+1])):
            return 1.0
        return 0.0
    v=0.0
    d1=knots[i+degree]-knots[i]
    if d1>1e-12: v+=(t-knots[i])/d1*bspline_basis(i,degree-1,knots,t)
    d2=knots[i+degree+1]-knots[i+1]
    if d2>1e-12: v+=(knots[i+degree+1]-t)/d2*bspline_basis(i+1,degree-1,knots,t)
    return v

def bspline_curve(ctrl, degree, knots, ts):
    out = np.zeros((len(ts), 2), float)
    for j,t in enumerate(ts):
        wsum = np.zeros(2)
        for i in range(len(ctrl)):
            w = bspline_basis(i, degree, knots, t)
            if w>0: wsum += w*ctrl[i]
        out[j]=wsum
    return out

# ----- 유틸 -----
def read_matrix(path):
    M = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            toks = [t for t in s.replace(",", " ").split() if t]
            try:
                vals = [float(t) for t in toks]
            except Exception:
                continue

            # 맨 앞 값이 mm 사이즈로 보이면 제거
            if (len(vals) % 2 == 1) and (200.0 <= vals[0] <= 320.0):
                vals = vals[1:]

            # 여전히 홀수면 꼬리 값 1개 제거(로그 남김)
            if len(vals) % 2 == 1:
                print(f"[WARN] {os.path.basename(path)}: odd-length row -> dropping last value")
                vals = vals[:-1]

            if len(vals) >= 4:
                M.append(np.array(vals, float))
    if not M:
        raise RuntimeError(f"no rows in {path}")
    return M


def arclen_resample(poly, n):
    P=poly.reshape(-1,2)
    seg=np.linalg.norm(np.diff(P,axis=0),axis=1)
    u=np.zeros(len(P)); u[1:]=np.cumsum(seg)
    L=u[-1]
    if L<=1e-9: return np.repeat(P[:1], n, axis=0).reshape(-1)
    u/=L
    s=np.linspace(0,1,n,endpoint=True)
    x=np.interp(s,u,P[:,0]); y=np.interp(s,u,P[:,1])
    return np.stack([x,y],axis=1).reshape(-1)

def unify_counts_list(list_of_flat, n_ctrl=None):
    if n_ctrl is None:
        # 최빈 포인트 수
        counts={}
        for v in list_of_flat:
            counts[len(v)//2]=counts.get(len(v)//2,0)+1
        n_ctrl=max(counts,key=counts.get)
    out=[]
    for v in list_of_flat:
        if len(v)//2 != n_ctrl:
            v = arclen_resample(v, n_ctrl)
        out.append(v)
    return n_ctrl, out

def to_ctrl(v): return v.reshape(-1,2)

def save_series_onefile(path, vec230, vec250, vec270, sep=","):
    """
    한 파일에 3행 저장:
      230, x1,y1,x2,y2, ...
      250, x1,y1,x2,y2, ...
      270, x1,y1,x2,y2, ...
    """
    def make_row(size, vec):
        return [str(int(size))] + [f"{v:.6f}" for v in np.asarray(vec).reshape(-1)]

    # (선택) 헤더 넣고 싶으면 아래 3줄 주석 해제
    # n = len(vec230)//2
    # header = ["size"] + [f"x{i}" for i in range(1,n+1)] + [f"y{i}" for i in range(1,n+1)]
    # writer.writerow(header)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerow(make_row(230, vec230))
        writer.writerow(make_row(250, vec250))
        writer.writerow(make_row(270, vec270))
    print(f"[OK] saved → {path} (rows: 230, 250, 270)")


def plot_pair(ctrl230, ctrlT, title):
    fig, ax = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)

    # 곡선 먼저 둘 다 계산
    K0 = open_uniform_knot_vector(len(ctrl230), DEGREE)
    K1 = open_uniform_knot_vector(len(ctrlT),   DEGREE)
    t  = np.linspace(0,1,SPLINE_SAMPLES,endpoint=False)
    C0 = bspline_curve(ctrl230,DEGREE,K0,t)
    C1 = bspline_curve(ctrlT,   DEGREE,K1,t)

    # 공통 축 범위
    all_xy = np.vstack([C0, C1, ctrl230, ctrlT])
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)

    # 왼쪽: 230
    ax[0].plot(C0[:,0], C0[:,1], lw=2); ax[0].scatter(ctrl230[:,0], ctrl230[:,1], s=18)
    ax[0].set_title("230 input")
    # 오른쪽: target
    ax[1].plot(C1[:,0], C1[:,1], lw=2); ax[1].scatter(ctrlT[:,0], ctrlT[:,1], s=18)
    ax[1].set_title(title)

    for k in (0,1):
        ax[k].set_aspect("equal")
        ax[k].set_xlim(xmin, xmax)
        ax[k].set_ylim(ymax, ymin)  # 아래로 증가하는 이미지 좌표계이면 invert 대신 이렇게
        ax[k].grid(True, alpha=0.3)
        # ax[k].invert_yaxis() 를 쓰고 싶다면 위 set_ylim 대신:
        #   ax[k].set_ylim(ymin, ymax); ax[k].invert_yaxis()
    plt.show()



def main():
    # 학습 데이터 파일(각 5행 가정)
    p230="train_230.csv"; p250="train_250.csv"; p270="train_270.csv"
    if not all(os.path.isfile(p) for p in [p230,p250,p270]):
        print("학습 CSV가 필요합니다:\n - train_230.csv\n - train_250.csv\n - train_270.csv\n각 행: x1,y1,x2,y2,...  (5행 권장)")
        sys.exit(1)

    X_list = read_matrix(p230)
    Y250_list = read_matrix(p250)
    Y270_list = read_matrix(p270)

    # 포인트 수 통일
    n230, X_list = unify_counts_list(X_list, N_CTRL)
    n250, Y250_list = unify_counts_list(Y250_list, n230)  # 230과 동일 개수로 맞춤
    n270, Y270_list = unify_counts_list(Y270_list, n230)

    X = np.vstack(X_list)                # (5, 2N)
    Y250 = np.vstack(Y250_list)          # (5, 2N)
    Y270 = np.vstack(Y270_list)          # (5, 2N)

    # 커널 릿지 회귀(RBF): α는 정규화(스무딩), γ는 길이척도(=1/(2ℓ²))
    krr250 = KernelRidge(alpha=1e-1, kernel="rbf", gamma=1e-4)
    krr270 = KernelRidge(alpha=1e-1, kernel="rbf", gamma=1e-4)
    krr250.fit(X, Y250)
    krr270.fit(X, Y270)

    # 새로운 230 입력 파일
    new230_path = "new_230.csv"
    if not os.path.isfile(new230_path):
        print(f"새로운 230 컨트롤포인트 파일이 필요합니다: {new230_path}\n(한 행: x1,y1,x2,y2,...)")
        sys.exit(0)

    new230 = np.array(read_matrix(new230_path)[0])
    if len(new230)//2 != n230:
        new230 = arclen_resample(new230, n230)

    pred250 = krr250.predict(new230[None,:])[0]
    pred270 = krr270.predict(new230[None,:])[0]

    # 저장
    save_series_onefile("pred_series_230_250_270.csv", new230, pred250, pred270)


    # 시각화
    plot_pair(to_ctrl(new230), to_ctrl(pred250), "250 predicted")
    plot_pair(to_ctrl(new230), to_ctrl(pred270), "270 predicted")

if __name__ == "__main__":
    main()
