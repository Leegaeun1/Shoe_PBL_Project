#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
size_to_ctrlpoints_gp.py
- ctrl_points.csv(행: size,x1,y1,...)로부터 가우시안 프로세스 회귀로
  임의의 사이즈(예: 220~300mm, 5mm 간격) 컨트롤포인트를 추정.
- B-스플라인(오픈 유니폼, Cox–de Boor)으로 시각화.
"""

import os, sys, csv, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, WhiteKernel

# ================== 경로/설정 ==================
CSV_PATH = "control_points_master_20250930.csv"   # 기존 데이터
SAVE_PRED = "ctrl_points_pred.csv"  # 예측 저장(새 파일)
TARGET_RANGE = (220, 290)                # 예: 220~290mm
STEP = 5                                  # 5mm 간격
N_CTRL = None                             # None이면 데이터의 '최빈' 포인트 개수로 자동 통일
DEGREE = 3                                # B-spline degree
SPLINE_SAMPLES = 400
# ==============================================

# --------- B-spline 그림그리기----------

def open_uniform_knot_vector(n_ctrl, degree): 
    '''곡선 그릴때 필요한 숫자들의 배열(노트 벡터) 생성'''
    knots = np.concatenate([ 
        np.zeros(degree + 1),
        np.arange(1, n_ctrl - degree),
        np.full(degree + 1, n_ctrl - degree)
    ])
    return knots / np.max(knots)


def bspline_basis(i, degree, knots, t): 
    '''수많은 제어점들 중 특정한 한개의 제어점이 곡선에 얼마나 영향 미치는지 계산
        i번째 제어점이 곡선의 특정 위치 t에서, knots라는 규칙에 따라 얼마나 큰 영향력(w)을 미치는지 계산. '''
    if degree == 0:
        last = (i + 1 == len(knots) - 1)
        if (knots[i] <= t < knots[i+1]) or (last and np.isclose(t, knots[i+1])):
            return 1.0
        return 0.0
    v = 0.0
    d1 = knots[i+degree] - knots[i]
    if d1 > 1e-12:
        v += (t - knots[i]) / d1 * bspline_basis(i, degree-1, knots, t)
    d2 = knots[i+degree+1] - knots[i+1]
    if d2 > 1e-12:
        v += (knots[i+degree+1] - t) / d2 * bspline_basis(i+1, degree-1, knots, t)
    return v

def bspline_curve(ctrl, degree, knots, ts):
    ''' 모든 제어점의 영향력을 합쳐서 최종적인 부드러운 곡선 완성.
    제어점들의 좌표 받아서 곡선 위의 여러 지점(ts) 각각에 대해 모든 제어점의 영향력을
    bsplain_basis로 계산하고 합산 -> 이 점들을 쭉 이으면 부드러운 곡선이됨'''
    out = np.zeros((len(ts), 2), float)
    n = len(ctrl)
    for j, t in enumerate(ts):
        p = np.zeros(2, float)
        for i in range(n):
            w = bspline_basis(i, degree, knots, t)
            if w > 0.0: p += w * ctrl[i]
        out[j] = p
    return out
# ---------------------------------------------------------------

# --------- 유틸: 로드/정규화/재표본/플롯 ----------
def load_rows(path):
    '''데이터를 읽어옴.
    한줄씩 읽어서 (사이즈,[x,y 좌표들]) 목록으로 리턴'''
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            toks = [t for t in ln.replace(",", " ").split() if t]
            vals = list(map(float, toks))
            if len(vals) < 3: continue
            size = int(round(vals[0]))
            xy = np.array(vals[1:], float)
            if len(xy) % 2 == 1: xy = xy[:-1]
            pts = xy.reshape(-1, 2)
            rows.append((size, pts))
    return rows

def arclen_resample(poly, n):
    '''원래의 모양 최대한 유지하면서 점의 개수 늘리거나 줄임.
        poly(점들)을 모두 잇는 선의 전체 길이 계산 후 그 길이를 기준으로 목표 개수(n)만큼
        등간격으로 점 다시 찍을 위치 계산하여 새로운 점들의 좌표 만들어 돌려줌'''
    P = poly.copy()
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.zeros(len(P))
    u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-9:
        return np.repeat(P[:1], n, axis=0)
    u /= L
    s = np.linspace(0, 1, n, endpoint=True)
    x = np.interp(s, u, P[:,0])
    y = np.interp(s, u, P[:,1])
    return np.stack([x,y], axis=1)

def unify_counts(size_pts, n_ctrl=None):
    # 가장 흔한 포인트 개수로 맞추거나 지정값으로 통일
    
    ''' 데이터마다 제각각인 컨트롤 포인트 개수를 하나로 통일시킴.
    개수 부족하거나 많은 데이터는 arclen_resample함수 호출하여 개수 조절 '''
    if n_ctrl is None:
        counts = {}
        for _, pts in size_pts:
            counts[len(pts)] = counts.get(len(pts),0)+1
        n_ctrl = max(counts, key=counts.get)
    out = []
    for s, pts in size_pts:
        if len(pts) != n_ctrl:
            pts = arclen_resample(pts, n_ctrl)
        out.append((s, pts))
    return n_ctrl, out

def flatten_points(pts):
    '''[[x1,y1],[x2,y2]]형태를 [x1,y1,x2,y2]형태로 만듬'''
    return pts.reshape(-1)

def unflatten(vec):
    '''다시 되돌림'''
    a = np.asarray(vec)
    return a.reshape(-1, 2)

def plot_preview(pred_map, title="Predicted Controls via GPR"):
    '''시각화하여 보여줌'''
    # pred_map: {size: (N,2)}
    sizes = sorted(pred_map.keys())
    n_show = min(6, len(sizes))
    show = np.linspace(0, len(sizes)-1, n_show, dtype=int)
    fig, ax = plt.subplots(figsize=(6,8))
    ax.set_aspect("equal","datalim"); ax.grid(True, alpha=0.3); ax.invert_yaxis()
    for idx in show:
        s = sizes[idx]
        ctrl = pred_map[s]
        k = open_uniform_knot_vector(len(ctrl), DEGREE)
        t = np.linspace(0,1,SPLINE_SAMPLES,endpoint=False)
        C = bspline_curve(ctrl, DEGREE, k, t)
        ax.plot(C[:,0], C[:,1], lw=2, label=f"{s}mm")
        ax.scatter(ctrl[:,0], ctrl[:,1], s=18)
    ax.legend(); ax.set_title(title); plt.show()

# --------------------- 메인 ---------------------
def main():
    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: not found → {CSV_PATH}")
        sys.exit(1)

    rows = load_rows(CSV_PATH)  # [(size, (Ni,2)), ...]
    if not rows:
        print("ERROR: no data rows.")
        sys.exit(1)

    # 컨트롤포인트 개수 통일
    n_ctrl, rows = unify_counts(rows, N_CTRL)
    X = np.array([[s] for s,_ in rows], float)         # (M x 1), 입력데이터(문제지)
    '''크기 데이터만 모아서 입력으로 만듬. 
    rows는 현재 (크기, 좌표배열)이므로 s만 필요. [[230],[240]] 이런식으로 저장'''
    Y = np.stack([flatten_points(p) for _,p in rows])  # (M x 2*n_ctrl), 출력데이터(정답지)
    '''컨트롤포인트 좌표 데이터 모아서 출력으로 만듬.
        [[x1,y1,x2,y2....]]'''

    # GPR 커널(부드러움 + 장기추세 + 노이즈)
    '''학습 모델의 성격이나 학습 전략 정함. 
        RBF : 데이터가 부드럽게 변할것임. 크기가 약간 변하면 모양도 약간 변할것.
        RationalQuadratic : RBF보다 유연해서, 크기가 많이 변할때와 적게 변할 때의 다양한 변화 패턴 잡아낼수있음
        DotProduct : 데이터에 선형적인 경향(크기가 커질수록 점들이 비례해서 멀어지는 등)을 파악
        WhiteKernel : 실제 데이터에 섞여있을 미세한 노이즈나 측정 오차 고려.'''
    kernel = (1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
              + 0.5 * RationalQuadratic(alpha=1.0, length_scale=20.0)
              + 0.2 * DotProduct(sigma_0=1.0)
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1)))

    # 출력 독립 가정: 각 차원별 GPR을 하나로 학습 (벡터 회귀)
    '''모델 학습.좌표 하나하나 예측하는 전문가 모델 여러개 만듬.
        x1좌표 전문가, y1좌표 전문가...'''
    models = []
    for d in range(Y.shape[1]):
        ''' Y.shape[1]은 정답지의 열 개수(예측해야할 총 좌표의 개수) -> 전문가를 한명씩 훈련시킴.
            GaussianProcessRegressor 라는 전문가 훈련 기계 만듬. kernel주입.'''
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10, random_state=0)
        ''' X : 모든 전문가에게 동일한 모든 크기 정보 줌.
            Y[:,d] : 정답지는 각 전문가의 전문 분야에 맞는거만줌. d가 0이면 첫번째열(x1좌표)만 줌.
                => 하나의 좌표를 예측하는 전문가 '''
        gp.fit(X, Y[:,d])
        '''훈련이 끝난건 models에 저장됨.'''
        models.append(gp)

    # -------- 새로운 결과 생성 -----------
    # 예측 사이즈(230 이하/이상, 5mm 간격)
    s_min, s_max = TARGET_RANGE
    target_sizes = np.arange(s_min, s_max+1e-9, STEP, dtype=int)
    '''예측하고싶은 크기목록 만듬. [230,235,,,290]'''

    pred = {}
    with open(SAVE_PRED, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for s in target_sizes:
            x = np.array([[float(s)]]) # 새로운 문제. 230일때는 어떻게 해야해?
            yhat = np.array([m.predict(x)[0] for m in models])  
            '''예측의 핵심! 전문가를 한명씩 찾아가서, 새로운 문제인 x를 보여주고 무엇인지 물어서 예측해서 알려줌.
            [예측된x1,예측된y1,,,,]'''
            ctrl = unflatten(yhat) # 다시 [[x1,y1]]형태로 만듬
            pred[s] = ctrl
            row = [s] + [f"{v:.6f}" for v in yhat]
            w.writerow(row) # 기록
    print(f"[OK] predicted {len(target_sizes)} sizes → {SAVE_PRED}")

    # 미리보기(선택)
    plot_preview(pred, title="Predicted Controls via Gaussian Process Regression")

if __name__ == "__main__":
    main()
