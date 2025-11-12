#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_230_to_250_270_krr_only.py

- 학습: train_230.csv, train_250.csv, train_270.csv   (각 행: [size?, x1,y1,x2,y2,...])
- 입력: new_230.csv                                   (1행: [size?, x1,y1,...])
- 출력: mapped_230_250_270.csv                        (3행: 230 / 250 / 270)

특징
- CSV 맨 앞 size(200~320)가 있으면 제거, 홀수 길이는 꼬리값 drop
- 모든 행을 '가장 흔한 포인트 수'로 chord-length 리샘플
- Kernel Ridge Regression (KRR)만 사용 (MultiOutputRegressor 없이 전체 좌표 벡터 직접 회귀)
- 옵션: 델타 회귀, 사이즈 락(세로 길이 정확히 230/250/270), (0,0) 기준 정렬
"""

import os, re, csv
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
# KRR 사용을 위해 KernelRidge만 가져옵니다. (MultiOutputRegressor는 KRR에 불필요)
from sklearn.kernel_ridge import KernelRidge 


# ========= 설정 =========
TRAIN_230 = "train_230.csv"
TRAIN_250 = "train_250.csv"
TRAIN_270 = "train_270.csv"
NEW_230   = "new_230.csv"
OUT_CSV   = "mapped_230_250_270.csv"

USE_DELTA_REGRESSION = True    # True면 Δ=Y−X를 학습, 예측은 X+Δ̂
LOCK_HEIGHT = True             # True면 세로 길이를 정확히 230/250/270으로 강제
LOCK_MODE   = "yonly"          # "yonly"(폭 유지) or "iso"(가로세로 등비)
LOCK_ANCHOR = "miny"           # "miny" or "centroid"
LOCK_INPUT_230 = True         # 입력 new230도 정확히 230으로 맞출지 여부

SHIFT_TO_ZERO = True           # 예측 후 좌상단(0,0) 기준으로 평행이동
# ========================

# --------- CSV 로드 & 전처리 ---------
_NUM = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][\+\-]?\d+)?$')
def _num_tokens(line):
    toks = [t for t in line.replace(",", " ").replace(";", " ").split() if t]
    out = []
    for t in toks:
        t = t.strip().lstrip("\ufeff")
        if _NUM.match(t):
            out.append(float(t))
    return out

def read_rows(path): 
    '''숫자 데이터를 읽어옴.'''
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if not s or s.startswith("#"): continue
            vals = _num_tokens(s)
            if not vals: continue
            # 맨 앞 값이 mm 사이즈면 제거
            if (len(vals) % 2 == 1) and (200.0 <= vals[0] <= 320.0):
                vals = vals[1:]
            # 홀수면 꼬리 제거
            if len(vals) % 2 == 1:
                vals = vals[:-1]
            if len(vals) >= 4:
                rows.append(np.array(vals, float))
    if not rows:
        raise RuntimeError(f"No valid numeric rows in {path}")
    return rows

def vector_to_points(v):
    '''1차원 벡터를 2차원 좌표 (N, 2) 배열로 변환.'''
    v = np.asarray(v, float)
    if v.size % 2 == 1: v = v[:-1]
    return v.reshape(-1, 2)

def points_to_vector(P):
    '''2차원 좌표 배열을 1차원 벡터로 변환.'''
    return np.asarray(P, float).reshape(-1)

# --------- 리샘플(등시선) ---------
def chordlen_resample(P, n):
    '''주어진 좌표들을 연결한 곡선의 전체 길이 구함 -> 등간격으로 이동하여 n개의 새로운 점 추출'''
    P = np.asarray(P, float)
    if len(P) == n:
        return P.copy()
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.zeros(len(P)); u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-9:
        return np.repeat(P[:1], n, axis=0)
    u /= L
    s = np.linspace(0, 1, n, endpoint=True)
    x = np.interp(s, u, P[:,0]); y = np.interp(s, u, P[:,1])
    return np.stack([x, y], axis=1)

def most_common_points_count(rows_list):
    ''' 전체 학습 데이터에서 가장 빈번하게 나타나는 점의 개수를 찾아 리샘플링의 기준(n)으로 삼음'''
    counts = [vector_to_points(r).shape[0] for r in rows_list]
    c = Counter(counts).most_common(1)[0][0]
    return int(c)

# --------- 유틸(스케일·정렬) ---------
def height_of(P):
    P = vector_to_points(P) if P.ndim == 1 else np.asarray(P)
    return float(P[:,1].max() - P[:,1].min())

def lock_height_yonly(P, target_mm, anchor="miny"):
    """세로만 스케일(폭 유지). P: (K,2) 또는 (2K,) 허용."""
    Q = vector_to_points(P) if P.ndim == 1 else P.copy()
    y0, y1 = Q[:,1].min(), Q[:,1].max()
    h = y1 - y0
    if h <= 1e-9: return Q if P.ndim==2 else points_to_vector(Q)
    r = float(target_mm) / h
    if anchor == "miny":
        Q[:,1] = (Q[:,1] - y0) * r + y0
    else:
        c = Q.mean(axis=0)
        Q[:,1] = (Q[:,1] - c[1]) * r + c[1]
    return Q if P.ndim==2 else points_to_vector(Q)

def lock_height_iso(P, target_mm, anchor="centroid"):
    """가로·세로 등비 스케일."""
    Q = vector_to_points(P) if P.ndim == 1 else P.copy()
    y0, y1 = Q[:,1].min(), Q[:,1].max()
    h = y1 - y0
    if h <= 1e-9: return Q if P.ndim==2 else points_to_vector(Q)
    r = float(target_mm) / h
    origin = (Q.mean(axis=0) if anchor=="centroid" else np.array([Q[:,0].mean(), y0], float))
    Q = (Q - origin) * r + origin
    return Q if P.ndim==2 else points_to_vector(Q)

def shift_min_to_zero(P, axes="both"):
    """좌상단 (0,0) 기준으로 평행이동."""
    Q = vector_to_points(P) if P.ndim == 1 else P.copy()
    if "x" in axes: Q[:,0] -= Q[:,0].min()
    if "y" in axes: Q[:,1] -= Q[:,1].min()
    return Q if P.ndim==2 else points_to_vector(Q)

# --------- KRR(단독) 매핑 ---------
class PureKRRMap:
    """
    X(230 벡터) -> Y(target 벡터) 매핑을 위한 KRR 모델.
    - 표준화 적용
    - USE_DELTA_REGRESSION=True면 Δ=Y−X를 학습, 예측은 X+Δ̂
    """
    def __init__(self, alpha=1e-1, gamma=1e-4, use_delta=USE_DELTA_REGRESSION):
        # KRR 파라미터 기본값 설정
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.use_delta = use_delta
        # KernelRidge는 MultiOutputRegressor 없이 다중 출력 지원
        self.model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
        self._X_mean_ = None

    def fit(self, X, Y): 
        '''모델 학습함수'''
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        Xs = self.scaler_x.fit_transform(X)
        if self.use_delta:
            Y_use = Y - X
        else:
            Y_use = Y
        Ys = self.scaler_y.fit_transform(Y_use)
        self.model.fit(Xs, Ys)
        self._X_mean_ = X.mean(axis=0)
        return self

    def predict(self, Xnew):
        '''새로운 입력 데이터 Xnew에 대한 예측 수행.'''
        Xnew = np.asarray(Xnew, float)
        Xs = self.scaler_x.transform(Xnew)
        Ys = self.model.predict(Xs)
        Y_use = self.scaler_y.inverse_transform(Ys)
        if self.use_delta:
            return Xnew + Y_use
        return Y_use

# --------- 저장 ---------
def save_series_onefile(path, P230, P250, P270, sep=","):
    def row(size, P):
        # P가 2D (K, 2)이면 1D로 변환 후 저장
        return [str(int(size))] + [f"{v:.6f}" for v in np.asarray(P).reshape(-1)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=sep)
        w.writerow(row(230, P230))
        w.writerow(row(250, P250))
        w.writerow(row(270, P270))
    print(f"[OK] saved → {path} (rows: 230, 250, 270)")

# --------- 메인 ---------
def main():
    # 1) 학습 로드
    rows230 = read_rows(TRAIN_230)
    rows250 = read_rows(TRAIN_250)
    rows270 = read_rows(TRAIN_270)

    n = min(len(rows230), len(rows250), len(rows270))
    rows230, rows250, rows270 = rows230[:n], rows250[:n], rows270[:n]
    if n < 3:
        print(f"[WARN] very few samples (n={n}). KRR may perform poorly.")

    # 2) 포인트 수 통일(최빈)
    target_pts = most_common_points_count(rows230 + rows250 + rows270)

    def to_vec_resampled(v):
        P = vector_to_points(v)
        P = chordlen_resample(P, target_pts)
        return points_to_vector(P)

    X230 = np.vstack([to_vec_resampled(v) for v in rows230])  # (N, 2K)
    Y250 = np.vstack([to_vec_resampled(v) for v in rows250])  # (N, 2K)
    Y270 = np.vstack([to_vec_resampled(v) for v in rows270])  # (N, 2K)

    # 3) 새 230
    new230_vec = to_vec_resampled(read_rows(NEW_230)[0])
    new230_pts = vector_to_points(new230_vec)

    # 4) KRR 학습
    map_230_to_250 = PureKRRMap(use_delta=USE_DELTA_REGRESSION)
    map_230_to_270 = PureKRRMap(use_delta=USE_DELTA_REGRESSION)
    map_230_to_250.fit(X230, Y250)
    map_230_to_270.fit(X230, Y270)

    # 5) 예측 (벡터)
    P250_vec = map_230_to_250.predict(new230_vec[None, :])[0]
    P270_vec = map_230_to_270.predict(new230_vec[None, :])[0]

    # 6) 후처리: 사이즈 락 + (0,0) 기준 정렬
    P230 = new230_pts.copy()
    P250 = vector_to_points(P250_vec)
    P270 = vector_to_points(P270_vec)

    # 사이즈 락 (세로 길이 강제 보정)
    if LOCK_HEIGHT:
        lock_func = lock_height_iso if LOCK_MODE == "iso" else lock_height_yonly
        
        if LOCK_INPUT_230:
            P230 = lock_func(P230, 230.0, anchor=LOCK_ANCHOR)
            
        P250 = lock_func(P250, 250.0, anchor=LOCK_ANCHOR)
        P270 = lock_func(P270, 270.0, anchor=LOCK_ANCHOR)
    
    # 0,0 기준 정렬
    if SHIFT_TO_ZERO:
        P230 = shift_min_to_zero(P230, axes="both")
        P250 = shift_min_to_zero(P250, axes="both")
        P270 = shift_min_to_zero(P270, axes="both")

    # 7) 저장(3행 한 파일)
    save_series_onefile(OUT_CSV, P230, P250, P270)

    # 8) 진단 출력
    def H(P): return height_of(P)
    print(f"[HEIGHT mm] 230={H(P230):.2f}, 250_pred={H(P250):.2f}, 270_pred={H(P270):.2f}")
    print(f"[POINTS] K={len(P230)} (all equal across rows)")

if __name__ == "__main__":
    main()