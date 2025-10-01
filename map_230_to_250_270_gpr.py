#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_230_to_250_270_gpr_only.py

- 학습: train_230.csv, train_250.csv, train_270.csv   (각 행: [size?, x1,y1,x2,y2,...])
- 입력: new_230.csv                                   (1행: [size?, x1,y1,...])
- 출력: mapped_230_250_270.csv                        (3행: 230 / 250 / 270)

특징
- CSV 맨 앞 size(200~320)가 있으면 제거, 홀수 길이는 꼬리값 drop
- 모든 행을 '가장 흔한 포인트 수'로 chord-length 리샘플
- PCA 없이 GPR만 사용 (MultiOutputRegressor로 전체 좌표 벡터 직접 회귀)
"""

import os, re, csv
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# --------- 경로 ---------
TRAIN_230 = "train_230.csv"
TRAIN_250 = "train_250.csv"
TRAIN_270 = "train_270.csv"
NEW_230   = "new_230.csv"
OUT_CSV   = "mapped_230_250_270.csv"
# ------------------------

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
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if not s or s.startswith("#"): continue
            vals = _num_tokens(s)
            if not vals: continue
            # 맨 앞 값이 mm 사이즈로 보이면 제거
            if (len(vals) % 2 == 1) and (200.0 <= vals[0] <= 320.0):
                vals = vals[1:]
            # 그래도 홀수면 꼬리 제거
            if len(vals) % 2 == 1:
                vals = vals[:-1]
            if len(vals) >= 4:
                rows.append(np.array(vals, float))
    if not rows:
        raise RuntimeError(f"No valid numeric rows in {path}")
    return rows

def vector_to_points(v):
    v = np.asarray(v, float)
    if v.size % 2 == 1:
        v = v[:-1]
    return v.reshape(-1, 2)

def points_to_vector(P):
    return np.asarray(P, float).reshape(-1)

# --------- 리샘플(등시선) ---------
def chordlen_resample(P, n):
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
    counts = [vector_to_points(r).shape[0] for r in rows_list]
    c = Counter(counts).most_common(1)[0][0]
    return int(c)

# --------- GPR(단독) 매핑 ---------
class PureGPRMap:
    """
    X(230 벡터) -> Y(target 벡터)
    - X, Y 둘 다 표준화만 하고 PCA는 사용하지 않음
    - 전체 좌표(2K 차원)를 MultiOutputRegressor로 한 번에 예측
    """
    def __init__(self, length_scale=5.0, noise_level=1e-3,
                 n_restarts=5, random_state=0):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3)) \
                 + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-6, 1e-1))
        self.model = MultiOutputRegressor(
            GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=False,
                                     random_state=random_state, n_restarts_optimizer=n_restarts)
        )

    def fit(self, X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        Xs = self.scaler_x.fit_transform(X)
        Ys = self.scaler_y.fit_transform(Y)
        self.model.fit(Xs, Ys)
        return self

    def predict(self, Xnew):
        Xnew = np.asarray(Xnew, float)
        Xs = self.scaler_x.transform(Xnew)
        Ys = self.model.predict(Xs)
        Y = self.scaler_y.inverse_transform(Ys)
        return Y

# --------- 저장 ---------
def save_series_onefile(path, P230, P250, P270, sep=","):
    def row(size, P):
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
        print(f"[WARN] very few samples (n={n}). GPR may overfit.")

    # 2) 전체에서 가장 흔한 포인트 수로 리샘플
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

    # 4) GPR 단독 맵 학습
    map_230_to_250 = PureGPRMap(length_scale=5.0, noise_level=1e-3, n_restarts=5, random_state=0)
    map_230_to_270 = PureGPRMap(length_scale=5.0, noise_level=1e-3, n_restarts=5, random_state=0)
    map_230_to_250.fit(X230, Y250)
    map_230_to_270.fit(X230, Y270)

    # 5) 예측
    P250_vec = map_230_to_250.predict(new230_vec[None, :])[0]
    P270_vec = map_230_to_270.predict(new230_vec[None, :])[0]
    P250 = vector_to_points(P250_vec)
    P270 = vector_to_points(P270_vec)

    # 6) 저장(3행 한 파일)
    save_series_onefile(OUT_CSV, new230_pts, P250, P270)

if __name__ == "__main__":
    main()
