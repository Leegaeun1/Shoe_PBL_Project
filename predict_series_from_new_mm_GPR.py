#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_series_from_new_230_gpr_trackselect.py

- 학습 CSV 안에 여러 사람(모델)의 다양한 사이즈가 섞여 있어도,
  '학습 CSV의 최소 사이즈(min_size)' 형상을 베이스로 삼아
  new_230(사용자 입력)의 형상과 가장 가까운 '트랙'을 자동 선택.
- 선택된 트랙의 각 사이즈(s)에서, 베이스 형상 대비 접선/법선 변형량(dt, dn)을 계산.
- dt(s), dn(s)를 Gaussian Process Regression(GPR)으로 학습하여
  목표 사이즈(230, 235, ..., 290)의 변형량을 예측하고,
  new_230 형상에 합성하여 컨트롤 포인트를 생성.

입력:
  TRAIN_CSV = "control_points_master_20251013.csv"  # 행: size, x1,y1,x2,y2,...
  NEW230_CSV = "new_230.csv"                        # 한 줄: (선택) size, x1,y1,...

출력:
  SAVE_PRED  = "pred_series.csv"                    # 행: size, x1,y1,...

필요:
  numpy, scikit-learn
"""

import os, re, csv
import numpy as np

# -------------------- 경로 설정 --------------------
TRAIN_CSV = "control_points_master_20251013.csv"  # 학습 데이터(여러 사이즈)
NEW230_CSV = "new_230.csv"                        # 새로운 230 데이터(사이즈 값 포함 가능)
SAVE_PRED  = "pred_series.csv"                    # 저장 파일
# --------------------------------------------------

# ---------- 강인한 파일 리더 (CSV/BOM/CP949 등) ----------
def _read_text(path, encodings=("utf-8-sig","utf-8","cp949","euc-kr","latin-1")):
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception as e:
            last_err = e
            continue
    raise last_err

_NUM = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][\+\-]?\d+)?$')
def _num_tokens(line):
    toks = [t for t in line.replace(",", " ").replace(";", " ").split() if t]
    out = []
    for t in toks:
        t = t.strip().lstrip("\ufeff")
        if _NUM.match(t):
            out.append(float(t))
    return out

def load_train_rows(path):
    """
    학습 CSV에서 (size, P) 목록을 읽는다.
    - 사이즈 필터링을 제거하여(또는 완화하여) 230보다 작은 사이즈도 허용.
    """
    text = _read_text(path)
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): 
            continue
        vals = _num_tokens(ln)
        if len(vals) < 3: 
            continue
        size = int(round(vals[0]))
        xy = np.array(vals[1:], float)
        if len(xy) % 2 == 1:  # 꼬임 방지
            xy = xy[:-1]
        P = xy.reshape(-1,2)
        rows.append((size, P))
    if not rows:
        raise RuntimeError("No valid train rows found in the training CSV.")
    return rows

def load_new230_any(path):
    text = _read_text(path)
    first = None
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): 
            continue
        vals = _num_tokens(ln)
        if len(vals) >= 2:
            first = np.array(vals, float)
            break
    if first is None:
        raise RuntimeError(f"No numeric row found in {path}")

    # 맨 앞 값이 사이즈(mm)면 드랍
    if 100.0 <= first[0] <= 500.0 and (len(first)-1) >= 2:
        first = first[1:]

    if len(first) % 2 == 1:
        first = first[:-1]

    if len(first) < 4 or len(first) % 2 != 0:
        raise ValueError("new_230.csv must contain an even number of coordinates (>=4).")

    return first.reshape(-1,2)

# ---------------- 기하 유틸 ----------------
def chordlen_resample(P, n):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.zeros(len(P))
    if len(P) > 1:
        u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-9:
        return np.repeat(P[:1], n, axis=0)
    u /= L
    s = np.linspace(0,1,n,endpoint=True)
    x = np.interp(s, u, P[:,0]); y = np.interp(s, u, P[:,1])
    return np.stack([x,y], axis=1)

def _align_score(P, Q):
    return float(np.sum((P-Q)**2))

def cyclic_align(P, Q):
    """P와 Q를 순환 시프트로 최적 정렬 (정방향/역방향 모두 시도)."""
    n = len(P)
    best = (None, 10**30, 0, False)  # (Q_best, score, shift, reversed?)
    for rev in [False, True]:
        R = Q[::-1].copy() if rev else Q.copy()
        for k in range(n):
            Rk = np.roll(R, -k, axis=0)
            sc = _align_score(P, Rk)
            if sc < best[1]:
                best = (Rk, sc, k, rev)
    return best  # Q_best, score, shift, reversed?

def tangents_normals(P):
    N = len(P)
    T = np.zeros_like(P)
    if N >= 2:
        T[1:-1] = P[2:] - P[:-2]
        T[0]  = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    denom = np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    T = T / denom
    Nvec = np.stack([-T[:,1], T[:,0]], axis=1)
    return T, Nvec

# -------- 트랙 선택 (베이스 = 학습 CSV의 최소 사이즈) --------
def select_track(rows, P_new_resampled, min_keep=230, max_keep=290):
    """
    rows: [(size, P)], 여러 사람 데이터 섞여 있음
    1) 학습 CSV 내 최솟값 사이즈(min_size) 후보들 중에서, new_230과 가장 가까운 형상을 베이스로 선택
    2) 각 size s(여기서는 [min_keep..max_keep] 범위)에 대해 베이스와 가장 가까운 행을 1개씩 채택
    반환: sorted track [(s, P_s_aligned)], base_at_min_size
    """
    # 모든 행을 new_230 길이에 맞춤
    L = len(P_new_resampled)
    rows_std = [(s, chordlen_resample(P, L)) for (s,P) in rows]

    # 최솟값 사이즈(min_size) 후보 선택
    all_sizes = [s for s,_ in rows_std]
    min_size = min(all_sizes)
    cand_base = [(s,P) for (s,P) in rows_std if s == min_size]
    if not cand_base:
        raise RuntimeError("No rows at the minimum size found in training data.")
    # new_230와 가장 가까운 베이스 찾기
    best = (None, 10**30, None)
    for _, P in cand_base:
        Q_best, sc, _, _ = cyclic_align(P_new_resampled, P)
        if sc < best[1]:
            best = (P, sc, Q_best)
    base_train_min, base_dist, base_aligned = best
    base = base_train_min  # 베이스는 '학습 최솟값 사이즈'의 원본(정렬 전) 형상

    # 2) 관심 사이즈 범위[min_keep..max_keep]에서 각 s의 '베이스에 가장 가까운 행'을 선정
    sizes_all = sorted(set(s for s,_ in rows_std if (min_keep is None or s >= min_keep) and (max_keep is None or s <= max_keep)))
    track = []
    for s in sizes_all:
        cand = [P for (ss,P) in rows_std if ss == s]
        if not cand: 
            continue
        best_s = (None, 10**30)
        for P in cand:
            Q_best, sc, _, _ = cyclic_align(base, P)
            if sc < best_s[1]:
                best_s = (Q_best, sc)
        track.append((s, best_s[0]))

    track.sort(key=lambda x: x[0])
    return track, base

def pca_major_axis(P):
    """
    P: (N,2)
    반환: v1(PC1 단위벡터), v2(PC2 단위벡터), heel_idx(PC1 최소 투영점 인덱스), L(PC1 길이)
    """
    C = P - P.mean(axis=0, keepdims=True)
    # SVD로 직교기저 획득 (Vt의 첫 행이 PC1)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    v1 = Vt[0]    # (2,)
    v2 = Vt[1]    # (2,)
    # PC1 투영으로 heel/toe 찾기
    z1 = (P @ v1)
    heel_idx = int(np.argmin(z1))
    L = float(z1.max() - z1.min())
    return v1, v2, heel_idx, L

def shrink_along_pc1(P, target_L, eps=1e-9):
    """
    P를 PC1(heel 고정) 방향으로만 비율 축소해서 target_L에 맞춘다.
    현재 길이 <= target_L이면 원본 반환.
    """
    v1, v2, heel_idx, L = pca_major_axis(P)
    if L <= target_L + eps:
        return P  # 줄일 필요 없음
    heel = P[heel_idx]  # heel을 기준점으로 고정
    R = P - heel
    # v1,v2 좌표계로 변환
    r1 = R @ v1  # (N,)
    r2 = R @ v2  # (N,)
    # heel 기준 r1.min()은 0 또는 0 근처
    L_current = float(r1.max() - r1.min())
    if L_current < eps:
        return P
    alpha = target_L / L_current  # 0 < alpha < 1
    r1_new = r1 * alpha
    P_new = heel + np.outer(r1_new, v1) + np.outer(r2, v2)
    return P_new

def enforce_size_caps_monotone(P_list, sizes, tol_mm=0.0):
    """
    P_list: [P_s ...] 예측된 형상 리스트 (sizes와 동일 길이, 230→290 오름차순)
    sizes : [230,235,...] (오름차순)
    tol_mm: 허용 오차(+1mm 이내 OK)

    단계:
    1) 각 P_s의 PC1 길이 L_s 계산
    2) 상한 U_s = s + tol_mm 로 클립 (넘으면 줄임)
    3) 큰 사이즈로 갈수록 길이가 감소하지 않도록 '오직 감소만 허용'하는 역방향 단조 보정
       (작은 사이즈가 큰 사이즈보다 길면 작은 사이즈를 더 줄임)
    4) 각 P_s를 필요 시 PC1 방향으로만 축소 (heel 고정)
    """
    n = len(P_list)
    # 1) 길이 계산
    L_pred = []
    for P in P_list:
        _, _, _, L = pca_major_axis(P)
        L_pred.append(L)
    L_pred = np.array(L_pred, float)

    # 2) 상한 클립: U_s = s + tol_mm (요구: "넘지 않음, 오차 1mm 허용")
    U = np.array(sizes, float) + float(tol_mm)
    L_cap = np.minimum(L_pred, U)

    # 3) 역방향 단조(비감소) 보정: 작은 사이즈 ≤ 큰 사이즈가 되도록
    #    오른쪽→왼쪽으로 누적 최소값을 유지(오직 감소만 허용)
    L_adj = L_cap.copy()
    for i in range(n-2, -1, -1):
        L_adj[i] = min(L_adj[i], L_adj[i+1])

    # 4) 스케일 적용
    P_adj_list = []
    for P, Lp, La in zip(P_list, L_pred, L_adj):
        if Lp <= La + 1e-9:
            P_adj_list.append(P)  # 손대지 않음
        else:
            P_adj_list.append(shrink_along_pc1(P, La))
    return P_adj_list, L_pred, L_adj

# -------- 꼬리(외삽) 가드가 포함된 GPR 예측 --------
def _linear_fit_multi(x, Y):
    """x: (k,), Y: (k,L) -> 각 열에 대해 y = a*x + b 를 벡터화로 적합, 반환 (a,b) 각각 (L,)"""
    x = np.asarray(x, float)
    Y = np.asarray(Y, float)
    X = np.stack([x, np.ones_like(x)], axis=1)          # (k,2)
    XtX = X.T @ X                                       # (2,2)
    # 안정성용 작은 정규화
    XtX += 1e-12 * np.eye(2)
    beta = np.linalg.inv(XtX) @ (X.T @ Y)               # (2,L)
    a, b = beta[0], beta[1]                             # 각 (L,)
    return a, b

def _linear_predict_multi(a, b, x):
    """a,b: (L,), x: float -> (L,)"""
    return a * float(x) + b

def _blend_to_boundary(Y_linear, Y_boundary, dist_mm, tau_mm=8.0):
    """
    경계값(Y_boundary)와 선형외삽(Y_linear)을 부드럽게 블렌딩.
    dist_mm: 경계까지의 거리(양수), tau_mm이 작을수록 경계 쪽으로 강하게 수축.
    """
    # gamma=1이면 경계값, 0이면 선형외삽
    gamma = np.exp(-dist_mm / max(tau_mm, 1e-6))
    return gamma * Y_boundary + (1.0 - gamma) * Y_linear

def gpr_fit_predict_safe(x_train, Y, x_test, n_restarts=2, random_state=0,
                         tail_k_local=3, tail_tau_mm=2.0):
    """
    학습구간 내부는 GPR, 바깥(왼쪽/오른쪽 꼬리)은 '국소 선형 + 경계 수축'을 사용.
    커널은 (C*RBF + White + C*DotProduct)로 선형 추세를 보강.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, DotProduct
    except Exception as e:
        raise ImportError("scikit-learn이 필요합니다. `pip install scikit-learn`") from e

    x_train = np.asarray(x_train, float)
    Y = np.asarray(Y, float)
    x_test = np.asarray(x_test, float)

    # 정렬
    order = np.argsort(x_train)
    x_train = x_train[order]
    Y = Y[order]

    X = x_train.reshape(-1,1)

    # 표본이 2개 미만이면 선형(구간) 폴백
    if X.shape[0] < 2:
        return linear_piecewise_predict(x_train, Y, x_test)

    # --- 내부 구간(최소~최대)용 GPR ---
    kernel = (C(1.0, (1e-3, 1e3)) * RBF(length_scale=70.0, length_scale_bounds=(1.0, 300.0))
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
              + C(1.0, (1e-3, 1e3)) * DotProduct())  # 선형추세

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=random_state
    )

    # 다변량 출력 지원이 안 되는 환경 대비
    try:
        gpr.fit(X, Y)
        def gpr_predict(xx):
            return gpr.predict(xx.reshape(-1,1))
    except Exception:
        # 컬럼별 GPR
        def gpr_predict(xx):
            preds = []
            for j in range(Y.shape[1]):
                yj = Y[:, j]
                gpr_j = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=0.0,
                    normalize_y=True,
                    n_restarts_optimizer=n_restarts,
                    random_state=random_state
                )
                gpr_j.fit(X, yj)
                preds.append(gpr_j.predict(xx.reshape(-1,1)))
            return np.stack(preds, axis=1)

    xmin, xmax = x_train[0], x_train[-1]
    K, L = len(x_test), Y.shape[1]
    out = np.zeros((K, L), float)

    # 경계 벡터(240 또는 275에서의 값)
    Y_left_boundary  = Y[0]
    Y_right_boundary = Y[-1]

    # 왼쪽 꼬리(작은 사이즈)용 국소 선형 계수
    kL = min(max(tail_k_local, 2), len(x_train))
    aL, bL = _linear_fit_multi(x_train[:kL], Y[:kL])

    # 오른쪽 꼬리(큰 사이즈)용 국소 선형 계수
    aR, bR = _linear_fit_multi(x_train[-kL:], Y[-kL:])

    for i, st in enumerate(x_test):
        if st < xmin:  # 왼쪽 외삽
            y_lin = _linear_predict_multi(aL, bL, st)
            out[i] = _blend_to_boundary(y_lin, Y_left_boundary, dist_mm=(xmin - st), tau_mm=tail_tau_mm)
        elif st > xmax:  # 오른쪽 외삽
            y_lin = _linear_predict_multi(aR, bR, st)
            out[i] = _blend_to_boundary(y_lin, Y_right_boundary, dist_mm=(st - xmax), tau_mm=tail_tau_mm)
        else:
            out[i] = gpr_predict(np.array([st]))[0]
    return out

# -------- GPR 보간/외삽 유틸 --------
def gpr_fit_predict(x_train, Y, x_test, n_restarts=2, random_state=0):
    """
    x_train: (M,) float sizes
    Y:       (M, L)  각 포인트 인덱스별 타겟(여기서는 dt 또는 dn)
    x_test:  (K,)     예측할 사이즈들
    반환:    (K, L)
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
    except Exception as e:
        raise ImportError("scikit-learn이 필요합니다. `pip install scikit-learn`로 설치하세요.") from e

    X = np.array(x_train, float).reshape(-1, 1)
    Xt = np.array(x_test,  float).reshape(-1, 1)
    Y = np.asarray(Y, float)

    # 표본이 2개 미만이면 GPR 학습이 의미 없으므로 선형 보간/외삽으로 폴백
    if X.shape[0] < 2:
        return linear_piecewise_predict(x_train, Y, x_test)

    # 비교적 완만한 길이 스케일과 노이즈를 가진 기본 커널
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=70.0, length_scale_bounds=(1.0, 300.0)) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=random_state
    )

    # 다변량 Y를 한 번에 처리 (scikit-learn GPR은 (n_samples, n_targets) 지원)
    try:
        gpr.fit(X, Y)
        Y_pred = gpr.predict(Xt)  # (K, L)
        return Y_pred
    except Exception:
        # 환경에 따라 다변량 미지원일 경우 컬럼별로 개별 GPR 수행
        preds = []
        for j in range(Y.shape[1]):
            yj = Y[:, j]
            gpr_j = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.0,
                normalize_y=True,
                n_restarts_optimizer=n_restarts,
                random_state=random_state
            )
            gpr_j.fit(X, yj)
            preds.append(gpr_j.predict(Xt))  # (K,)
        return np.stack(preds, axis=1)  # (K, L)

def linear_piecewise_predict(s_train, Y, s_targets):
    """
    구간선형 보간/외삽 (GPR 폴백).
    s_train: (M,)
    Y:       (M, L)
    s_targets: (K,)
    """
    s_train = np.array(s_train, float)
    Y = np.asarray(Y, float)
    out = np.zeros((len(s_targets), Y.shape[1]), float)
    order = np.argsort(s_train)
    s_train = s_train[order]
    Y = Y[order]

    for i, st in enumerate(s_targets):
        if st <= s_train[0]:
            a, b = 0, min(1, len(s_train)-1)
        elif st >= s_train[-1]:
            a, b = max(0, len(s_train)-2), len(s_train)-1
        else:
            idx = np.searchsorted(s_train, st)
            a, b = idx-1, idx
        denom = (s_train[b]-s_train[a]) if b!=a else 1.0
        t = (st - s_train[a]) / (denom + 1e-12)
        out[i] = (1-t)*Y[a] + t*Y[b]
    return out

# ----------------------------- 메인 -----------------------------
def main():
    # 0) 로드
    rows = load_train_rows(TRAIN_CSV)   # [(size, P)]
    P_new = load_new230_any(NEW230_CSV) # (N0,2)

    # 1) 트랙 선택 (베이스 = 학습 CSV의 최솟값 사이즈)
    track, base = select_track(rows, P_new, min_keep=230, max_keep=290)
    sizes_train = [s for s,_ in track]
    Ps = [P for _,P in track]
    if not Ps:
        raise RuntimeError("선택된 트랙에서 230~290 구간에 해당하는 학습 샘플이 없습니다.")

    # 2) new_230 길이에 맞춰 동일 포인트 수로
    L = len(P_new)
    Ps = [chordlen_resample(P, L) for P in Ps]
    base = chordlen_resample(base, L)
    P_new = chordlen_resample(P_new, L)

    # 3) 각 P_s를 base에 정렬
    aligned = []
    for s, P in zip(sizes_train, Ps):
        Q_best, _, _, _ = cyclic_align(base, P)
        aligned.append((s, Q_best))
    sizes_train = [s for s,_ in aligned]
    Ps = [P for _,P in aligned]

    # 4) 변형장(dt,dn) 구축 (base 기준)
    T, Nvec = tangents_normals(base)
    Ydt, Ydn = [], []
    for P in Ps:
        d = P - base
        # 접선/법선 투영량
        Ydt.append((d*T).sum(axis=1))      # (L,)
        Ydn.append((d*Nvec).sum(axis=1))   # (L,)
    Ydt = np.stack(Ydt, axis=0)  # (M, L)
    Ydn = np.stack(Ydn, axis=0)  # (M, L)

    # 5) GPR로 dt(s), dn(s) 예측 → 목표 사이즈(230..290, 5mm 간격)
    sizes_target = np.arange(230, 295, 5, dtype=int)  # 230..290
    # 외삽 안전한 예측(230,235가 더 커지는 문제 방지)
    dt_pred = gpr_fit_predict_safe(
        sizes_train, Ydt, sizes_target.astype(float),
        n_restarts=50, random_state=0,
        tail_k_local=3,   # 240,245,250 등을 사용해 왼쪽기울기 산출
        tail_tau_mm=1.0   # 경계(240)로 수축하는 길이 스케일(mm). 더 작으면 더 강하게 수축
    )
    dn_pred = gpr_fit_predict_safe(
        sizes_train, Ydn, sizes_target.astype(float),
        n_restarts=50, random_state=0,
        tail_k_local=3,
        tail_tau_mm=0.1
    )
    sizes_target = np.arange(230, 295, 5, dtype=int)  # 230..290
    Tn, Nn = tangents_normals(P_new)

    # 1) 먼저 모든 사이즈의 예측 형상을 모은다
    pred_shapes = []
    for i, s in enumerate(sizes_target):
        P = P_new + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None]
        pred_shapes.append(P)

    # 2) 사이즈 상한(+단조) 가드 적용 (PCA 기반)
    pred_shapes_adj, L_before, L_after = enforce_size_caps_monotone(
        pred_shapes, sizes_target.tolist(), tol_mm=0.0
    )

    # 3) 저장
    with open(SAVE_PRED, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for s, P in zip(sizes_target, pred_shapes_adj):
            row = [int(s)] + [f"{v:.6f}" for v in P.reshape(-1)]
            w.writerow(row)

    print(f"[OK] saved → {SAVE_PRED}")
    print("[INFO] PC1 lengths before/after cap (mm):")
    print("       sizes :", sizes_target.tolist())
    print("       before:", [round(x,3) for x in L_before.tolist()])
    print("       after :", [round(x,3) for x in L_after.tolist()])
    # # 6) new_230의 접선/법선에 합성하여 좌표 생성 & 저장
    # Tn, Nn = tangents_normals(P_new)
    # with open(SAVE_PRED, "w", encoding="utf-8", newline="") as f:
    #     w = csv.writer(f)
    #     for i, s in enumerate(sizes_target):
    #         # (L,) → (L,1)로 브로드캐스트
    #         P = P_new + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None]
    #         row = [int(s)] + [f"{v:.6f}" for v in P.reshape(-1)]
    #         w.writerow(row)
    # print(f"[OK] saved → {SAVE_PRED}")

    # 디버그: 선택된 트랙 정보 및 학습 범위 표시
    tr_min, tr_max = (min(sizes_train), max(sizes_train)) if sizes_train else (None, None)
    ext_note = ""
    if tr_min is not None and (230 < tr_min or 290 > tr_max):
        ext_note = " (경고: 일부 타겟 사이즈는 학습 범위를 벗어나 외삽입니다)"
    print(f"[INFO] chosen train sizes: {sizes_train[:10]}{'...' if len(sizes_train)>10 else ''}")
    print(f"[INFO] GPR train range: [{tr_min}, {tr_max}] → predict 230..290{ext_note}")

if __name__ == "__main__":
    main()
