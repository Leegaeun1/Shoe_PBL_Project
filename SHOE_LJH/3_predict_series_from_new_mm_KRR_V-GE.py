#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_series_from_new_230_krr_trackselect.py

- 학습 CSV의 '최소 사이즈' 형상을 베이스로 트랙 자동 선택
- 베이스 대비 (접선 dt, 법선 dn) 변형량을 학습
- 사이즈→변형량 관계를 Kernel Ridge Regression(KRR, RBF+Linear 커널)으로 회귀
- 목표 사이즈(230..290, 5mm 간격)를 예측하고 new_230에 합성
- 예측 형상은 PCA(PC1) 길이가 s+1mm를 넘지 않도록 캡 및 단조 보정

입력:
  TRAIN_CSV = "control_points_master_20251013.csv"  # size, x1,y1,...
  NEW_CSV = "new_size.csv"                        # (선택) size, x1,y1,...

출력:
  SAVE_PRED  = "pred_series.csv"                    # size, x1,y1,...

필요:
  numpy, scikit-learn
"""

'''학습데이터(사이즈 몇개) + 새로운 사이즈 하나 => 230~290까지 사이즈 생성'''
import os, re, csv
import numpy as np

# -------------------- 경로 설정 --------------------
Data_DIR = "control_points_master_test_Q" 

TRAIN_CSV =os.path.join(
    Data_DIR,
    "control_points_master_L_20251104.csv"
)

NEW_CSV =os.path.join(
    Data_DIR,
    "control_points_master_test_Q.csv"
)

SAVE_PRED =os.path.join(
    Data_DIR,
    "pred_Data_230_280_ge_KRR.csv"
)
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
    '''숫자 토큰만 추출하여 float리스트로 반환. 쉼표나 ;로 구분된거 처리'''
    toks = [t for t in line.replace(",", " ").replace(";", " ").split() if t]
    out = []
    for t in toks:
        t = t.strip().lstrip("\ufeff")
        if _NUM.match(t):
            out.append(float(t))
    return out

def load_train_rows(path):
    """
    학습 CSV에서 (type, size, P) 목록을 읽는다.
    [수정됨] CSV 형식: type, side, size, x1, y1, ...
    """
    text = _read_text(path)
    rows = []
    header_skipped = False
    
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
            
        if not header_skipped: # 헤더 스킵
            ln_lower = ln.lower()
            if "size" in ln_lower and "x1" in ln_lower and ("type" in ln_lower or "Type" in ln):
                header_skipped = True
                continue
        
        # 콤마(,)와 공백( )을 모두 구분자로 처리 (CSV의 ", " 형식 대응)
        toks = [t.strip() for t in re.split(r'[,\s]+', ln) if t.strip()]

        # [type, side, size, x1, y1, ...] 최소 5개 토큰 필요
        if len(toks) < 5: 
            continue 
        
        try:
            # CSV 컬럼 순서: type, side, size, ...
            type_str = toks[0]
            # side_str = toks[1] # (사용 안 함)
            
            if not _NUM.match(toks[2]):
                print(f"Warning: Skipping row, 'size' column (toks[2]) is not numeric: {toks[2]}")
                continue
                
            size = int(round(float(toks[2]))) # Col 3: size
            xy_vals_str = toks[3:] # 좌표값
            
            xy = np.array([float(v) for v in xy_vals_str if _NUM.match(v)], float)
            
            if len(xy) < 4 or len(xy) % 2 != 0:
                print(f"Warning: Skipping row, invalid coord count: {len(xy)}")
                continue 
                
            P = xy.reshape(-1, 2)
            rows.append((type_str, size, P))
            
        except Exception as e:
            print(f"Warning: Skipping malformed row: {ln[:50]}... | Error: {e}")
            continue
            
    if not rows:
        raise RuntimeError("No valid (Type, size, P) train rows found in the training CSV.")
    return rows


# ---------------- 기하 유틸 -----------------
def chordlen_resample(P, n):
    '''주어진 형상 좌표(P)를 n개의 포인트로 리샘플링
       -> 모든 형상이 동일한 수의 제어점 가짐'''
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.array([])
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
    '''P와 Q의 최소제곱오차(MSE) 계산하여 정렬 점수 반환'''
    return float(np.sum((P-Q)**2))

def cyclic_align(P, Q):
    """P와 Q를 순환 시프트로 최적 정렬 (정방향/역방향 모두 시도)
    정렬된 Q, 점수, 시프트량, 뒤집기 여부 반환."""
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
    '''P의 각 제어점에서 접선 벡터(T)와 법선 벡터 계산(N)
    변형량을 접선/법선 방향으로 분해하는데 사용됨'''
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

def pca_major_axis(P):
    '''PCA를 수행하여 P의 PC1방향 벡터, PC2벡터, PC1 방향 길이, 뒤꿈치 인덱스 계산
    길이 제약 및 보정에 사용됨'''
    C = P - P.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    v1 = Vt[0]; v2 = Vt[1]
    # PC1 투영 및 길이
    z1 = (P @ v1)
    heel_idx = int(np.argmin(z1))
    L1 = float(z1.max() - z1.min())
    
    # PC2 투영 및 길이 (폭)
    z2 = (P @ v2) # PC2 방향으로 투영
    L2 = float(z2.max() - z2.min()) # PC2 길이 (폭)
    return v1, v2, heel_idx, L1,L2

def shrink_along_pc1(P, target_L, eps=1e-9):
    '''P의 PC1 길이가 target_L보다 길 경우 뒤꿈치(hell)을 구정점으로 하여 PC1방향으로만 축소
    -> 목표 길이에 맞춤'''
    v1, v2, heel_idx, L1,L2 = pca_major_axis(P)
    if L1 <= target_L + eps:
        return P
    heel = P[heel_idx]
    R = P - heel
    r1 = R @ v1
    r2 = R @ v2
    L_current = float(r1.max() - r1.min())
    if L_current < eps:
        return P
    alpha = target_L / L_current  # 0<alpha<1
    r1_new = r1 * alpha
    P_new = heel + np.outer(r1_new, v1) + np.outer(r2, v2)
    return P_new
def shrink_along_pc2(P, target_L2, eps=1e-9):
    # PC1, PC2 축 및 길이 계산
    v1, v2, _, L1, L2 = pca_major_axis(P) # L2도 받아옴
    
    if L2 <= target_L2 + eps:
        return P
    
    # 중심을 원점으로 이동
    center = P.mean(axis=0)
    R = P - center
    
    # PC1과 PC2 방향 성분 추출
    r1 = R @ v1
    r2 = R @ v2
    
    L2_current = L2
    if L2_current < eps:
        return P
    
    # PC2 방향(r2)만 목표 비율로 축소 (alpha_2 < 1)
    alpha_2 = target_L2 / L2_current
    r2_new = r2 * alpha_2 
    
    # 축소된 성분을 중심으로 다시 합하여 새 형상 생성
    P_new = center + np.outer(r1, v1) + np.outer(r2_new, v2)
    return P_new
# -------- 트랙 선택 (베이스 = 학습 CSV의 최소 사이즈) --------
def find_best_track_and_base(all_rows, P_new_resampled):
    """
    [MODIFIED] 'Type'을 기준으로 트랙을 선택하는 함수
    all_rows: [(type, size, P)]
    P_new_resampled: (L, 2) 새로운 230 샘플
    
    1) 'Type'별로 데이터를 그룹화
    2) 각 Type의 '최소 사이즈' 형상을 'Type 베이스'로 사용
    3) 'Type 베이스'와 'P_new'를 비교하여 가장 유사한 Type을 선택
    4) 해당 Type의 전체 트랙과 GPR 기준 '베이스'를 반환
    """
    tracks_by_type = {}
    for type_str, size, P in all_rows:
        if type_str not in tracks_by_type:
            tracks_by_type[type_str] = []
        tracks_by_type[type_str].append((size, P))

    if not tracks_by_type:
        raise RuntimeError("No tracks found after grouping by Type.")

    L = len(P_new_resampled)
    best_match = (None, 10**30, None) # (best_type, best_score, best_track_list)

    print(f"Finding best match for new sample among {len(tracks_by_type)} Types...")
    
    for type_str, track_list in tracks_by_type.items():
        if not track_list:
            continue
        
        min_size_row = min(track_list, key=lambda x: x[0])
        base_P_for_type = min_size_row[1]
        
        base_P_resampled = chordlen_resample(base_P_for_type, L)
        _, sc, _, _ = cyclic_align(P_new_resampled, base_P_resampled)
        
        print(f"  - Type '{type_str}' (base size {min_size_row[0]}): Score = {sc:.2f}")
        
        if sc < best_match[1]:
            best_match = (type_str, sc, track_list)

    best_type, best_score, best_track_list = best_match
    if best_type is None:
        raise RuntimeError("Failed to find any matching type track.")

    print(f"==> Best Match Found: Type '{best_type}' (Score: {best_score:.2f})")

    base_row = min(best_track_list, key=lambda x: x[0])
    base_P = base_row[1]
    
    # 학습에 사용할 트랙 데이터 필터링 (230-290 범위)
    track_filtered = [(s, P) for s, P in best_track_list if 230 <= s <= 290]
    
    if not track_filtered:
        print(f"Warning: No samples in 230-290 range for '{best_type}'. Using all {len(best_track_list)} samples.")
        track_filtered = best_track_list
        if not track_filtered:
             raise RuntimeError(f"Type '{best_type}' was selected but contains no data.")

    track_filtered.sort(key=lambda x: x[0])
    return track_filtered, base_P, best_type
def enforce_size_caps_monotone(P_list, sizes, tol_mm=1.0):
    '''예측된 형상 길이 제약 적용
    1. 각 사이즈 s의 예측 PC1길이가 s+ tol_mm을 초과하지 않도록 최대 길이 제한
    2. s가 증가함에 따라 PC1 길이가 단조 증가하도록 보정
    3. 이 보정된 길이를 초과하는 현상은 shrink_along_pc1로 축소 보정함'''
    n = len(P_list)
    L_pred = []
    for P in P_list:
        _, _, _, L = pca_major_axis(P)
        L_pred.append(L)
    L_pred = np.array(L_pred, float)

    U = np.array(sizes, float) + float(tol_mm)
    L_cap = np.minimum(L_pred, U)

    # 역방향 단조(오직 감소만 허용)
    L_adj = L_cap.copy()
    for i in range(n-2, -1, -1):
        L_adj[i] = min(L_adj[i], L_adj[i+1])

    P_adj_list = []
    for P, Lp, La in zip(P_list, L_pred, L_adj):
        if Lp <= La + 1e-9:
            P_adj_list.append(P)
        else:
            P_adj_list.append(shrink_along_pc1(P, La))
    return P_adj_list, L_pred, L_adj
def enforce_size_caps_monotone_dual(P_list, sizes, tol_L1=1.0, tol_L2=1.0): # 이름 변경 및 폭 tolerance 추가
    n = len(P_list)
    L1_pred, L2_pred = [], []
    
    for P in P_list:
        _, _, _, L1, L2 = pca_major_axis(P) # L1, L2 모두 받음
        L1_pred.append(L1)
        L2_pred.append(L2)
        
    L1_pred = np.array(L1_pred, float)
    L2_pred = np.array(L2_pred, float)

    # --- 1. PC1 (길이) 보정 ---
    U1 = np.array(sizes, float) + float(tol_L1)
    L1_cap = np.minimum(L1_pred, U1)
    L1_adj = L1_cap.copy()
    # 역방향 단조 (감소만 허용)
    for i in range(n-2, -1, -1):
        L1_adj[i] = min(L1_adj[i], L1_adj[i+1])
        
    # --- 2. PC2 (폭) 보정 ---
    # PC2의 목표 길이 U2: 폭 오차도 1mm 이하로 제한
    # 여기서 목표 사이즈는 보통 길이 기준이므로, 폭의 목표값 U2는 단순한 s+tol이 아닌,
    # 예측된 PC1 길이에 비례하거나, TRAIN data의 L2 경향에 기반한 L2_pred + tol_L2로 제한할 수 있습니다.
    # 단순화를 위해, 여기서는 TRAIN data의 L2 예측값에 절대 오차 tol_L2만 더한 값으로 제한하겠습니다.
    
    # **중요**: L2 예측값 자체는 L2_pred이므로, L2가 's'에 묶이지 않으므로,
    # L2_cap은 L2_pred + tol_L2로 제한하는 것이 합리적입니다.
    U2 = L2_pred + float(tol_L2) 
    L2_cap = np.minimum(L2_pred, U2)
    
    L2_adj = L2_cap.copy()
    # PC2도 단조 증가하도록 강제 (폭도 사이즈에 따라 단조 증가해야 함)
    for i in range(n-2, -1, -1):
        L2_adj[i] = min(L2_adj[i], L2_adj[i+1])

    # --- 3. 형상 축소 적용 ---
    P_adj_list = []
    for P, L1p, L1a, L2p, L2a in zip(P_list, L1_pred, L1_adj, L2_pred, L2_adj):
        P_temp = P.copy()
        
        # 1차: PC1 길이 보정 적용
        if L1p > L1a + 1e-9:
            P_temp = shrink_along_pc1(P_temp, L1a)
            # PC1 축소 후 PC2 길이가 변했을 수 있으나, 여기서는 근사치로 처리함.
        
        # 2차: PC2 길이 보정 적용 (PC1 보정 후 재측정 필요하지만, 단순화를 위해 L2a만 사용)
        if L2p > L2a + 1e-9:
            P_temp = shrink_along_pc2(P_temp, L2a)
            
        P_adj_list.append(P_temp)
        
    return P_adj_list, L1_pred, L1_adj, L2_pred, L2_adj # L2 결과도 반환
# -------- KRR 회귀 및 외삽 안정화 함수 --------
def _linear_fit_multi(x, Y):
    '''다차원 데이터 Y에 대한 x의 선형회귀 
    y = ax + b 계수 a,b를 계산함. KRR의 외삽 안정화에 사용됨'''
    x = np.asarray(x, float)
    Y = np.asarray(Y, float)
    X = np.stack([x, np.ones_like(x)], axis=1)
    XtX = X.T @ X
    XtX += 1e-12 * np.eye(2)
    beta = np.linalg.inv(XtX) @ (X.T @ Y)
    a, b = beta[0], beta[1]
    return a, b

def _linear_predict_multi(a, b, x): # 계산된 선형 계수 사용하여 새로운 x값에 대한 선형 예측 값 계산
    return a * float(x) + b

def _blend_to_boundary(Y_linear, Y_boundary, dist_mm, tau_mm=8.0):
    '''선형 외삽값(Y_linear)과 학습 경계값(Y_boundary)을 거리에 따라 블렌딩하여 외삽 안정화.
    -dist_mm = 0(경계)이면 Y_boundary 반환, dist_mm 클수록 Y_linear에 가까워짐.'''
    gamma = np.exp(-dist_mm / max(tau_mm, 1e-6))  # 1이면 경계값, 0이면 선형외삽
    return gamma * Y_boundary + (1.0 - gamma) * Y_linear

# -------- KRR (RBF+Linear) 안전 예측 --------
def krr_fit_predict_safe(x_train, Y, x_test,
                         alpha=1e-2, length_scale=20.0,
                         linear_weight=1.0, rbf_weight=1.0,
                         tail_k_local=3, tail_tau_mm=8.0):
    '''핵심 학습 및 예측 함수
    1. Kernel Ridge모델을 사용하여 학습 X_train(사이즈)와 Y(변형량) 사이의 관계 모델링함.
    2. 합성 커널(RBF+Linear)을 사용하여 비선형과 선형 추세 모두 포착
    3. 예측 범위가 학습 범위 벗어나면 국소 섷녕 외삽 수행 -> 경계 블랜딩 통해 예측값 안정화
    4. 학습 범위 내에서는 순수 KRR 예측값 사용'''
    try:
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
    except Exception as e:
        raise ImportError("scikit-learn이 필요합니다. `pip install scikit-learn`") from e

    x_train = np.asarray(x_train, float)
    Y = np.asarray(Y, float)
    x_test = np.asarray(x_test, float)

    order = np.argsort(x_train)
    x_train = x_train[order]
    Y = Y[order]

    X = x_train.reshape(-1,1)
    gamma = 1.0 / (2.0 * (float(length_scale)**2) + 1e-12)

    # 합성 커널 팩토리
    def kernel_callable(A, B):
        # --- 모든 입력을 2D(float)로 강제 변환 ---
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        if B.ndim == 1:
            B = B.reshape(-1, 1)

        # --- 합성 커널: RBF + Linear ---
        K = 0.0
        if rbf_weight != 0.0:
            K += float(rbf_weight) * rbf_kernel(A, B, gamma=gamma)
        if linear_weight != 0.0:
            K += float(linear_weight) * linear_kernel(A, B)
        return K


    # 표본이 2개 미만이면 선형 폴백
    if X.shape[0] < 2:
        return linear_piecewise_predict(x_train, Y, x_test)

    # 멀티타깃 지원 시 한 번에, 아니면 컬럼별
    def fit_and_predict(xx):
        try:
            model = KernelRidge(alpha=float(alpha), kernel=kernel_callable)
            model.fit(X, Y)
            return model.predict(xx.reshape(-1,1))
        except Exception:
            preds = []
            for j in range(Y.shape[1]):
                model_j = KernelRidge(alpha=float(alpha), kernel=kernel_callable)
                model_j.fit(X, Y[:, j])
                preds.append(model_j.predict(xx.reshape(-1,1)))
            return np.stack(preds, axis=1)

    xmin, xmax = x_train[0], x_train[-1]
    K, L = len(x_test), Y.shape[1]
    out = np.zeros((K, L), float)

    Y_left_boundary  = Y[0]
    Y_right_boundary = Y[-1]

    kL = min(max(tail_k_local, 2), len(x_train))
    aL, bL = _linear_fit_multi(x_train[:kL], Y[:kL])
    aR, bR = _linear_fit_multi(x_train[-kL:], Y[-kL:])

    for i, st in enumerate(x_test):
        if st < xmin:
            y_lin = _linear_predict_multi(aL, bL, st)
            out[i] = _blend_to_boundary(y_lin, Y_left_boundary, dist_mm=(xmin - st), tau_mm=tail_tau_mm)
        elif st > xmax:
            y_lin = _linear_predict_multi(aR, bR, st)
            out[i] = _blend_to_boundary(y_lin, Y_right_boundary, dist_mm=(st - xmax), tau_mm=tail_tau_mm)
        else:
            out[i] = fit_and_predict(np.array([st]))[0]
    return out

# -------- 선형 보간/외삽 폴백 --------
def linear_piecewise_predict(s_train, Y, s_targets):
    s_train = np.array(s_train, float)
    Y = np.asarray(Y, float)
    out = np.zeros((len(s_targets), Y.shape[1]), float)
    order = np.argsort(s_train)
    s_train = s_train[order]; Y = Y[order]

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



def parse_side_from_filename(path, default="N/A"):
    """
    'control_points_master_L_...csv' 같은 파일명에서 'L' 또는 'R'을 추론합니다.
    """
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    base_lower = base.lower()
    
    # 예: _l_ 또는 _l.csv
    if "_l_" in base_lower or base_lower.endswith("_l"):
        return "L"
    if "_r_" in base_lower or base_lower.endswith("_r"):
        return "R"
    
    # 파일명 토큰에서 'L' 또는 'R' 자체를 찾기 (예: new_230_L.csv)
    tokens = re.findall(r'[a-zA-Z]+', base)
    for t in tokens:
        if t.upper() == 'L':
            return 'L'
        if t.upper() == 'R':
            return 'R'
            
    return default


def load_new230_any(path):  
    '''목표 시리즈의 베이스 형상으로 사용할 new_size.csv 파일에서 좌표를 로드. 
    파일 내용의 맨 앞 값이 사이즈 값이라면 이를 제거하고, 나머지 좌표만 NumPy 배열로 반환'''
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


# ----------------------------- 메인 -----------------------------
def main():
    # 0) 로드
    all_rows = load_train_rows(TRAIN_CSV) # 학습 데이터 (수정된 함수 사용)
    P_new_raw = load_new230_any(NEW_CSV) # 기준 사이즈 데이터

    # 1) [MODIFIED] 'Type' 기반 트랙 선택
    # P_new와 동일한 포인트 수 L
    L = len(P_new_raw)
    P_new_resampled = chordlen_resample(P_new_raw, L) # 비교를 위해 리샘플링
    
    # P_new와 가장 유사한 Type의 track과 base를 찾음
    track, base, best_type = find_best_track_and_base(all_rows, P_new_resampled)

    sizes_train = [s for s,_ in track]
    Ps = [P for _,P in track]
    if not Ps:
        raise RuntimeError("선택된 트랙에서 230~290 구간에 해당하는 학습 샘플이 없습니다.")

    # 2) 동일 포인트 수로 리샘플
    L = len(P_new_raw)
    Ps = [chordlen_resample(P, L) for P in Ps] 
    base = chordlen_resample(base, L)
    P_new = chordlen_resample(P_new_raw, L) # P_new도 L개로 통일

    # ... (기존 2, 3, 4, 5번 항목은 그대로 둠) ...
    # [핵심 수정 1: Y축 원점 정렬]
    # (이하 KRR 스크립트의 기존 로직 유지)
    all_P = [P_new, base] + Ps
    min_y = min(P.min(axis=0)[1] for P in all_P)
    
    if min_y < 0.0: 
        dy = -min_y
        P_new[:, 1] += dy
        base[:, 1] += dy
        for P in Ps:
            P[:, 1] += dy
        print(f"[INFO] Y-axis adjusted by {dy:.3f} to align minimum Y coordinate to 0.")

    P_new_length_before = pca_major_axis(P_new)[-2] # L1
    print(f"[INFO] P_new base length maintained at: {P_new_length_before:.3f}mm")

    # 3) base 정렬
    aligned = []
    for s, P in zip(sizes_train, Ps):
        Q_best, _, _, _ = cyclic_align(base, P)
        aligned.append((s, Q_best))
    sizes_train = [s for s,_ in aligned]
    Ps = [P for _,P in aligned]

    # 4) 변형장(dt,dn)
    T, Nvec = tangents_normals(base)
    Ydt, Ydn = [], []
    for P in Ps:
        d = P - base
        Ydt.append((d*T).sum(axis=1))
        Ydn.append((d*Nvec).sum(axis=1))
    Ydt = np.stack(Ydt, axis=0)
    Ydn = np.stack(Ydn, axis=0)

    # 5) KRR로 dt(s), dn(s) 예측
    sizes_target = np.arange(230, 295, 5, dtype=int)
    DT_TAU = 1.0 
    DN_TAU = 0.1 
    
    dt_pred = krr_fit_predict_safe(
        sizes_train, Ydt, sizes_target.astype(float),
        alpha=1e-2, length_scale=20.0,
        linear_weight=1.0, rbf_weight=1.0,
        tail_k_local=3, tail_tau_mm=DT_TAU
    )
    dn_pred = krr_fit_predict_safe(
        sizes_train, Ydn, sizes_target.astype(float),
        alpha=1e-2, length_scale=20.0,
        linear_weight=1.0, rbf_weight=1.0,
        tail_k_local=3, tail_tau_mm=DN_TAU
    )
    
    # 6) 합성 → 길이 가드 → 저장
    Tn, Nn = tangents_normals(P_new)

    pred_shapes = []
    for i, s in enumerate(sizes_target):
        P = P_new + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None]
        pred_shapes.append(P)

    # (KRR의 고유 로직인 _dual 함수 호출은 그대로 유지)
    pred_shapes_adj, L1_before, L1_after, L2_before, L2_after = enforce_size_caps_monotone_dual(
        pred_shapes, sizes_target.tolist(), tol_L1=1.0, tol_L2=1.0 
    )

    # ★★★ [MODIFIED] 저장 로직 전체를 교체 ★★★
    
    # 'side' 추론 (TRAIN_CSV 파일명 기반)
    side_str = parse_side_from_filename(TRAIN_CSV, default="N/A")
    print(f"[INFO] Inferred side='{side_str}' from training data path '{TRAIN_CSV}'")

    with open(SAVE_PRED, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        
        # 1. 헤더 생성 및 저장
        num_points = P_new.shape[0] # (L)
        header = ["Type", "side", "size"]
        for i in range(1, num_points + 1):
            header.append(f"x{i}")
            header.append(f"y{i}")
        w.writerow(header) # 헤더 쓰기
        
        # 2. 데이터 행 저장
        for s, P in zip(sizes_target, pred_shapes_adj):
            # 행 구성: [Type, side, size, x1, y1, x2, y2, ...]
            row = [
                best_type,  # (예: "Type05")
                side_str,   # (예: "L")
                int(s)      # (예: 230)
            ] + [f"{v:.6f}" for v in P.reshape(-1)] # 좌표값
            
            w.writerow(row)
    # ★★★ (여기까지 교체) ★★★

    print(f"[OK] saved -> {SAVE_PRED}")
    print(f"[INFO] Based on new sample's match with Type: '{best_type}'") # best_type 로깅
    print("[INFO] PC1 Lengths before/after cap (mm):")
    print("       sizes :", sizes_target.tolist())
    print("       before:", [round(x,3) for x in L1_before.tolist()])
    print("       after :", [round(x,3) for x in L1_after.tolist()])
    print("[INFO] PC2 Widths before/after cap (mm):")
    print("       sizes :", sizes_target.tolist())
    print("       before:", [round(x,3) for x in L2_before.tolist()])
    print("       after :", [round(x,3) for x in L2_after.tolist()])

    tr_min, tr_max = (min(sizes_train), max(sizes_train)) if sizes_train else (None, None)
    ext_note = ""
    if tr_min is not None and (230 < tr_min or 290 > tr_max):
        ext_note = " (경고: 일부 타겟 사이즈는 학습 범위를 벗어나 외삽입니다)"
    print(f"[INFO] chosen train sizes from type '{best_type}': {sizes_train[:10]}{'...' if len(sizes_train)>10 else ''}")
    print(f"[INFO] KRR train range: [{tr_min}, {tr_max}] -> predict 230..290{ext_note}")

if __name__ == "__main__":
    main()