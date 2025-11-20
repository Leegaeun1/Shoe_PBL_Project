#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_series_type_aware_gpr.py

- [MODIFIED] 'Type'을 고려하여 GPR 모델을 선택합니다.
- 학습 CSV 안에 여러 'Type'이 섞여 있을 때,
  new_230 형상과 가장 가까운 'Type'을 자동 선택.
- 선택된 Type의 '최소 사이즈'를 베이스로 삼아,
  해당 Type의 dt(s), dn(s)를 GPR로 학습.
- 이 GPR 모델을 new_230 형상에 합성하여 시리즈 예측.

"""

import os, re, csv
import numpy as np

# -------------------- 경로 설정 --------------------
# [MODIFIED] Use the uploaded file name
Data_DIR = "Fin_Excel_Data1" 

TRAIN_CSV =os.path.join(
    Data_DIR,
    "control_points_master_L_20251104.csv"
)

NEW230_CSV =os.path.join(
    Data_DIR,
    "control_points_master_test_Q.csv"
)

SAVE_PRED =os.path.join(
    Data_DIR,
    "pred_Data_230_280_ge.csv"
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
    # This helper is now only used for NEW230_CSV
    toks = [t for t in line.replace(",", " ").replace(";", " ").split() if t]
    out = []
    for t in toks:
        t = t.strip().lstrip("\ufeff")
        if _NUM.match(t):
            out.append(float(t))
    return out

# ★★★ [MODIFIED] 'Type' 열을 읽도록 수정된 로더 ★★★
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
            # 'size', 'type', 'x1'이 모두 헤더에 있는지 확인
            ln_lower = ln.lower()
            if "size" in ln_lower and "x1" in ln_lower and ("type" in ln_lower or "Type" in ln):
                header_skipped = True
                continue
        
        # ★★★ [수정] 콤마(,)와 공백( )을 모두 구분자로 처리 (CSV의 ", " 형식 대응)
        toks = [t.strip() for t in re.split(r'[,\s]+', ln) if t.strip()]

        # [type, side, size, x1, y1, ...] 최소 5개 토큰 필요
        if len(toks) < 5: 
            continue 
        
        try:
            # ★★★ [수정] CSV 컬럼 순서 변경 ★★★
            # toks[0] = type
            # toks[1] = side
            # toks[2] = size
            # toks[3:] = coords
            
            type_str = toks[0]
            # side_str = toks[1] # (GPR 학습에 사용하지 않으므로 무시)
            
            # size가 숫자인지 확인
            if not _NUM.match(toks[2]):
                print(f"Warning: Skipping row, 'size' column (toks[2]) is not numeric: {toks[2]}")
                continue
                
            size = int(round(float(toks[2]))) # Col 3: size
            
            # 좌표값 (toks[3]부터 끝까지)
            xy_vals_str = toks[3:]
            
            # 숫자만 float으로 변환
            xy = np.array([float(v) for v in xy_vals_str if _NUM.match(v)], float)
            
            if len(xy) < 4 or len(xy) % 2 != 0:
                print(f"Warning: Skipping row, invalid coord count: {len(xy)}")
                continue # 유효한 좌표가 아님
                
            P = xy.reshape(-1, 2)
            rows.append((type_str, size, P))
            
        except Exception as e:
            print(f"Warning: Skipping malformed row: {ln[:50]}... | Error: {e}")
            continue
            
    if not rows:
        raise RuntimeError("No valid (Type, size, P) train rows found in the training CSV.")
    return rows



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
    text = _read_text(path)
    first = None
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): 
            continue
        vals = _num_tokens(ln) # 숫자만 모두 가져옴
        if len(vals) >= 2:
            first = np.array(vals, float)
            break
    if first is None:
        raise RuntimeError(f"No numeric row found in {path}")

    # 맨 앞 값이 사이즈(mm)면 드랍 (230)
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

# ★★★ [MODIFIED] 'Type'을 기준으로 트랙을 선택하는 함수 ★★★
def find_best_track_and_base(all_rows, P_new_resampled):
    """
    all_rows: [(type, size, P)]
    P_new_resampled: (L, 2) 새로운 230 샘플
    
    1) 'Type'별로 데이터를 그룹화
    2) 각 Type의 '최소 사이즈' 형상을 'Type 베이스'로 사용
    3) 'Type 베이스'와 'P_new'를 비교하여 가장 유사한 Type을 선택
    4) 해당 Type의 전체 트랙과 GPR 기준 '베이스'를 반환
    
    반환: (track, base, best_type_str)
     - track: [(s, P)] (선택된 타입, 230~280 범위)
     - base: (N, 2) (선택된 타입의 최소 사이즈 원본 형상)
     - best_type_str: "Type00" 등
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
        
        # 2) 이 Type의 '베이스' (최소 사이즈) 찾기
        min_size_row = min(track_list, key=lambda x: x[0])
        base_P_for_type = min_size_row[1]
        
        # 3) 비교를 위해 리샘플링 및 정렬
        base_P_resampled = chordlen_resample(base_P_for_type, L)
        _, sc, _, _ = cyclic_align(P_new_resampled, base_P_resampled)
        
        print(f"  - Type '{type_str}' (base size {min_size_row[0]}): Score = {sc:.2f}")
        
        if sc < best_match[1]:
            best_match = (type_str, sc, track_list)

    best_type, best_score, best_track_list = best_match
    if best_type is None:
        raise RuntimeError("Failed to find any matching type track.")

    print(f"==> Best Match Found: Type '{best_type}' (Score: {best_score:.2f})")

    # GPR을 위한 베이스 확정 (선택된 타입의 최소 사이즈, 원본)
    base_row = min(best_track_list, key=lambda x: x[0])
    base_P = base_row[1]
    
    # 학습에 사용할 트랙 데이터 필터링 (원본 230-290, 새 요청 230-280)
    # GPR은 외삽(extrapolation)이 필요할 수 있으므로, 해당 Type의 *모든* 데이터를 사용하는 것이 좋음
    # (원본 스크립트의 230-290 필터링을 유지하되, 경고 메시지 추가)
    track_filtered = [(s, P) for s, P in best_track_list if 230 <= s <= 290]
    
    if not track_filtered:
        print(f"Warning: No samples in 230-290 range for '{best_type}'. Using all {len(best_track_list)} samples for this type.")
        track_filtered = best_track_list
        if not track_filtered:
             raise RuntimeError(f"Type '{best_type}' was selected but contains no data.")

    track_filtered.sort(key=lambda x: x[0])
    return track_filtered, base_P, best_type


# -------- GPR 및 예측 함수 (원본과 동일) --------
def pca_major_axis(P):
    """
    P: (N,2)
    반환: v1(PC1 단위벡터), v2(PC2 단위벡터), heel_idx(PC1 최소 투영점 인덱스), L(PC1 길이)
    """
    C = P - P.mean(axis=0, keepdims=True)
    # SVD로 직교기저 획득 (Vt의 첫 행이 PC1)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    v1 = Vt[0]      # (2,)
    v2 = Vt[1]      # (2,)
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
    X = np.stack([x, np.ones_like(x)], axis=1)        # (k,2)
    XtX = X.T @ X                                         # (2,2)
    # 안정성용 작은 정규화
    XtX += 1e-12 * np.eye(2)
    beta = np.linalg.inv(XtX) @ (X.T @ Y)             # (2,L)
    a, b = beta[0], beta[1]                           # 각 (L,)
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

# ----------------------------- 메인 -----------------------------
def main():
    # 0) 로드
    # [MODIFIED] (type, size, P) 로드
    all_rows = load_train_rows(TRAIN_CSV)
    P_new = load_new230_any(NEW230_CSV) # (N0,2)

    # 1) [MODIFIED] 'Type' 기반 트랙 선택
    # P_new와 동일한 포인트 수 L
    L = len(P_new)
    P_new_resampled = chordlen_resample(P_new, L) # P_new도 리샘플링 (정확한 비교 위함)
    
    # P_new와 가장 유사한 Type의 track과 base를 찾음
    track, base, best_type = find_best_track_and_base(all_rows, P_new_resampled)
    
    sizes_train = [s for s, _ in track]
    Ps = [P for _, P in track]
    
    # 2) new_230(L) 길이에 맞춰 동일 포인트 수로
    Ps = [chordlen_resample(P, L) for P in Ps]
    base = chordlen_resample(base, L)
    P_new = P_new_resampled # 이미 L 길이로 리샘플링됨

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
        Ydt.append((d*T).sum(axis=1))
        Ydn.append((d*Nvec).sum(axis=1))
    Ydt = np.stack(Ydt, axis=0)
    Ydn = np.stack(Ydn, axis=0)

    # 5) GPR로 dt(s), dn(s) 예측
    # [MODIFIED] 230~280까지 (arange 285)
    sizes_target = np.arange(230, 285, 5, dtype=int) # 230..280
    
    dt_pred = gpr_fit_predict_safe(
        sizes_train, Ydt, sizes_target.astype(float),
        n_restarts=50, random_state=0,
        tail_k_local=3, tail_tau_mm=1.0
    )
    dn_pred = gpr_fit_predict_safe(
        sizes_train, Ydn, sizes_target.astype(float),
        n_restarts=50, random_state=0,
        tail_k_local=3, tail_tau_mm=0.1
    )

    # 6) [MODIFIED] new_230의 접선/법선에 합성
    Tn, Nn = tangents_normals(P_new) # P_new 사용

    # 7) [MODIFIED] 사이즈 상한 적용
    pred_shapes = []
    for i, s in enumerate(sizes_target):
        P = P_new + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None]
        pred_shapes.append(P)

    pred_shapes_adj, L_before, L_after = enforce_size_caps_monotone(
        pred_shapes, sizes_target.tolist(), tol_mm=0.0
    )

    # 8) [MODIFIED] 헤더 및 Type/Side 포함하여 저장
    
    # ★★★ 'side' 추론 (TRAIN_CSV 파일명 기반) ★★★
    # (또는 NEW230_CSV에서 추론하려면 NEW230_CSV를 인자로 넣으세요)
    side_str = parse_side_from_filename(TRAIN_CSV, default="N/A")
    print(f"[INFO] Inferred side='{side_str}' from training data path '{TRAIN_CSV}'")

    with open(SAVE_PRED, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        
        # ★★★ 1. 헤더 생성 및 저장 ★★★
        # P_new (L, 2)의 L (포인트 수, 예: 40)을 가져옴
        num_points = P_new.shape[0] 
        header = ["Type", "side", "size"]
        for i in range(1, num_points + 1):
            header.append(f"x{i}")
            header.append(f"y{i}")
        
        w.writerow(header) # CSV 파일에 헤더 쓰기
        
        # ★★★ 2. 데이터 행 저장 ★★★
        for s, P in zip(sizes_target, pred_shapes_adj):
            # 행 구성: [Type, side, size, x1, y1, x2, y2, ...]
            row = [
                best_type,  # (예: "Type05")
                side_str,   # (예: "L")
                int(s)      # (예: 230)
            ] + [f"{v:.6f}" for v in P.reshape(-1)] # 좌표값
            
            w.writerow(row)

    print(f"\n[OK] Saved predictions to -> {SAVE_PRED}")
    print(f"[INFO] Based on new sample's match with Type: '{best_type}'")
    # (이하 기존 print문 동일)
    print("[INFO] PC1 lengths before/after cap (mm):")
    print("      sizes :", sizes_target.tolist())
    print("      before:", [round(x,3) for x in L_before.tolist()])
    print("      after :", [round(x,3) for x in L_after.tolist()])

    tr_min, tr_max = (min(sizes_train), max(sizes_train)) if sizes_train else (None, None)
    ext_note = ""
    if tr_min is not None and (230 < tr_min or 280 > tr_max):
        ext_note = " (Warning: Some targets were extrapolated)"
    print(f"[INFO] Chosen train sizes from type '{best_type}': {sizes_train[:10]}{'...' if len(sizes_train)>10 else ''}")
    print(f"[INFO] GPR train range: [{tr_min}, {tr_max}] -> predict 230..280{ext_note}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Failed to run prediction: {e}")
        import traceback
        traceback.print_exc()