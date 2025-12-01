#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_series_type_aware_krr_combined_modified.py

- 특정 Type과 Size (TEST_TYPE, TEST_SIZE_MM)를 홀드아웃하여 P_new로 사용.
- 학습 데이터(TEST_TYPE 제외)에서 P_new와 가장 유사한 Type을 선택.
- 선택된 Type의 변형 트랙을 KRR로 학습하여 P_new에 합성.
"""

import os, re, csv
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

# --------------------  경로 및 설정  --------------------
Data_DIR = "Fin_Excel_Data1"

#  테스트에 사용할 타입(hold-out 타입)과 사이즈
TEST_TYPE = "Type07" # 예측 시리즈의 기준 타입 (홀드아웃할 타입)
TEST_SIZE_MM = 250 # 예측 시리즈의 기준 사이즈 (P_new로 사용)

#  TRAIN / NEW 모두 동일한 마스터 CSV 사용 (요청 사항 반영)
MASTER_CSV = os.path.join(
    Data_DIR,
    "control_points_master_L_20251118.csv"
)
TRAIN_CSV = MASTER_CSV
NEW230_CSV = MASTER_CSV  # P_new 로드에 사용 (경로만 동일)

#  예측 결과 파일 이름에 테스트 타입이 들어가도록 (요청 사항 반영)
SAVE_PRED = os.path.join(
    Data_DIR,
    f"pred_Data_230_280_KRR_{TEST_TYPE}.csv"
)
# --------------------------------------------------


# ---------- 파일 로드 유틸리티 ----------

def _read_text(path, encodings=("utf-8-sig","utf-8","cp949","euc-kr","latin-1")):
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                # 파일 내용에 포함된 \ua0 문자를 일반 공백으로 치환하여 반환
                return f.read().replace(r'\ua0', ' ')
        except Exception as e:
            last_err = e
            continue
    raise last_err

_NUM = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][\+\-]?\d+)?$')
def _num_tokens(line):
    # 라인 내에서 \ua0을 포함한 모든 공백 문자가 re.split에 의해 처리되도록 합니다.
    toks = [t for t in line.replace(",", " ").replace(";", " ").split() if t]
    out = []
    for t in toks:
        t = t.strip().lstrip("\ufeff")
        if _NUM.match(t):
            out.append(float(t))
    return out

def parse_side_from_filename(path, default="N/A"):
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    base_lower = base.lower()
    if "_l_" in base_lower or base_lower.endswith("_l"): return "L"
    if "_r_" in base_lower or base_lower.endswith("_r"): return "R"
    tokens = re.findall(r'[a-zA-Z]+', base)
    for t in tokens:
        if t.upper() == 'L': return 'L'
        if t.upper() == 'R': return 'R'
    return default


# --- ★ 홀드아웃 기능이 추가된 로더 함수 ★ ---
def load_train_and_new_from_same_csv(path, test_type, test_size):
    """
    path 하나에서 TEST_TYPE의 TEST_SIZE 행을 P_new로 분리하고,
    TEST_TYPE의 모든 데이터를 학습용(all_rows)에서 제외한다 (홀드아웃).
    """
    text = _read_text(path)
    all_rows = [] # 학습 데이터 (TEST_TYPE 제외)
    new_rows = [] # P_new 데이터
    header_skipped = False
    
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
            
        if not header_skipped:
            ln_lower = ln.lower()
            if "size" in ln_lower and "x1" in ln_lower and ("type" in ln_lower or "Type" in ln):
                header_skipped = True
                continue
        
        # 공백과 쉼표를 모두 구분자로 사용하여 토큰화
        toks = [t.strip() for t in re.split(r'[,\s]+', ln) if t.strip()]
        if len(toks) < 5: continue

        try:
            type_str = toks[0]
            side_str = toks[1]
            size = int(round(float(toks[2])))
            xy_vals_str = toks[3:]
            xy = np.array([float(v) for v in xy_vals_str if _NUM.match(v)], float)
            if len(xy) < 4 or len(xy) % 2 != 0: continue
            P = xy.reshape(-1, 2)
        except Exception:
            continue
            
        # ★ 홀드아웃 로직: TEST_TYPE, TEST_SIZE 행만 new_rows로 분리
        if type_str == test_type and size == test_size:
            if not new_rows: # 첫 번째 발견된 행만 P_new로 사용
                new_rows.append((P, side_str))
        # ★ TEST_TYPE과 동일한 모든 행을 학습 데이터에서 제외
        elif type_str != test_type:
            all_rows.append((type_str, size, P))
            
    if not all_rows:
        raise RuntimeError(f"No train rows left after excluding type='{test_type}'.")
    if not new_rows:
        raise RuntimeError(f"No row found for test_type='{test_type}', size={test_size}.")

    P_new_raw, side_new = new_rows[0]
    print(f"[INFO] Held-out P_new: Type='{test_type}', Size={test_size}, Side='{side_new}'")
    return all_rows, P_new_raw, side_new


# ---------------- 기하 유틸 -----------------

def chordlen_resample(P, n):
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
    return float(np.sum((P-Q)**2))

def cyclic_align(P, Q):
    n = len(P)
    best = (None, 10**30, 0, False)
    for rev in [False, True]:
        R = Q[::-1].copy() if rev else Q.copy()
        for k in range(n):
            Rk = np.roll(R, -k, axis=0)
            sc = _align_score(P, Rk)
            if sc < best[1]:
                best = (Rk, sc, k, rev)
    return best

def tangents_normals(P):
    N = len(P)
    T = np.zeros_like(P)
    if N >= 2:
        T[1:-1] = P[2:] - P[:-2]
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    denom = np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    T = T / denom
    Nvec = np.stack([-T[:,1], T[:,0]], axis=1)
    return T, Nvec

def pca_major_axis(P):
    C = P - P.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    v1 = Vt[0]; v2 = Vt[1]
    z1 = (P @ v1)
    heel_idx = int(np.argmin(z1))
    L1 = float(z1.max() - z1.min())
    z2 = (P @ v2)
    L2 = float(z2.max() - z2.min())
    return v1, v2, heel_idx, L1,L2

def shrink_along_pc1(P, target_L, eps=1e-9):
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
    alpha = target_L / L_current
    r1_new = r1 * alpha
    P_new = heel + np.outer(r1_new, v1) + np.outer(r2, v2)
    return P_new
    
def shrink_along_pc2(P, target_L2, eps=1e-9):
    v1, v2, _, L1, L2 = pca_major_axis(P)
    
    if L2 <= target_L2 + eps:
        return P
    
    center = P.mean(axis=0)
    R = P - center
    r1 = R @ v1
    r2 = R @ v2
    
    L2_current = L2
    if L2_current < eps:
        return P
    
    alpha_2 = target_L2 / L2_current
    r2_new = r2 * alpha_2 
    
    P_new = center + np.outer(r1, v1) + np.outer(r2_new, v2)
    return P_new
    
# -------- 트랙 선택 (베이스 = 학습 CSV의 최소 사이즈) --------
def find_best_track_and_base(all_rows, P_new_resampled):
    tracks_by_type = {}
    for type_str, size, P in all_rows:
        if type_str not in tracks_by_type:
            tracks_by_type[type_str] = []
        tracks_by_type[type_str].append((size, P))

    if not tracks_by_type: raise RuntimeError("No tracks found after grouping by Type.")

    L = len(P_new_resampled)
    best_match = (None, 10**30, None) 

    print(f"Finding best match for new sample among {len(tracks_by_type)} Types...")
    
    for type_str, track_list in tracks_by_type.items():
        if not track_list: continue
        
        min_size_row = min(track_list, key=lambda x: x[0])
        base_P_for_type = min_size_row[1]
        
        base_P_resampled = chordlen_resample(base_P_for_type, L)
        _, sc, _, _ = cyclic_align(P_new_resampled, base_P_resampled)
        
        print(f"  - Type '{type_str}' (base size {min_size_row[0]}): Score = {sc:.2f}")
        
        if sc < best_match[1]:
            best_match = (type_str, sc, track_list)

    best_type, best_score, best_track_list = best_match
    if best_type is None: raise RuntimeError("Failed to find any matching type track.")

    print(f"==> Best Match Found: Type '{best_type}' (Score: {best_score:.2f})")

    base_row = min(best_track_list, key=lambda x: x[0])
    base_P = base_row[1]
    
    track_filtered = [(s, P) for s, P in best_track_list if 230 <= s <= 290]
    if not track_filtered:
        print(f"Warning: No samples in 230-290 range for '{best_type}'. Using all {len(best_track_list)} samples.")
        track_filtered = best_track_list
        if not track_filtered: raise RuntimeError(f"Type '{best_type}' was selected but contains no data.")

    track_filtered.sort(key=lambda x: x[0])
    return track_filtered, base_P, best_type

def enforce_size_caps_monotone_dual(P_list, sizes, tol_L1=1.0, tol_L2=1.0):
    n = len(P_list)
    L1_pred, L2_pred = [], []
    
    for P in P_list:
        _, _, _, L1, L2 = pca_major_axis(P)
        L1_pred.append(L1)
        L2_pred.append(L2)
        
    L1_pred = np.array(L1_pred, float)
    L2_pred = np.array(L2_pred, float)

    # --- 1. PC1 (길이) 보정 ---
    U1 = np.array(sizes, float) + float(tol_L1)
    L1_cap = np.minimum(L1_pred, U1)
    L1_adj = L1_cap.copy()
    for i in range(n-2, -1, -1):
        L1_adj[i] = min(L1_adj[i], L1_adj[i+1])
        
    # --- 2. PC2 (폭) 보정 ---
    U2 = L2_pred + float(tol_L2) 
    L2_cap = np.minimum(L2_pred, U2)
    
    L2_adj = L2_cap.copy()
    for i in range(n-2, -1, -1):
        L2_adj[i] = min(L2_adj[i], L2_adj[i+1])

    # --- 3. 형상 축소 적용 ---
    P_adj_list = []
    for P, L1p, L1a, L2p, L2a in zip(P_list, L1_pred, L1_adj, L2_pred, L2_adj):
        P_temp = P.copy()
        
        if L1p > L1a + 1e-9:
            P_temp = shrink_along_pc1(P_temp, L1a)
        
        if L2p > L2a + 1e-9:
            P_temp = shrink_along_pc2(P_temp, L2a)
            
        P_adj_list.append(P_temp)
        
    return P_adj_list, L1_pred, L1_adj, L2_pred, L2_adj

# -------- KRR 회귀 및 외삽 안정화 함수 --------
def _linear_fit_multi(x, Y):
    x = np.asarray(x, float)
    Y = np.asarray(Y, float)
    X = np.stack([x, np.ones_like(x)], axis=1)
    XtX = X.T @ X
    XtX += 1e-12 * np.eye(2)
    beta = np.linalg.inv(XtX) @ (X.T @ Y)
    a, b = beta[0], beta[1]
    return a, b

def _linear_predict_multi(a, b, x):
    return a * float(x) + b

def _blend_to_boundary(Y_linear, Y_boundary, dist_mm, tau_mm=8.0):
    gamma = np.exp(-dist_mm / max(tau_mm, 1e-6))
    return gamma * Y_boundary + (1.0 - gamma) * Y_linear

# -------- KRR (RBF+Linear) 안전 예측 --------
def krr_fit_predict_safe(x_train, Y, x_test,
                         alpha=1e-2, length_scale=20.0,
                         linear_weight=1.0, rbf_weight=1.0,
                         tail_k_local=3, tail_tau_mm=8.0):
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

    def kernel_callable(A, B):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if A.ndim == 1: A = A.reshape(-1, 1)
        if B.ndim == 1: B = B.reshape(-1, 1)

        K = 0.0
        if rbf_weight != 0.0:
            K += float(rbf_weight) * rbf_kernel(A, B, gamma=gamma)
        if linear_weight != 0.0:
            K += float(linear_weight) * linear_kernel(A, B)
        return K

    if X.shape[0] < 2:
        return linear_piecewise_predict(x_train, Y, x_test)

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

    Y_left_boundary = Y[0]
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


def main():
    # 0) 로드: TEST_TYPE, TEST_SIZE_MM을 홀드아웃하여 P_new와 학습 데이터(all_rows) 분리
    try:
        all_rows, P_new_raw, side_str = load_train_and_new_from_same_csv(
            TRAIN_CSV,
            test_type=TEST_TYPE,
            test_size=TEST_SIZE_MM,
        )
    except RuntimeError as e:
        print(f"\n[FATAL ERROR] Data loading failed: {e}")
        print(f"Please check if '{TEST_TYPE}' at size {TEST_SIZE_MM} exists and other types are available for training.")
        return

    # 1) 'Type' 기반 트랙 선택
    L = len(P_new_raw)
    P_new_resampled = chordlen_resample(P_new_raw, L)
    
    track, base, best_type = find_best_track_and_base(all_rows, P_new_resampled)

    sizes_train = [s for s,_ in track]
    Ps = [P for _,P in track]

    # 2) 동일 포인트 수로 리샘플 및 Y축 정렬
    Ps = [chordlen_resample(P, L) for P in Ps] 
    base = chordlen_resample(base, L)
    P_new = chordlen_resample(P_new_raw, L)

    all_P = [P_new, base] + Ps
    min_y = min(P.min(axis=0)[1] for P in all_P)
    
    if min_y < 0.0: 
        dy = -min_y
        P_new[:, 1] += dy
        base[:, 1] += dy
        for P in Ps: P[:, 1] += dy
        print(f"[INFO] Y-axis adjusted by {dy:.3f} to align minimum Y coordinate to 0.")

    # 3) base 정렬
    aligned = []
    for s, P in zip(sizes_train, Ps):
        Q_best, _, _, _ = cyclic_align(base, P)
        aligned.append((s, Q_best))
    sizes_train = [s for s,_ in aligned]
    Ps = [P for _,P in aligned]

    # 4) 변형장(dt,dn)
    T_base, Nvec_base = tangents_normals(base) # base 기준의 T, Nvec 사용
    Ydt, Ydn = [], []
    for P in Ps:
        d = P - base
        Ydt.append((d*T_base).sum(axis=1))
        Ydn.append((d*Nvec_base).sum(axis=1))
    Ydt = np.stack(Ydt, axis=0)
    Ydn = np.stack(Ydn, axis=0)

    # 5) KRR로 dt(s), dn(s) 예측
    sizes_target = np.arange(230, 295, 5, dtype=int)
    DT_TAU = 1.0 
    DN_TAU = 0.1 
    
    try:
        dt_pred = krr_fit_predict_safe(sizes_train, Ydt, sizes_target.astype(float), alpha=1e-2, length_scale=20.0, linear_weight=1.0, rbf_weight=1.0, tail_k_local=3, tail_tau_mm=DT_TAU)
        dn_pred = krr_fit_predict_safe(sizes_train, Ydn, sizes_target.astype(float), alpha=1e-2, length_scale=20.0, linear_weight=1.0, rbf_weight=1.0, tail_k_local=3, tail_tau_mm=DN_TAU)
    except ImportError:
        print("\n[ERROR] scikit-learn이 필요합니다. `pip install scikit-learn`")
        return

    # 6) 합성 및 길이 가드
    Tn, Nn = tangents_normals(P_new) # P_new 기준의 T, Nvec에 예측된 변위량을 합성

    pred_shapes = []
    for i, s in enumerate(sizes_target):
        P = P_new + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None]
        pred_shapes.append(P)

    pred_shapes_adj, L1_before, L1_after, L2_before, L2_after = enforce_size_caps_monotone_dual(
        pred_shapes, sizes_target.tolist(), tol_L1=1.0, tol_L2=1.0 
    )

    # 7) 파일 저장 (홀드아웃된 Type 사용)
    with open(SAVE_PRED, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        
        num_points = P_new.shape[0] 
        header = ["Type", "side", "size"]
        for i in range(1, num_points + 1):
            header.append(f"x{i}")
            header.append(f"y{i}")
        w.writerow(header)
        
        for s, P in zip(sizes_target, pred_shapes_adj):
            row = [
                TEST_TYPE, 
                side_str, 
                int(s)
            ] + [f"{v:.6f}" for v in P.reshape(-1)]
            w.writerow(row)

    print(f"\n[OK] Saved predictions to -> {SAVE_PRED}")
    print(f"[INFO] Prediction Target Type (Held-out): '{TEST_TYPE}'")
    print(f"[INFO] Best matching Type from training data: '{best_type}'")

    print("[INFO] PC1 Lengths before/after cap (mm):")
    print("       sizes :", sizes_target.tolist())
    print("       before:", [round(x,3) for x in L1_before.tolist()])
    print("       after :", [round(x,3) for x in L1_after.tolist()])
    print("[INFO] PC2 Widths before/after cap (mm):")
    print("       sizes :", sizes_target.tolist())
    print("       before:", [round(x,3) for x in L2_before.tolist()])
    print("       after :", [round(x,3) for x in L2_after.tolist()])

    tr_min, tr_max = (min(sizes_train), max(sizes_train)) if sizes_train else (None, None)
    ext_note = " (Warning: Some targets were extrapolated)" if tr_min is not None and (230 < tr_min or 290 > tr_max) else ""
    print(f"[INFO] Chosen train sizes from type '{best_type}': {sizes_train[:10]}{'...' if len(sizes_train)>10 else ''}")
    print(f"[INFO] KRR train range: [{tr_min}, {tr_max}] -> predict 230..290{ext_note}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")