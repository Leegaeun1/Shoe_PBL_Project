#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_series_all_types_unified_svr.py

[기능]
1. Data_DIR 내 MASTER_CSV 로드.
2. 존재하는 모든 Type(Type00~Type07 등)을 순회.
3. 각 Type을 Hold-out(테스트용)으로 설정하고, 해당 Type의 '최소 사이즈'를 P_new로 자동 지정.
4. 나머지 Type들 중에서 P_new와 가장 유사한 Type을 'Best Match'로 선정하여 SVR 학습.
5. 230~280mm 구간 예측 및 통합 저장.
6. [NEW] 실행 시간 및 매칭 정보를 별도 요약 파일로 저장.
"""

import os, re, csv
import numpy as np
import time
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

# ================= 설정 =================
Data_DIR = "20251120/CTRL100"
MASTER_CSV_NAME = "control_points_master_L_20251124.csv"

# 1. 통합 결과 저장 파일명
SAVE_FILENAME = "pred_Data_SVR_230_280.csv"
SAVE_PRED_PATH = os.path.join(Data_DIR, SAVE_FILENAME)

# 2. 실행 시간 요약 저장 경로
SAVE_SUMMARY_FILENAME = "svr_runtime_summary.csv"
SAVE_SUMMARY_PATH = os.path.join(Data_DIR, SAVE_SUMMARY_FILENAME)

# 예측할 사이즈 범위 (230부터 280까지, 5단위)
TARGET_SIZES = np.arange(230, 285, 5, dtype=int) 

MASTER_CSV = os.path.join(Data_DIR, MASTER_CSV_NAME)

# ---------- 파일 리더 유틸리티 ----------
def _read_text(path, encodings=("utf-8-sig","utf-8","cp949","euc-kr","latin-1")):
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read().replace(r'\ua0', ' ')
        except Exception as e:
            last_err = e
            continue
    raise last_err

_NUM = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][\+\-]?\d+)?$')

# ---------- 데이터 전체 로드 함수 ----------
def load_full_master_data(path):
    text = _read_text(path)
    data_dict = {}
    header_skipped = False
    
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        if not header_skipped:
            ln_lower = ln.lower()
            if "size" in ln_lower and "x1" in ln_lower:
                header_skipped = True
                continue
        toks = [t.strip() for t in re.split(r"[,\s]+", ln) if t.strip()]
        if len(toks) < 5: continue

        try:
            type_str, side_str = toks[0], toks[1]
            if not _NUM.match(toks[2]): continue
            size = int(round(float(toks[2])))
            xy_vals_str = toks[3:]
            xy = np.array([float(v) for v in xy_vals_str if _NUM.match(v)], float)
            if len(xy) < 4 or len(xy) % 2 != 0: continue
            P = xy.reshape(-1, 2)

            if type_str not in data_dict: data_dict[type_str] = {}
            data_dict[type_str][size] = (P, side_str)
        except Exception: continue
            
    return data_dict

# ---------------- 기하 유틸 -----------------
def chordlen_resample(P, n):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.array([])
    u = np.zeros(len(P))
    if len(P) > 1: u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-9: return np.repeat(P[:1], n, axis=0)
    u /= L
    s = np.linspace(0,1,n,endpoint=True)
    x = np.interp(s, u, P[:,0]); y = np.interp(s, u, P[:,1])
    return np.stack([x,y], axis=1)

def _align_score(P, Q): return float(np.sum((P-Q)**2))

def cyclic_align(P, Q):
    n = len(P)
    best = (None, 10**30, 0, False)
    for rev in [False, True]:
        R = Q[::-1].copy() if rev else Q.copy()
        for k in range(n):
            Rk = np.roll(R, -k, axis=0)
            sc = _align_score(P, Rk)
            if sc < best[1]: best = (Rk, sc, k, rev)
    return best

def tangents_normals(P):
    N = len(P)
    T = np.zeros_like(P)
    if N >= 2:
        T[1:-1] = P[2:] - P[:-2]
        T[0], T[-1] = P[1] - P[0], P[-1] - P[-2]
    denom = np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    T = T / denom
    Nvec = np.stack([-T[:,1], T[:,0]], axis=1)
    return T, Nvec

def pca_major_axis(P):
    C = P - P.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    v1, v2 = Vt[0], Vt[1]
    z1 = (P @ v1)
    heel_idx = int(np.argmin(z1))
    L1 = float(z1.max() - z1.min())
    z2 = (P @ v2)
    L2 = float(z2.max() - z2.min())
    return v1, v2, heel_idx, L1, L2

def shrink_along_pc1(P, target_L, eps=1e-9):
    v1, v2, heel_idx, L1, L2 = pca_major_axis(P)
    if L1 <= target_L + eps: return P
    heel = P[heel_idx]
    R = P - heel
    r1, r2 = R @ v1, R @ v2
    L_current = float(r1.max() - r1.min())
    if L_current < eps: return P
    alpha = target_L / L_current
    P_new = heel + np.outer(r1 * alpha, v1) + np.outer(r2, v2)
    return P_new
    
def shrink_along_pc2(P, target_L2, eps=1e-9):
    v1, v2, _, L1, L2 = pca_major_axis(P)
    if L2 <= target_L2 + eps: return P
    center = P.mean(axis=0)
    R = P - center
    r1, r2 = R @ v1, R @ v2
    L2_current = L2
    if L2_current < eps: return P
    alpha_2 = target_L2 / L2_current
    r2_new = r2 * alpha_2 
    P_new = center + np.outer(r1, v1) + np.outer(r2_new, v2)
    return P_new

def enforce_size_caps_monotone_dual(P_list, sizes, tol_L1=1.0, tol_L2=1.0):
    n = len(P_list)
    L1_pred, L2_pred = [], []
    for P in P_list:
        _, _, _, L1, L2 = pca_major_axis(P)
        L1_pred.append(L1)
        L2_pred.append(L2)
    L1_pred = np.array(L1_pred, float)
    L2_pred = np.array(L2_pred, float)

    U1 = np.array(sizes, float) + float(tol_L1)
    L1_cap = np.minimum(L1_pred, U1)
    L1_adj = L1_cap.copy()
    for i in range(n-2, -1, -1):
        L1_adj[i] = min(L1_adj[i], L1_adj[i+1])
        
    U2 = L2_pred + float(tol_L2) 
    L2_cap = np.minimum(L2_pred, U2)
    L2_adj = L2_cap.copy()
    for i in range(n-2, -1, -1):
        L2_adj[i] = min(L2_adj[i], L2_adj[i+1])

    P_adj_list = []
    for P, L1p, L1a, L2p, L2a in zip(P_list, L1_pred, L1_adj, L2_pred, L2_adj):
        P_temp = P.copy()
        if L1p > L1a + 1e-9:
            P_temp = shrink_along_pc1(P_temp, L1a)
        if L2p > L2a + 1e-9:
            P_temp = shrink_along_pc2(P_temp, L2a)
        P_adj_list.append(P_temp)
    return P_adj_list, L1_pred, L1_adj, L2_pred, L2_adj

# -------- SVR 관련 함수들 --------
def _linear_fit_multi(x, Y):
    x, Y = np.asarray(x, float), np.asarray(Y, float)
    X = np.vstack([x, np.ones_like(x)]).T
    # 최소제곱 해
    coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return coef[0], coef[1]

def _linear_predict_multi(a, b, x): return a * float(x) + b

def _blend_to_boundary(Y_linear, Y_boundary, dist_mm, tau_mm=8.0):
    gamma = np.exp(-dist_mm / max(tau_mm, 1e-6))
    return gamma * Y_boundary + (1.0 - gamma) * Y_linear

def linear_piecewise_predict(s_train, Y, s_targets):
    s_train, Y = np.array(s_train, float), np.asarray(Y, float)
    out = np.zeros((len(s_targets), Y.shape[1]), float)
    order = np.argsort(s_train)
    s_train, Y = s_train[order], Y[order]
    for i, st in enumerate(s_targets):
        if st <= s_train[0]: a, b = 0, min(1, len(s_train)-1)
        elif st >= s_train[-1]: a, b = max(0, len(s_train)-2), len(s_train)-1
        else:
            idx = np.searchsorted(s_train, st)
            a, b = idx-1, idx
        denom = (s_train[b]-s_train[a]) if b!=a else 1.0
        t = (st - s_train[a]) / (denom + 1e-12)
        out[i] = (1-t)*Y[a] + t*Y[b]
    return out

def svr_fit_predict_safe(x_train, Y, x_test, C=100.0, epsilon=0.1, length_scale=20.0, 
                         tail_k_local=3, tail_tau_mm=1.0):
    try:
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
    except Exception as e: raise ImportError("scikit-learn required.") from e
    
    x_train, Y, x_test = np.asarray(x_train, float), np.asarray(Y, float), np.asarray(x_test, float)
    order = np.argsort(x_train)
    x_train, Y = x_train[order], Y[order]
    
    if len(x_train) < 2: return linear_piecewise_predict(x_train, Y, x_test)

    # SVR 모델 설정 (RBF 커널)
    gamma = 1.0 / (2.0 * (float(length_scale)**2) + 1e-12)
    model = MultiOutputRegressor(SVR(kernel='rbf', C=float(C), epsilon=float(epsilon), gamma=gamma))
    
    X = x_train.reshape(-1,1)
    model.fit(X, Y)

    xmin, xmax = x_train[0], x_train[-1]
    out = np.zeros((len(x_test), Y.shape[1]), float)
    
    # 외삽용 국소 선형 모델
    kL = min(max(tail_k_local, 2), len(x_train))
    aL, bL = _linear_fit_multi(x_train[:kL], Y[:kL])
    aR, bR = _linear_fit_multi(x_train[-kL:], Y[-kL:])
    
    for i, st in enumerate(x_test):
        if st < xmin:
            y_lin = _linear_predict_multi(aL, bL, st)
            out[i] = _blend_to_boundary(y_lin, Y[0], (xmin - st), tail_tau_mm)
        elif st > xmax:
            y_lin = _linear_predict_multi(aR, bR, st)
            out[i] = _blend_to_boundary(y_lin, Y[-1], (st - xmax), tail_tau_mm)
        else:
            out[i] = model.predict(np.array([[st]]))[0]
    return out

# -------- 트랙 선택 (Best Match Logic) --------
def find_best_track_and_base(train_data_dict, P_new_resampled):
    L_points = len(P_new_resampled)
    best_match = (None, 10**30, None) 
    
    for type_str, size_map in train_data_dict.items():
        if not size_map: continue
        
        min_s = min(size_map.keys())
        base_P, _ = size_map[min_s]
        
        base_P_resampled = chordlen_resample(base_P, L_points)
        _, sc, _, _ = cyclic_align(P_new_resampled, base_P_resampled)
        
        if sc < best_match[1]:
            track_list = []
            for s, (p_data, _) in size_map.items():
                track_list.append((s, p_data))
            track_list.sort(key=lambda x: x[0])
            best_match = (type_str, sc, track_list)

    best_type, _, best_track_list = best_match
    if best_type is None: raise RuntimeError("매칭되는 트랙을 찾을 수 없습니다.")

    base_row = min(best_track_list, key=lambda x: x[0])
    base_P = base_row[1]
    
    return best_track_list, base_P, best_type

# ================= 핵심 처리 로직 (1개 타입에 대한 예측) =================
def process_single_type(target_type, full_data):
    """
    target_type을 홀드아웃하고, 나머지 데이터로 학습하여 예측 결과를 반환
    """
    # 1. Target 데이터
    target_map = full_data.get(target_type)
    if not target_map: return [], "No Data", 0

    min_size_target = min(target_map.keys())
    P_new_raw, side_str = target_map[min_size_target]
    
    # 2. 학습 데이터
    train_data_dict = {k: v for k, v in full_data.items() if k != target_type}
    if not train_data_dict: return [], "No Train Data", 0

    # 3. 트랙 선택
    L = len(P_new_raw)
    P_new_resampled = chordlen_resample(P_new_raw, L)
    
    try:
        track, base, best_match_type = find_best_track_and_base(train_data_dict, P_new_resampled)
    except Exception as e:
        print(f"  [Error] {e}")
        return [], "Error", 0

    sizes_train = [s for s, _ in track]
    Ps = [P for _, P in track]
    
    # 4. 정렬
    Ps = [chordlen_resample(P, L) for P in Ps]
    base = chordlen_resample(base, L)
    P_new = P_new_resampled 

    # Y축 정렬
    all_P = [P_new, base] + Ps
    min_y = min(P.min(axis=0)[1] for P in all_P)
    if min_y < 0.0:
        dy = -min_y
        P_new[:, 1] += dy
        base[:, 1] += dy
        for P in Ps: P[:, 1] += dy

    aligned_Ps = []
    for P in Ps:
        Q_best, _, _, _ = cyclic_align(base, P)
        aligned_Ps.append(Q_best)
    Ps = aligned_Ps

    # 5. SVR 학습 (dt, dn 분리 예측)
    T_base, Nvec_base = tangents_normals(base)
    Ydt = np.stack([( (P-base)*T_base ).sum(axis=1) for P in Ps])
    Ydn = np.stack([( (P-base)*Nvec_base ).sum(axis=1) for P in Ps])

    # SVR 하이퍼파라미터 (C=100, eps=0.1, LS=20)
    try:
        dt_pred = svr_fit_predict_safe(sizes_train, Ydt, TARGET_SIZES.astype(float),
                                       C=100.0, epsilon=0.1, length_scale=20.0, tail_tau_mm=2.0)
        dn_pred = svr_fit_predict_safe(sizes_train, Ydn, TARGET_SIZES.astype(float),
                                       C=100.0, epsilon=0.1, length_scale=20.0, tail_tau_mm=0.2)
    except Exception as e:
        print(f"  [Error] SVR fit failed: {e}")
        return [], "SVR Fail", 0

    # 6. 합성 및 보정
    Tn, Nn = tangents_normals(P_new)
    pred_shapes = [P_new + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None] for i in range(len(TARGET_SIZES))]
    pred_shapes_adj, _, _, _, _ = enforce_size_caps_monotone_dual(pred_shapes, TARGET_SIZES.tolist())

    # 7. 결과 행 생성
    rows = []
    for s, P_final in zip(TARGET_SIZES, pred_shapes_adj):
        row_data = [target_type, side_str, int(s)] + [f"{v:.6f}" for v in P_final.reshape(-1)]
        rows.append(row_data)
        
    return rows, best_match_type, L

# ================= 메인 실행 =================
def main():
    print(f"Loading Master Data from {MASTER_CSV}...")
    if not os.path.exists(MASTER_CSV):
        print(f"[FATAL] File not found: {MASTER_CSV}")
        return

    full_data = load_full_master_data(MASTER_CSV)
    all_types = sorted([t for t in full_data.keys() if t.startswith("Type")])
    print(f"Found Types: {all_types}")
    
    all_results = []
    runtime_stats = [] # 실행 시간 저장용
    
    # 루프 실행
    for t_type in all_types:
        print(f"\n>>> Processing Target: {t_type} ...")
        min_size = min(full_data[t_type].keys()) if full_data.get(t_type) else "N/A"

        start_t = time.perf_counter() # 타이머 시작
        
        type_results, matched_type, points_count = process_single_type(t_type, full_data)
        
        end_t = time.perf_counter() # 타이머 종료
        elapsed = end_t - start_t
        
        if type_results:
            all_results.extend(type_results)
            print(f"  [DONE] Matched: '{matched_type}', Time: {elapsed:.2f}s")
            
            # 통계 저장
            runtime_stats.append({
                "Type": t_type,
                "Base_Size": min_size,
                "Matched_Type": matched_type,
                "Time_sec": round(elapsed, 4),
                "Points": points_count
            })
        else:
            print(f"  [FAIL] Could not process {t_type}")
            runtime_stats.append({
                "Type": t_type,
                "Base_Size": min_size,
                "Matched_Type": "ERROR",
                "Time_sec": 0.0,
                "Points": 0
            })

    # 1. 통합 결과 저장
    if all_results:
        n_coords = (len(all_results[0]) - 3) // 2
        header = ["Type", "side", "size"] + [f"{ax}{i}" for i in range(1, n_coords+1) for ax in ("x","y")]

        with open(SAVE_PRED_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(all_results)
            
        print(f"\n[SUCCESS] All types processed. Unified file saved to:\n  -> {SAVE_PRED_PATH}")
    else:
        print("\n[WARNING] No results generated.")
        
    # 2. 실행 시간 요약 파일 저장
    if runtime_stats:
        fieldnames = ["Type", "Base_Size", "Matched_Type", "Time_sec", "Points"]
        with open(SAVE_SUMMARY_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(runtime_stats)
        print(f"[SAVED] Runtime Summary  -> {SAVE_SUMMARY_PATH}")

if __name__ == "__main__":
    main()