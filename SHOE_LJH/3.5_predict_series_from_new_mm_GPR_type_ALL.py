import os, re, csv
import numpy as np
import time
from datetime import datetime

# NOTE: 이 스크립트를 실행하려면 scikit-learn이 필요합니다.

# ========= 전역 설정 =========
Data_DIR = "20251120/CTRL100"
MASTER_CSV = os.path.join(
    Data_DIR,
    "control_points_master_L_20251124.csv" 
)

# 1. 통합 예측 결과 저장 경로
SAVE_UNIFIED_FILENAME = "pred_Data_GPR_230_280.csv"
SAVE_PRED_PATH = os.path.join(Data_DIR, SAVE_UNIFIED_FILENAME)

# 2. 실행 시간 요약 저장 경로
SAVE_SUMMARY_FILENAME = "gpr_runtime_summary.csv"
SAVE_SUMMARY_PATH = os.path.join(Data_DIR, SAVE_SUMMARY_FILENAME)

# 목표 사이즈 범위 (230 ~ 280, 5mm 단위)
SIZES_TARGET_INT = np.arange(230, 285, 5, dtype=int) 

# GPR 파라미터
N_RESTARTS = 50 
RANDOM_STATE = 0 
TAIL_K_LOCAL = 3
TAIL_TAU_MM_DT = 1.0
TAIL_TAU_MM_DN = 0.1

# ---------- 강인한 파일 리더 ----------
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

def load_train_rows(path):
    text = _read_text(path)
    rows = []
    header_skipped = False
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        if not header_skipped: 
            ln_lower = ln.lower()
            if "size" in ln_lower and "x1" in ln_lower:
                header_skipped = True
                continue
        toks = [t.strip() for t in re.split(r'[,\s]+', ln) if t.strip()]
        if len(toks) < 5: continue 
        try:
            type_str = toks[0]
            if not _NUM.match(toks[2]): continue
            size = int(round(float(toks[2]))) 
            xy_vals_str = toks[3:]
            xy = np.array([float(v) for v in xy_vals_str if _NUM.match(v)], float)
            if len(xy) < 4 or len(xy) % 2 != 0: continue
            P = xy.reshape(-1, 2)
            rows.append((type_str, size, P))
        except Exception:
            continue
    if not rows: raise RuntimeError("No valid rows found.")
    return rows

def load_train_and_new_from_same_csv(path, test_type, test_size):
    text = _read_text(path)
    train_rows, new_rows = [], []
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
            type_str = toks[0]
            side_str = toks[1]
            if not _NUM.match(toks[2]): continue
            size = int(round(float(toks[2])))
            xy_vals_str = toks[3:]
            xy = np.array([float(v) for v in xy_vals_str if _NUM.match(v)], float)
            if len(xy) < 4 or len(xy) % 2 != 0: continue
            P = xy.reshape(-1, 2)
        except Exception: continue

        if type_str == test_type and size == test_size:
            new_rows.append((P, side_str))
        elif type_str != test_type:
            train_rows.append((type_str, size, P))

    if not train_rows: raise RuntimeError(f"No train rows left after excluding type='{test_type}'.")
    if not new_rows: raise RuntimeError(f"No row found for test_type='{test_type}', size={test_size}.")
    return train_rows, new_rows[0][0], new_rows[0][1]

# --- 기하 및 GPR 유틸리티 ---
def chordlen_resample(P, n):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.zeros(len(P))
    if len(P) > 1: u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-9: return np.repeat(P[:1], n, axis=0)
    u /= L
    s = np.linspace(0,1,n,endpoint=True)
    x = np.interp(s, u, P[:,0]); y = np.interp(s, u, P[:,1])
    return np.stack([x,y], axis=1)

def cyclic_align(P, Q):
    n = len(P)
    best = (None, 10**30, 0, False)
    for rev in [False, True]:
        R = Q[::-1].copy() if rev else Q.copy()
        for k in range(n):
            Rk = np.roll(R, -k, axis=0)
            sc = np.sum((P-Rk)**2)
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
    L = float(z1.max() - z1.min())
    return v1, v2, heel_idx, L

def shrink_along_pc1(P, target_L, eps=1e-9):
    v1, v2, heel_idx, L = pca_major_axis(P)
    if L <= target_L + eps: return P
    heel = P[heel_idx]
    R = P - heel
    r1, r2 = R @ v1, R @ v2
    L_current = float(r1.max() - r1.min())
    if L_current < eps: return P
    alpha = target_L / L_current
    P_new = heel + np.outer(r1 * alpha, v1) + np.outer(r2, v2)
    return P_new

def enforce_size_caps_monotone(P_list, sizes, tol_mm=0.0):
    n = len(P_list)
    L_pred = []
    for P in P_list:
        _, _, _, L = pca_major_axis(P)
        L_pred.append(L)
    L_pred = np.array(L_pred, float)
    U = np.array(sizes, float) + float(tol_mm)
    L_cap = np.minimum(L_pred, U)
    L_adj = L_cap.copy()
    for i in range(n-2, -1, -1):
        L_adj[i] = min(L_adj[i], L_adj[i+1])
    P_adj_list = []
    for P, Lp, La in zip(P_list, L_pred, L_adj):
        if Lp <= La + 1e-9: P_adj_list.append(P)
        else: P_adj_list.append(shrink_along_pc1(P, La))
    return P_adj_list, L_pred, L_adj

def find_best_track_and_base(all_rows, P_new_resampled):
    tracks_by_type = {}
    for type_str, size, P in all_rows:
        if type_str not in tracks_by_type: tracks_by_type[type_str] = []
        tracks_by_type[type_str].append((size, P))
    if not tracks_by_type: raise RuntimeError("No tracks found.")
    L = len(P_new_resampled)
    best_match = (None, 10**30, None)
    for type_str, track_list in tracks_by_type.items():
        if not track_list: continue
        min_size_row = min(track_list, key=lambda x: x[0])
        base_P_resampled = chordlen_resample(min_size_row[1], L)
        _, sc, _, _ = cyclic_align(P_new_resampled, base_P_resampled)
        if sc < best_match[1]: best_match = (type_str, sc, track_list)
    best_type, _, best_track_list = best_match
    if best_type is None: raise RuntimeError("No matching type.")
    base_row = min(best_track_list, key=lambda x: x[0])
    track_filtered = [(s, P) for s, P in best_track_list if 230 <= s <= 290]
    if not track_filtered: track_filtered = best_track_list
    track_filtered.sort(key=lambda x: x[0])
    return track_filtered, base_row[1], best_type

def _linear_fit_multi(x, Y):
    x, Y = np.asarray(x, float), np.asarray(Y, float)
    X = np.stack([x, np.ones_like(x)], axis=1)
    XtX = X.T @ X + 1e-12 * np.eye(2)
    beta = np.linalg.inv(XtX) @ (X.T @ Y)
    return beta[0], beta[1]

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

def gpr_fit_predict_safe(x_train, Y, x_test, n_restarts=2, random_state=0, tail_k_local=3, tail_tau_mm=2.0):
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, DotProduct
    except Exception as e: raise ImportError("scikit-learn required.") from e
    
    x_train, Y, x_test = np.asarray(x_train, float), np.asarray(Y, float), np.asarray(x_test, float)
    order = np.argsort(x_train)
    x_train, Y = x_train[order], Y[order]
    X = x_train.reshape(-1,1)
    
    if X.shape[0] < 2: return linear_piecewise_predict(x_train, Y, x_test)

    kernel = (C(1.0) * RBF(length_scale=70.0) + WhiteKernel(noise_level=1e-3) + C(1.0) * DotProduct())
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=n_restarts, random_state=random_state)

    try:
        gpr.fit(X, Y)
        def gpr_predict(xx): return gpr.predict(xx.reshape(-1,1))
    except Exception:
        def gpr_predict(xx):
            preds = []
            for j in range(Y.shape[1]):
                gpr_j = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=n_restarts, random_state=random_state)
                gpr_j.fit(X, Y[:, j])
                preds.append(gpr_j.predict(xx.reshape(-1,1)))
            return np.stack(preds, axis=1)

    xmin, xmax = x_train[0], x_train[-1]
    out = np.zeros((len(x_test), Y.shape[1]), float)
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
            out[i] = gpr_predict(np.array([st]))[0]
    return out

def find_min_size_per_type_from_master(path):
    try: all_rows = load_train_rows(path)
    except Exception: return {}
    min_sizes = {}
    for type_str, size, P in all_rows:
        if type_str not in min_sizes or size < min_sizes[type_str]:
            min_sizes[type_str] = size
    return min_sizes

# --- 처리 함수 (결과값, 매칭정보, 포인트 개수 반환) ---
def process_gpr_for_type(test_type, test_size_mm):
    # 로드
    all_rows, P_new, side_str = load_train_and_new_from_same_csv(MASTER_CSV, test_type, test_size_mm)
    
    # 매칭
    L = len(P_new)
    P_new_resampled = chordlen_resample(P_new, L)
    track, base, best_type = find_best_track_and_base(all_rows, P_new_resampled)
    
    sizes_train = [s for s, _ in track]
    Ps = [chordlen_resample(P, L) for _, P in track]
    base = chordlen_resample(base, L)
    
    # 정렬
    aligned = []
    for s, P in zip(sizes_train, Ps):
        Q_best, _, _, _ = cyclic_align(base, P)
        aligned.append((s, Q_best))
    sizes_train = [s for s,_ in aligned]
    Ps = [P for _,P in aligned]

    # GPR
    T, Nvec = tangents_normals(base)
    Ydt = np.stack([( (P-base)*T ).sum(axis=1) for P in Ps])
    Ydn = np.stack([( (P-base)*Nvec ).sum(axis=1) for P in Ps])
    
    sizes_target = SIZES_TARGET_INT
    dt_pred = gpr_fit_predict_safe(sizes_train, Ydt, sizes_target.astype(float), n_restarts=N_RESTARTS, random_state=RANDOM_STATE, tail_k_local=TAIL_K_LOCAL, tail_tau_mm=TAIL_TAU_MM_DT)
    dn_pred = gpr_fit_predict_safe(sizes_train, Ydn, sizes_target.astype(float), n_restarts=N_RESTARTS, random_state=RANDOM_STATE, tail_k_local=TAIL_K_LOCAL, tail_tau_mm=TAIL_TAU_MM_DN)

    # 합성
    Tn, Nn = tangents_normals(P_new_resampled) 
    pred_shapes = [P_new_resampled + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None] for i in range(len(sizes_target))]
    pred_shapes_adj, _, _ = enforce_size_caps_monotone(pred_shapes, sizes_target.tolist(), tol_mm=0.0)

    # 결과 생성
    result_rows = []
    for s, P in zip(sizes_target, pred_shapes_adj):
        row = [test_type, side_str, int(s)] + [f"{v:.6f}" for v in P.reshape(-1)]
        result_rows.append(row)
        
    # L(포인트 개수)도 함께 반환
    return result_rows, best_type, L 

# ----------------------------- Main -----------------------------
def main():
    print("--- [START] Unified GPR Prediction with Runtime Summary ---")
    
    min_sizes = find_min_size_per_type_from_master(MASTER_CSV)
    if not min_sizes: return
    sorted_types = sorted(min_sizes.keys())
    print(f"[INFO] Target Types: {sorted_types}")
    
    all_unified_rows = [] 
    runtime_stats = [] # 실행 시간 저장용 리스트
    
    for test_type in sorted_types:
        min_size = min_sizes[test_type]
        print(f"\n>>> Processing '{test_type}' (Base: {min_size}mm)...")
        
        start_t = time.perf_counter() # 타이머 시작
        
        try:
            rows, matched_type, points_count = process_gpr_for_type(test_type, min_size)
            all_unified_rows.extend(rows)
            
            end_t = time.perf_counter() # 타이머 종료
            elapsed = end_t - start_t
            
            print(f"    [DONE] Matched: '{matched_type}', Time: {elapsed:.2f}s")
            
            # 통계 저장
            runtime_stats.append({
                "Type": test_type,
                "Base_Size": min_size,
                "Matched_Type": matched_type,
                "Time_sec": round(elapsed, 4),
                "Points": points_count
            })
            
        except Exception as e:
            print(f"    [ERROR] {test_type}: {e}")
            runtime_stats.append({
                "Type": test_type,
                "Base_Size": min_size,
                "Matched_Type": "ERROR",
                "Time_sec": 0.0,
                "Points": 0
            })

    # 1. 통합 예측 파일 저장
    if all_unified_rows:
        n_coords = (len(all_unified_rows[0]) - 3) // 2
        header = ["Type", "side", "size"] + [f"{ax}{i}" for i in range(1, n_coords+1) for ax in ("x","y")]
        
        os.makedirs(os.path.dirname(SAVE_PRED_PATH), exist_ok=True)
        with open(SAVE_PRED_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(all_unified_rows)
        print(f"\n[SAVED] Unified Prediction -> {SAVE_PRED_PATH}")
    
    # 2. 실행 시간 요약 파일 저장 
    if runtime_stats:
        os.makedirs(os.path.dirname(SAVE_SUMMARY_PATH), exist_ok=True)
        fieldnames = ["Type", "Base_Size", "Matched_Type", "Time_sec", "Points"]
        with open(SAVE_SUMMARY_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(runtime_stats)
        print(f"[SAVED] Runtime Summary  -> {SAVE_SUMMARY_PATH}")
    else:
        print("\n[WARN] No results generated.")

if __name__ == "__main__":
    main()