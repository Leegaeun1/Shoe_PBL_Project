import os, re, csv
import numpy as np
import time
from datetime import datetime

# NOTE: 이 스크립트를 실행하려면 scikit-learn이 필요합니다.

# ========= 전역 설정 =========
Data_DIR = "Fin_Excel_Data_CTRL50"
MASTER_CSV = os.path.join(
    Data_DIR,
    "control_points_master_L_20251119.csv" 
)

# 예측 파일 템플릿 (TEST_TYPE, TEST_SIZE가 동적으로 채워짐)
SAVE_PRED_TEMPLATE = os.path.join(
    Data_DIR,
    "GPR/pred_Data_230_280_GPR_{}.csv" # {Type}_{size}
)

# 목표 사이즈 범위 (GPR 예측 범위)
SIZES_TARGET_INT = np.arange(230, 285, 5, dtype=int) # 230..280

# GPR 파라미터 (원본 스크립트 유지)
N_RESTARTS = 50 
RANDOM_STATE = 0 
TAIL_K_LOCAL = 3
TAIL_TAU_MM_DT = 1.0
TAIL_TAU_MM_DN = 0.1

# ---------- 강인한 파일 리더 (기존 GPR 유틸리티 유지) ----------
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
    # (기존 GPR 스크립트의 load_train_rows 함수 내용 유지)
    text = _read_text(path)
    rows = []
    header_skipped = False
    
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
            
        if not header_skipped: 
            ln_lower = ln.lower()
            if "size" in ln_lower and "x1" in ln_lower and ("type" in ln_lower or "Type" in ln):
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
            
        except Exception as e:
            continue
            
    if not rows:
        raise RuntimeError("No valid (Type, size, P) train rows found in the training CSV.")
    return rows

def load_train_and_new_from_same_csv(path, test_type, test_size=230):
    # (기존 GPR 스크립트의 load_train_and_new_from_same_csv 함수 내용 유지)
    text = _read_text(path)
    train_rows = []
    new_rows   = []
    header_skipped = False

    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue

        if not header_skipped:
            ln_lower = ln.lower()
            if "size" in ln_lower and "x1" in ln_lower and ("type" in ln_lower or "Type" in ln):
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

        except Exception as e:
            continue

        if type_str == test_type and size == test_size:
            new_rows.append((P, side_str))
        elif type_str != test_type:
            train_rows.append((type_str, size, P))

    if not train_rows:
        raise RuntimeError(f"No train rows left in '{path}' after excluding type='{test_type}'.")
    if not new_rows:
        raise RuntimeError(f"No row found for test_type='{test_type}', size={test_size} in '{path}'.")

    if len(new_rows) > 1:
        print(f"[WARN] Multiple rows for test_type='{test_type}', size={test_size}; using the first one.")

    P_new, side_new = new_rows[0]
    return train_rows, P_new, side_new

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
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    denom = np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    T = T / denom
    Nvec = np.stack([-T[:,1], T[:,0]], axis=1)
    return T, Nvec

def pca_major_axis(P):
    C = P - P.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    v1 = Vt[0]
    v2 = Vt[1]
    z1 = (P @ v1)
    heel_idx = int(np.argmin(z1))
    L = float(z1.max() - z1.min())
    return v1, v2, heel_idx, L

def shrink_along_pc1(P, target_L, eps=1e-9):
    v1, v2, heel_idx, L = pca_major_axis(P)
    if L <= target_L + eps: return P
    heel = P[heel_idx]
    R = P - heel
    r1 = R @ v1
    r2 = R @ v2
    L_current = float(r1.max() - r1.min())
    if L_current < eps: return P
    alpha = target_L / L_current
    r1_new = r1 * alpha
    P_new = heel + np.outer(r1_new, v1) + np.outer(r2, v2)
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
        if Lp <= La + 1e-9:
            P_adj_list.append(P)
        else:
            P_adj_list.append(shrink_along_pc1(P, La))
    return P_adj_list, L_pred, L_adj

def find_best_track_and_base(all_rows, P_new_resampled):
    tracks_by_type = {}
    for type_str, size, P in all_rows:
        if type_str not in tracks_by_type: tracks_by_type[type_str] = []
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
        
        if sc < best_match[1]: best_match = (type_str, sc, track_list)

    best_type, best_score, best_track_list = best_match
    if best_type is None: raise RuntimeError("Failed to find any matching type track.")

    print(f"==> Best Match Found: Type '{best_type}' (Score: {best_score:.2f})")

    base_row = min(best_track_list, key=lambda x: x[0])
    base_P = base_row[1]
    
    track_filtered = [(s, P) for s, P in best_track_list if 230 <= s <= 290]
    
    if not track_filtered:
        print(f"Warning: No samples in 230-290 range for '{best_type}'. Using all {len(best_track_list)} samples for this type.")
        track_filtered = best_track_list
        if not track_filtered: raise RuntimeError(f"Type '{best_type}' was selected but contains no data.")

    track_filtered.sort(key=lambda x: x[0])
    return track_filtered, base_P, best_type

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

def linear_piecewise_predict(s_train, Y, s_targets):
    s_train = np.array(s_train, float)
    Y = np.asarray(Y, float)
    out = np.zeros((len(s_targets), Y.shape[1]), float)
    order = np.argsort(s_train)
    s_train = s_train[order]
    Y = Y[order]

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

def gpr_fit_predict_safe(x_train, Y, x_test, n_restarts=2, random_state=0,
                         tail_k_local=3, tail_tau_mm=2.0):
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, DotProduct
    except Exception as e:
        # scikit-learn은 이 환경에서 사용할 수 없으므로, 사용자 환경에 scikit-learn이 있다고 가정하고 코드를 제공합니다.
        # 이 가정이 틀릴 경우 사용자 환경에서 오류가 발생할 수 있습니다.
        raise ImportError("scikit-learn이 필요합니다. `pip install scikit-learn`") from e

    x_train = np.asarray(x_train, float)
    Y = np.asarray(Y, float)
    x_test = np.asarray(x_test, float)

    order = np.argsort(x_train)
    x_train = x_train[order]
    Y = Y[order]

    X = x_train.reshape(-1,1)

    if X.shape[0] < 2: return linear_piecewise_predict(x_train, Y, x_test)

    kernel = (C(1.0, (1e-3, 1e3)) * RBF(length_scale=70.0, length_scale_bounds=(1.0, 300.0))
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
              + C(1.0, (1e-3, 1e3)) * DotProduct())

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=random_state
    )

    try:
        gpr.fit(X, Y)
        def gpr_predict(xx): return gpr.predict(xx.reshape(-1,1))
    except Exception:
        def gpr_predict(xx):
            preds = []
            for j in range(Y.shape[1]):
                yj = Y[:, j]
                gpr_j = GaussianProcessRegressor(
                    kernel=kernel, alpha=0.0, normalize_y=True,
                    n_restarts_optimizer=n_restarts, random_state=random_state
                )
                gpr_j.fit(X, yj)
                preds.append(gpr_j.predict(xx.reshape(-1,1)))
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
            out[i] = gpr_predict(np.array([st]))[0]
    return out

# --- 최소 사이즈 찾기 유틸리티 ---
def find_min_size_per_type_from_master(path):
    """MASTER_CSV에서 각 Type별 최소 사이즈를 찾습니다."""
    try:
        all_rows = load_train_rows(path)
    except Exception as e:
        print(f"[FATAL ERROR] MASTER_CSV 로드 실패: {e}")
        return {}

    min_sizes = {}
    for type_str, size, P in all_rows:
        if type_str not in min_sizes or size < min_sizes[type_str]:
            min_sizes[type_str] = size
    return min_sizes

# --- GPR 예측 실행 및 시간 측정 함수 ---
def run_gpr_prediction_for_type(test_type, test_size_mm):
    """단일 Type에 대해 GPR 예측 전체 과정을 실행하고 시간을 측정합니다."""
    
    start_time = time.perf_counter()
    
    # 0) 로드 및 분리
    all_rows, P_new, side_str = load_train_and_new_from_same_csv(
        MASTER_CSV,
        test_type=test_type,
        test_size=test_size_mm,
    )
    
    # 1) 'Type' 기반 트랙 선택
    L = len(P_new)
    P_new_resampled = chordlen_resample(P_new, L)

    track, base, best_type = find_best_track_and_base(all_rows, P_new_resampled)
    
    sizes_train = [s for s, _ in track]
    Ps = [P for _, P in track]
    
    # 2) 포인트 수 조정 및 베이스 정렬
    Ps = [chordlen_resample(P, L) for P in Ps]
    base = chordlen_resample(base, L)
    P_new = P_new_resampled

    aligned = []
    for s, P in zip(sizes_train, Ps):
        Q_best, _, _, _ = cyclic_align(base, P)
        aligned.append((s, Q_best))
    sizes_train = [s for s,_ in aligned]
    Ps = [P for _,P in aligned]

    # 3) 변형장(dt,dn) 구축
    T, Nvec = tangents_normals(base)
    Ydt, Ydn = [], []
    for P in Ps:
        d = P - base
        Ydt.append((d*T).sum(axis=1))
        Ydn.append((d*Nvec).sum(axis=1))
    Ydt = np.stack(Ydt, axis=0)
    Ydn = np.stack(Ydn, axis=0)

    # 4) GPR로 dt(s), dn(s) 예측
    sizes_target = SIZES_TARGET_INT
    
    dt_pred = gpr_fit_predict_safe(
        sizes_train, Ydt, sizes_target.astype(float),
        n_restarts=N_RESTARTS, random_state=RANDOM_STATE,
        tail_k_local=TAIL_K_LOCAL, tail_tau_mm=TAIL_TAU_MM_DT
    )
    dn_pred = gpr_fit_predict_safe(
        sizes_train, Ydn, sizes_target.astype(float),
        n_restarts=N_RESTARTS, random_state=RANDOM_STATE,
        tail_k_local=TAIL_K_LOCAL, tail_tau_mm=TAIL_TAU_MM_DN
    )

    # 5) 예측된 변형장을 new_230에 합성
    Tn, Nn = tangents_normals(P_new) 
    pred_shapes = []
    for i, s in enumerate(sizes_target):
        P = P_new + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None]
        pred_shapes.append(P)

    # 6) 사이즈 상한 적용 및 단조 보정
    pred_shapes_adj, L_before, L_after = enforce_size_caps_monotone(
        pred_shapes, sizes_target.tolist(), tol_mm=0.0
    )
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    # 7) 예측 결과 저장 (기존 GPR 스크립트의 출력 파일에 저장)
    SAVE_PATH = SAVE_PRED_TEMPLATE.format(test_type, test_size_mm)
    with open(SAVE_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        num_points = P_new.shape[0] 
        header = ["Type", "side", "size"]
        for i in range(1, num_points + 1):
            header.append(f"x{i}")
            header.append(f"y{i}")
        w.writerow(header)
        
        for s, P in zip(sizes_target, pred_shapes_adj):
            row = [
                test_type, 
                side_str,
                int(s),
            ] + [f"{v:.6f}" for v in P.reshape(-1)]
            w.writerow(row)
            
    # 결과 정보 반환
    return {
        "Type": test_type,
        "Base_Size": test_size_mm,
        "Time_sec": elapsed_time,
        "Output_File": SAVE_PATH,
        "Inferred_Base_Type": best_type,
        "Points_Count": L,
    }


# ----------------------------- Main Orchestrator -----------------------------
def main():
    print("--- [START] 모든 타입 GPR 예측 및 시간 측정 스크립트 시작 ---")
    
    # 1. 모든 Type의 최소 사이즈 찾기 (Master CSV 스캔)
    min_sizes = find_min_size_per_type_from_master(MASTER_CSV)
    
    if not min_sizes:
        print(f"[FATAL ERROR] {MASTER_CSV} 파일에서 유효한 타입별 최소 사이즈를 찾을 수 없습니다. 스크립트를 종료합니다.")
        return

    # 2. 실행할 타입 목록 정의
    test_types = [f"Type{i:02d}" for i in range(8)]
    
    runtime_summary = []
    
    # 3. Type별 반복 실행 및 시간 측정
    for test_type in test_types:
        min_size = min_sizes.get(test_type)
        
        if min_size is None:
            print(f"[WARN] Type '{test_type}'에 해당하는 데이터가 MASTER_CSV에 없습니다. 스킵합니다.")
            continue
            
        print(f"\n[INFO] Starting Type '{test_type}' (Base Size: {min_size}mm)...")
        
        try:
            result = run_gpr_prediction_for_type(test_type, min_size)
            runtime_summary.append(result)
            print(f"[SUCCESS] Type '{test_type}' 예측 완료. Time: {result['Time_sec']:.3f} sec.")
            print(f"          결과 저장: {result['Output_File']}")
            
        except RuntimeError as e:
            print(f"[ERROR] Type '{test_type}' 실행 중 오류 발생: {e}")
            runtime_summary.append({"Type": test_type, "Base_Size": min_size, "Time_sec": 0.0, "Output_File": "ERROR"})
        except ImportError as e:
            print(f"\n[CRITICAL ERROR] GPR 실행 환경 오류: {e}")
            break


    # 4. 종합 CSV 파일 출력
    SUMMARY_CSV_PATH = os.path.join(MASTER_CSV, "..", "gpr_runtime_summary.csv")
    
    # 출력 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(SUMMARY_CSV_PATH), exist_ok=True)
    
    fieldnames = ["Type", "Base_Size", "Points_Count", "Time_sec", "Inferred_Base_Type", "Output_File"]
    
    with open(SUMMARY_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runtime_summary)

    print(f"\n--- [END] 스크립트 완료 ---")
    print(f"[OK] 종합 시간 측정 CSV 저장 완료: {os.path.abspath(SUMMARY_CSV_PATH)}")


if __name__ == "__main__":
    main()