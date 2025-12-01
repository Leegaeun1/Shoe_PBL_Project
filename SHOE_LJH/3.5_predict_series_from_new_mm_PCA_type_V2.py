#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_series_all_types_unified_pca.py

[기능]
1. Data_DIR 내 MASTER_CSV 로드.
2. 존재하는 모든 Type(Type00~Type07 등)을 순회.
3. 각 Type을 Hold-out(테스트용)으로 설정하고, 해당 Type의 '최소 사이즈'를 P_new로 자동 지정.
4. 나머지 Type들 중에서 P_new와 가장 형상/방향이 유사한 Type을 'Best Match'로 선정하여 PCA+LR 학습.
5. 230~280mm 구간을 예측.
6. 모든 Type의 예측 결과를 하나의 CSV 파일로 통합 저장.
7. [NEW] 실행 시간 및 매칭 정보를 별도 요약 파일로 저장.
"""

import os, re, csv
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# ================= 설정 =================
Data_DIR = "20251120/CTRL100"

# 1. 통합 예측 결과 저장 파일명
SAVE_FILENAME = "pred_Data_PCA_230_280.csv"
SAVE_PRED_PATH = os.path.join(Data_DIR, SAVE_FILENAME)

# 2. 실행 시간 요약 저장 경로
SAVE_SUMMARY_FILENAME = "pca_runtime_summary.csv"
SAVE_SUMMARY_PATH = os.path.join(Data_DIR, SAVE_SUMMARY_FILENAME)

# 예측할 사이즈 범위 (230부터 280까지, 5단위)
TARGET_SIZES = np.arange(230, 285, 5, dtype=int) 

MASTER_CSV = os.path.join(
    Data_DIR,
    "control_points_master_L_20251124.csv" 
)
# ---------- 파일 리더 유틸리티 ----------
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

def enforce_size_caps_monotone(P_list, sizes, tol_mm=1.0):
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
    return P_adj_list

# -------- 트랙 선택 (Best Match Logic) --------
def find_best_track_and_base(train_data_dict, P_new_resampled, alpha_pca=0.5):
    L_points = len(P_new_resampled)
    best_match = (None, 10**30, None) # (type, score, list_of_tuples)
    
    v1_new, _, _, _ = pca_major_axis(P_new_resampled)

    for type_str, size_map in train_data_dict.items():
        if not size_map: continue
        
        min_s = min(size_map.keys())
        base_P, _ = size_map[min_s]
        
        base_P_resampled = chordlen_resample(base_P, L_points)
        _, sc_dist, _, _ = cyclic_align(P_new_resampled, base_P_resampled)
        
        v1_base, _, _, _ = pca_major_axis(base_P_resampled)
        dir_sim = np.abs(np.dot(v1_new, v1_base))
        
        sc_total = sc_dist + alpha_pca * (1.0 - dir_sim)
        
        if sc_total < best_match[1]:
            track_list = []
            for s, (p_data, _) in size_map.items():
                track_list.append((s, p_data))
            track_list.sort(key=lambda x: x[0])
            best_match = (type_str, sc_total, track_list)

    best_type, _, best_track_list = best_match
    if best_type is None: raise RuntimeError("매칭되는 트랙을 찾을 수 없습니다.")

    base_row = min(best_track_list, key=lambda x: x[0])
    base_P = base_row[1]
    
    return best_track_list, base_P, best_type

# ================= 핵심 처리 로직 =================
def process_single_type(target_type, full_data):
    """
    반환: (rows, matched_type, points_count)
    """
    # 1. Target 데이터 준비 (Hold-out)
    target_map = full_data.get(target_type)
    if not target_map: return [], "No Data", 0

    min_size_target = min(target_map.keys())
    P_new_raw, side_str = target_map[min_size_target]
    
    # 2. 학습 데이터 준비
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
    
    # 4. 리샘플링 및 정렬
    Ps = [chordlen_resample(P, L) for P in Ps]
    base = chordlen_resample(base, L)
    P_new = P_new_resampled
    
    aligned_Ps = []
    for P in Ps:
        Q_best, _, _, _ = cyclic_align(base, P)
        aligned_Ps.append(Q_best)
    Ps = aligned_Ps

    # 5. PCA + LR 학습
    Yd = np.stack([(P - base).reshape(-1) for P in Ps], axis=0)
    n_samples = Yd.shape[0]
    n_components = min(n_samples - 1, 20)
    
    if n_components < 1: return [], "Insufficient Samples", 0

    pca = PCA(n_components=n_components, whiten=True, random_state=0)
    Z = pca.fit_transform(Yd)

    lr = LinearRegression()
    lr.fit(np.array(sizes_train).reshape(-1, 1), Z)

    # 6. 예측
    Xt = TARGET_SIZES.reshape(-1, 1)
    Z_pred = lr.predict(Xt)
    Yd_pred = pca.inverse_transform(Z_pred)

    pred_shapes = []
    for i in range(len(TARGET_SIZES)):
        d_pred = Yd_pred[i].reshape(L, 2)
        pred_shapes.append(P_new + d_pred)

    # 7. 길이 제약 보정
    pred_shapes_adj = enforce_size_caps_monotone(pred_shapes, TARGET_SIZES.tolist())

    # 8. 결과 행 생성
    rows = []
    for s, P_final in zip(TARGET_SIZES, pred_shapes_adj):
        row_data = [target_type, side_str, int(s)]
        for val in P_final.reshape(-1):
            row_data.append(f"{val:.6f}")
        rows.append(row_data)
        
    # L(포인트 개수)과 매칭된 타입 정보 함께 반환
    return rows, best_match_type, L

# ================= 메인 실행 =================
def main():
    print("Loading Master Data...")
    if not os.path.exists(MASTER_CSV):
        print(f"[Error] File not found: {MASTER_CSV}")
        return

    full_data = load_full_master_data(MASTER_CSV)
    all_types = sorted([t for t in full_data.keys() if t.startswith("Type")])
    print(f"Found Types: {all_types}")
    
    all_results = []
    runtime_stats = [] # 실행 시간 저장용 리스트
    
    # 루프 실행
    for t_type in all_types:
        print(f"\n>>> Processing Target: {t_type} ...")
        min_size = min(full_data[t_type].keys()) if full_data.get(t_type) else "N/A"
        
        start_t = time.perf_counter() # 타이머 시작
        
        # 예측 실행
        type_results, matched_type, points_count = process_single_type(t_type, full_data)
        
        end_t = time.perf_counter() # 타이머 종료
        elapsed = end_t - start_t
        
        if type_results:
            all_results.extend(type_results)
            print(f"  [DONE] Matched: '{matched_type}', Time: {elapsed:.2f}s")
            
            # ★ 통계 저장
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

    # 1. 통합 예측 결과 저장
    if all_results:
        n_coords = (len(all_results[0]) - 3) // 2
        header = ["Type", "side", "size"] + [f"{ax}{i}" for i in range(1, n_coords+1) for ax in ("x","y")]

        with open(SAVE_PRED_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(all_results)
            
        print(f"\n[SUCCESS] All types processed. Saved to:\n  -> {SAVE_PRED_PATH}")
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