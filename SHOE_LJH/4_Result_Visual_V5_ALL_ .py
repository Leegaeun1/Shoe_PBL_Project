import os, csv, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ========= 전역 설정 =========

# 1. 경로 설정
Data_DIR = "20251120/CTRL40"

# Master Reference 파일 (정답지)
MASTER_REF_FILE = os.path.join(Data_DIR, "control_points_master_L_20251120.csv")

# 2. 예측 모델 파일 경로 (통합 파일)
# 파일명이 실제 환경과 다르다면 이 부분을 수정해주세요.
PRED_FILES = {
    "GPR":   os.path.join(Data_DIR, "pred_Data_GPR_230_280.csv"),
    "KRR":   os.path.join(Data_DIR, "pred_Data_KRR_230_280.csv"),
    "SVR":   os.path.join(Data_DIR, "pred_Data_SVR_230_280.csv"),
    "PCA":   os.path.join(Data_DIR, "pred_Data_PCA_230_280.csv"),
    "RATIO_CTRL": os.path.join(Data_DIR, "pred_Data_RATIO_CTRL_230_280.csv"),
    "RATIO_OUTLINE": os.path.join(Data_DIR, "pred_Data_RATIO_OUTLINE_230_280.csv")
}

# 3. 시각화 스타일 설정 (색상 및 선 모양)
# ls (linestyle): '-' (solid), '--' (dashed), ':' (dotted), '-.' (dashdot)
STYLE_CONFIG = {
    "Ref":   {"color": "black",  "ls": "-",  "lw": 2.5, "alpha": 1.0, "label": "Reference"},
    "GPR":   {"color": "green",  "ls": "-",  "lw": 1.5, "alpha": 0.8},
    "KRR":   {"color": "blue",   "ls": "--", "lw": 1.5, "alpha": 0.8},
    "SVR":   {"color": "purple", "ls": "-.", "lw": 1.5, "alpha": 0.8},
    "PCA":   {"color": "red",    "ls": ":",  "lw": 2.0, "alpha": 0.9},
    "RATIO_CTRL": {"color": "orange", "ls": "--", "lw": 1.5, "alpha": 0.8},
    "RATIO_OUTLINE": {"color": "pink", "ls": "-", "lw": 1.5, "alpha": 0.8}
}

# 4. 분석 대상 및 출력 설정
TARGET_TYPES = [f"Type{i:02d}" for i in range(8)] # Type00 ~ Type07
OUT_ROOT = "20251120/CTRL40/Multi_Compare_Output"

USE_BSPLINE = True
DEGREE = 3
SAMPLES = 1500
CLOSED = False 
SAVE_PLOTS = True
SHOW_PLOTS = False

# ============================================

# ---------- 공용 유틸 ----------
_NUM = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][\+\-]?\d+)?$')
def _is_num(x: str) -> bool:
    return bool(_NUM.match(x))

def chordlen_resample(P, n):
    P = np.asarray(P, float)
    if len(P) <= 1: return np.repeat(P[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.zeros(len(P)); u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-12: return np.repeat(P[:1], n, axis=0)
    u /= L
    s = np.linspace(0,1,n,endpoint=True)
    x = np.interp(s, u, P[:,0]); y = np.interp(s, u, P[:,1])
    return np.stack([x,y], axis=1)

def cyclic_align(P, Q):
    n = len(P)
    best = (None, 1e30, 0, False)
    for rev in [False, True]:
        R = Q[::-1].copy() if rev else Q.copy()
        for k in range(n):
            Rk = np.roll(R, -k, axis=0)
            sc = np.sum((P-Rk)**2)
            if sc < best[1]: best = (Rk, sc, k, rev)
    return best[0]

def nearest_distances(A, B, chunk=3000):
    n = A.shape[0]
    out = np.empty(n, dtype=float)
    s = 0
    while s < n:
        e = min(s + chunk, n)
        diff = A[s:e, None, :] - B[None, :, :]
        d2 = np.sum(diff*diff, axis=2)
        out[s:e] = np.sqrt(d2.min(axis=1))
        s = e
    return out

def summarize_dist(d_ab, d_ba):
    d_ab = np.asarray(d_ab); d_ba = np.asarray(d_ba)
    res = {
        "RMSE": 0.0, # Will be filled later
        "Hausdorff_max": float(max(d_ab.max(), d_ba.max())),
        "Chamfer_mean": float(0.5*(d_ab.mean() + d_ba.mean())),
    }
    return res

def pca_major_axis(P):
    C = P - P.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    v1 = Vt[0] 
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = np.array([-v1[1], v1[0]])
    z1 = P @ v1
    heel_idx = int(np.argmin(z1))
    return v1, v2, heel_idx

def to_heel_up_frame(P, v1=None, v2=None, y0_shift=None):
    if v1 is None or v2 is None:
        v1, v2, _ = pca_major_axis(P)
    R = np.stack([v2, v1], axis=1)
    Pp = P @ R
    if y0_shift is None:
        y0_shift = Pp[:,1].min()
    Pp[:,1] -= y0_shift
    return Pp, R, y0_shift

def curve_length(C):
    return np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))

# ---------- B-spline Functions ----------
def open_uniform_knot_vector(n_ctrl, degree):
    kv = np.concatenate([
        np.zeros(degree+1),
        np.arange(1, n_ctrl - degree),
        np.full(degree+1, n_ctrl - degree),
    ])
    return kv / kv[-1]

def bspline_basis(i, k, knots, t):
    t = np.asarray(t)
    if k == 0:
        last = (i+1 == len(knots)-1)
        return np.where(
            (knots[i] <= t) & ((t < knots[i+1]) | (last & np.isclose(t, knots[i+1]))),
            1.0, 0.0
        )
    left_den = knots[i+k] - knots[i]
    right_den = knots[i+k+1] - knots[i+1]
    left = 0.0 if left_den <= 0 else ((t - knots[i]) / left_den) * bspline_basis(i, k-1, knots, t)
    right = 0.0 if right_den <= 0 else ((knots[i+k+1] - t) / right_den) * bspline_basis(i+1, k-1, knots, t)
    return left + right

def bspline_curve(ctrl, degree=3, samples=1000, closed=False):
    ctrl = np.asarray(ctrl, float)
    if closed:
        ctrl = np.concatenate([ctrl, ctrl[:degree]], axis=0)
    n = len(ctrl)
    knots = open_uniform_knot_vector(n, degree)
    t = np.linspace(0, 1, samples, endpoint=True)
    basis = np.stack([bspline_basis(i, degree, knots, t) for i in range(n)], axis=1)
    xy = basis @ ctrl
    return xy

# ---------- 통합 CSV 파서 (Type, Side, Size, Coords...) ----------
def load_unified_csv(path):
    """
    통합 CSV 파일(Master, GPR, PCA 등 모두 동일 포맷)을 읽습니다.
    Format: Type, side, size, x1, y1, x2, y2, ...
    Returns: db[Type][Size] = numpy_array(N, 2)
    """
    if not os.path.exists(path):
        print(f"[WARN] 파일이 존재하지 않습니다: {path}")
        return {}

    print(f"[INFO] Loading: {os.path.basename(path)}")
    full_db = {}
    
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            clean_row = [c.strip() for c in row]
            
            if len(clean_row) < 5: continue
            
            t_type = clean_row[0] # Type
            # 헤더 스킵
            if t_type.lower() in ['type', 'side', 'size']: continue 

            try:
                size = int(float(clean_row[2])) # Size
            except ValueError: continue

            # 좌표 파싱
            coord_data = clean_row[3:]
            points = []
            for i in range(0, len(coord_data)-1, 2):
                x_str, y_str = coord_data[i], coord_data[i+1]
                if not x_str or not y_str: break
                if not _is_num(x_str) or not _is_num(y_str): break
                points.append([float(x_str), float(y_str)])
            
            if not points: continue
            
            if t_type not in full_db: full_db[t_type] = {}
            full_db[t_type][size] = np.array(points)
            
    return full_db

# ---------- 오차 계산 및 메트릭 ----------
def calculate_metrics(C_ref_h, C_pred_h):
    ptp_dist = np.linalg.norm(C_ref_h - C_pred_h, axis=1)
    rmse = np.sqrt(np.mean(ptp_dist**2))
    
    len_A = curve_length(C_ref_h)
    len_B = curve_length(C_pred_h)
    length_error = np.abs(len_A - len_B)
    
    d_ab = nearest_distances(C_ref_h, C_pred_h, chunk=4000)
    d_ba = nearest_distances(C_pred_h, C_ref_h, chunk=4000)
    
    stats = summarize_dist(d_ab, d_ba)
    stats["RMSE"] = float(rmse)
    stats["Length_Error"] = float(length_error)
    
    return stats

# ---------- 다중 비교 시각화 ----------
def multi_compare_plot(all_data_for_size, size, type_label, out_dir):
    """
    all_data_for_size: {'Ref': P, 'GPR': P, 'PCA': P ...}
    """
    # 1. Ref 데이터 확인
    P_ref = all_data_for_size.get("Ref")
    if P_ref is None: return None

    # 2. Resampling Ref
    if USE_BSPLINE:
        C_ref = bspline_curve(P_ref, degree=DEGREE, samples=SAMPLES, closed=CLOSED)
    else:
        C_ref = chordlen_resample(P_ref, SAMPLES)
    
    # 3. 정렬 기준 (Ref의 Heel-Up 프레임)
    v1, v2, _ = pca_major_axis(C_ref)
    C_ref_h, _, _ = to_heel_up_frame(C_ref, v1=v1, v2=v2, y0_shift=None)
    
    plot_items = []
    # Ref 추가
    plot_items.append({
        "name": "Ref",
        "data": C_ref_h,
        "metrics": {},
        "style": STYLE_CONFIG["Ref"]
    })
    
    metrics_summary = {}

    # 4. 모델별 데이터 처리
    model_names = sorted([k for k in all_data_for_size.keys() if k != "Ref"])
    
    for name in model_names:
        P_pred = all_data_for_size[name]
        if P_pred is None: continue
        
        # Resampling
        if USE_BSPLINE:
            C_pred = bspline_curve(P_pred, degree=DEGREE, samples=SAMPLES, closed=CLOSED)
        else:
            C_pred = chordlen_resample(P_pred, SAMPLES)
        
        # Cyclic Align to Ref
        if CLOSED:
            C_pred = cyclic_align(C_ref, C_pred)
            
        # Transform to Heel-Up
        C_pred_h, _, _ = to_heel_up_frame(C_pred, v1=v1, v2=v2, y0_shift=None)
        
        # Metric Calculation
        met = calculate_metrics(C_ref_h, C_pred_h)
        metrics_summary[name] = met
        
        # Plot Item 추가
        style = STYLE_CONFIG.get(name, {"color": "gray", "ls": "-", "lw": 1.0})
        plot_items.append({
            "name": name,
            "data": C_pred_h,
            "metrics": met,
            "style": style
        })

    # 5. Plotting
    if SAVE_PLOTS:
        plt.figure(figsize=(7, 10))
        TRIM = 5 # 양 끝점 노이즈 제거용 인덱스
        
        title_lines = [f"Comparision: {type_label} - {int(size)}mm"]
        
        for item in plot_items:
            name = item["name"]
            data = item["data"]
            style = item["style"]
            
            # Label 생성 (Ref 제외하고 RMSE 표시)
            if name == "Ref":
                label_txt = "Origin"
            else:
                rmse = item["metrics"].get("RMSE", 0.0)
                label_txt = f"{name}"
                title_lines.append(f"{name}: RMSE={rmse:.3f}")
            
            # 그래프 그리기
            plt.plot(data[TRIM:-TRIM, 0], data[TRIM:-TRIM, 1],
                     color=style["color"], linestyle=style["ls"], linewidth=style["lw"],
                     alpha=style.get("alpha", 0.8), label=label_txt)

        ax = plt.gca()
        ax.set_aspect("equal", "box")
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        
        # Axis Setting
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.set_xlabel("Length (mm)")
        ax.set_ylabel("Height (mm)")
        
        plt.title("\n".join(title_lines), fontsize=11)
        plt.legend(loc='lower right', fontsize=9)
        
        out_png = os.path.join(out_dir, f"compare_{type_label}_{int(size)}.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        
        if SHOW_PLOTS: plt.show()
        else: plt.close()

    return metrics_summary

# ================= Main =================
def main():
    print("--- Unified Comparison Script Started ---")
    
    # 1. 데이터 로드 (한 번에 모든 Type 로드)
    # (1) Master Ref
    ref_db = load_unified_csv(MASTER_REF_FILE)
    if not ref_db:
        print("[CRITICAL] Master Reference 로드 실패. 경로를 확인하세요.")
        return

    # (2) Prediction Models
    model_dbs = {}
    for m_name, m_path in PRED_FILES.items():
        db = load_unified_csv(m_path)
        if db:
            model_dbs[m_name] = db
        else:
            print(f"[WARN] {m_name} 데이터 로드 실패 또는 파일 없음: {m_path}")

    # 2. CSV 저장 준비
    summary_rows = []
    csv_metrics = ["RMSE", "Hausdorff_max", "Chamfer_mean", "Length_Error"]
    
    # Header: Type, size, GPR_RMSE, GPR_Hausdorff..., PCA_RMSE...
    header = ["Type", "size"]
    available_models = sorted(model_dbs.keys())
    for m_name in available_models:
        for met in csv_metrics:
            header.append(f"{m_name}_{met}")

    # 3. 비교 루프 (Type -> Size)
    for type_label in TARGET_TYPES:
        print(f"\n[PROCESSING] {type_label} ...")
        
        # 저장 폴더
        type_dir = os.path.join(OUT_ROOT, type_label)
        os.makedirs(type_dir, exist_ok=True)
        
        # 해당 Type의 Ref 데이터 존재 여부
        if type_label not in ref_db:
            print(f"  - Master DB에 {type_label} 없음. Skip.")
            continue
            
        ref_sizes = set(ref_db[type_label].keys())
        
        # 공통 사이즈 찾기 (Ref + 로드된 모든 모델)
        common_sizes = ref_sizes.copy()
        for m_name, db in model_dbs.items():
            if type_label in db:
                common_sizes = common_sizes.intersection(db[type_label].keys())
            else:
                # 특정 모델에 해당 타입 데이터가 아예 없으면, 그 모델은 비교에서 제외되지만
                # 다른 모델들을 위해 루프는 돌아야 함.
                pass 
        
        sorted_sizes = sorted(list(common_sizes))
        
        # 학습에 사용되지 않은 230~290 범위만 필터링할 수도 있음 (선택사항)
        target_sizes_only = [s for s in sorted_sizes if 230 <= s <= 280]
        
        if not target_sizes_only:
            print(f"  - 공통 사이즈(230~280) 없음. (Found: {sorted_sizes})")
            continue
            
        print(f"  - Compare Sizes: {target_sizes_only}")
        
        for size in target_sizes_only:
            # 데이터 수집
            data_for_plot = {"Ref": ref_db[type_label][size]}
            
            for m_name in available_models:
                if type_label in model_dbs[m_name] and size in model_dbs[m_name][type_label]:
                    data_for_plot[m_name] = model_dbs[m_name][type_label][size]
            
            # 비교 및 시각화
            metrics = multi_compare_plot(data_for_plot, size, type_label, type_dir)
            
            if metrics:
                row = {"Type": type_label, "size": size}
                for m_name, m_vals in metrics.items():
                    for k, v in m_vals.items():
                        if k in csv_metrics:
                            row[f"{m_name}_{k}"] = v
                summary_rows.append(row)

    # 4. 결과 저장
    summary_csv = os.path.join(OUT_ROOT, "unified_model_comparison.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in summary_rows:
            # 소수점 포맷팅
            fmt_row = {k: r[k] for k in ["Type", "size"]}
            for k, v in r.items():
                if k not in ["Type", "size"] and isinstance(v, (float, int)):
                    fmt_row[k] = f"{v:.5f}"
            w.writerow(fmt_row)
            
    print(f"\n[DONE] 모든 비교 완료.")
    print(f"  - 그래프 이미지: {OUT_ROOT}/<Type>/")
    print(f"  - 요약 CSV: {summary_csv}")

if __name__ == "__main__":
    main()