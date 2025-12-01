import os, csv, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.path import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ========= 전역 설정 =========

# 1. 경로 설정
Data_DIR = "20251125/CTRL60"

# Master Reference 파일 (정답지)
MASTER_REF_FILE = os.path.join(Data_DIR, "control_points_master_L_20251124.csv")

# 2. 예측 모델 파일 경로 (통합 파일)
PRED_FILES = {
    "GPR":   os.path.join(Data_DIR, "pred_Data_GPR_230_280.csv"),
    "KRR":   os.path.join(Data_DIR, "pred_Data_KRR_230_280.csv"),
    "SVR":   os.path.join(Data_DIR, "pred_Data_SVR_230_280.csv"),
    "PCA":   os.path.join(Data_DIR, "pred_Data_PCA_230_280.csv"),
    "RATIO_CTRL": os.path.join(Data_DIR, "pred_Data_RATIO_CTRL_230_280.csv"),
    "RATIO_OUTLINE": os.path.join(Data_DIR, "pred_Data_RATIO_OUTLINE_230_280.csv")
}

# 3. 시각화 스타일 설정
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
TARGET_TYPES = [f"Type{i:02d}" for i in range(8)]
OUT_ROOT = os.path.join(Data_DIR,"Multi_Compare_Output")

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
        # 간단한 최적화를 위해 step을 둠 (속도 향상)
        steps = range(0, n, max(1, n//100)) 
        for k in steps:
            Rk = np.roll(R, -k, axis=0)
            sc = np.sum((P-Rk)**2)
            if sc < best[1]: best = (Rk, sc, k, rev)
    
    # Fine tuning around best k
    if best[0] is not None:
        bk = best[2]
        brev = best[3]
        R = Q[::-1].copy() if brev else Q.copy()
        search_range = range(bk - n//50, bk + n//50)
        for k in search_range:
            real_k = k % n
            Rk = np.roll(R, -real_k, axis=0)
            sc = np.sum((P-Rk)**2)
            if sc < best[1]: best = (Rk, sc, real_k, brev)
            
    return best[0] if best[0] is not None else Q

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
    # v1: Major(Y), v2: Minor(X) in target frame?
    # 기존 코드 로직: R = [v2, v1] -> x_new = P dot v2 (Width), y_new = P dot v1 (Length)
    R = np.stack([v2, v1], axis=1)
    Pp = P @ R
    if y0_shift is None:
        y0_shift = Pp[:,1].min()
    Pp[:,1] -= y0_shift
    return Pp, R, y0_shift

def curve_length(C):
    return np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))

# ---------- New Metric Helpers ----------
def get_area_shoelace(P):
    """Shoelace formula for polygon area"""
    x = P[:, 0]
    y = P[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_width_aligned(P_aligned):
    """
    P_aligned[:, 0] is Minor Axis (Width direction)
    P_aligned[:, 1] is Major Axis (Length direction)
    """
    return P_aligned[:, 0].max() - P_aligned[:, 0].min()

def get_iou_dice(P1, P2, grid_res=200):
    """Calculate IoU and Dice using grid rasterization"""
    # 1. Bounding Box
    all_pts = np.vstack([P1, P2])
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)
    pad_x = (max_x - min_x) * 0.1
    pad_y = (max_y - min_y) * 0.1
    
    x_grid = np.linspace(min_x - pad_x, max_x + pad_x, grid_res)
    y_grid = np.linspace(min_y - pad_y, max_y + pad_y, grid_res)
    xv, yv = np.meshgrid(x_grid, y_grid)
    points = np.vstack([xv.flatten(), yv.flatten()]).T
    
    # 2. Rasterize
    path1 = Path(P1)
    path2 = Path(P2)
    # contains_points returns boolean mask
    mask1 = path1.contains_points(points)
    mask2 = path2.contains_points(points)
    
    # 3. Intersection & Union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    area1 = mask1.sum()
    area2 = mask2.sum()
    
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (area1 + area2) if (area1 + area2) > 0 else 0.0
    
    return iou, dice

def get_emd(P1, P2, downsample_n=200):
    """
    Earth Mover's Distance using Hungarian Algorithm.
    Downsamples data to downsample_n points to save computation time.
    """
    n1, n2 = len(P1), len(P2)
    
    # Simple uniform downsampling
    if n1 > downsample_n:
        idx1 = np.linspace(0, n1-1, downsample_n).astype(int)
        P1_s = P1[idx1]
    else:
        P1_s = P1
        
    if n2 > downsample_n:
        idx2 = np.linspace(0, n2-1, downsample_n).astype(int)
        P2_s = P2[idx2]
    else:
        P2_s = P2
        
    # Cost Matrix
    d_matrix = cdist(P1_s, P2_s)
    
    # Linear Sum Assignment (Hungarian Algo)
    row_ind, col_ind = linear_sum_assignment(d_matrix)
    
    # Average Match Distance
    emd_cost = d_matrix[row_ind, col_ind].sum() / len(row_ind)
    return emd_cost

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

# ---------- 통합 CSV 파서 ----------
def load_unified_csv(path):
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
            
            t_type = clean_row[0] 
            if t_type.lower() in ['type', 'side', 'size']: continue 

            try:
                size = int(float(clean_row[2])) 
            except ValueError: continue

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
    # 1. Point-to-Point Distances (Assumed Aligned)
    ptp_dist = np.linalg.norm(C_ref_h - C_pred_h, axis=1)
    rmse = np.sqrt(np.mean(ptp_dist**2))
    
    # 2. Length Error
    len_A = curve_length(C_ref_h)
    len_B = curve_length(C_pred_h)
    length_error = np.abs(len_A - len_B)
    
    # 3. Geometric Distances (Hausdorff, Chamfer)
    d_ab = nearest_distances(C_ref_h, C_pred_h, chunk=4000)
    d_ba = nearest_distances(C_pred_h, C_ref_h, chunk=4000)
    stats = summarize_dist(d_ab, d_ba)
    
    # 4. Add Basics
    stats["RMSE"] = float(rmse)
    stats["Length_Error"] = float(length_error)
    
    # --- NEW METRICS ---
    
    # 5. MAE (Mean Absolute Error)
    stats["MAE"] = float(np.mean(ptp_dist))
    
    # 6. Width Error (using minor axis range)
    w_ref = get_width_aligned(C_ref_h)
    w_pred = get_width_aligned(C_pred_h)
    stats["Width_Error"] = float(np.abs(w_ref - w_pred))
    
    # 7. Area Error
    a_ref = get_area_shoelace(C_ref_h)
    a_pred = get_area_shoelace(C_pred_h)
    stats["Area_Error"] = float(np.abs(a_ref - a_pred))
    
    # 8. IoU & Dice (Grid based approximation)
    iou, dice = get_iou_dice(C_ref_h, C_pred_h, grid_res=200)
    stats["IoU"] = float(iou)
    stats["Dice"] = float(dice)
    
    # 9. EMD (Earth Mover's Distance)
    # Computing exact matching on resampled points (N~200)
    emd_val = get_emd(C_ref_h, C_pred_h, downsample_n=200)
    stats["EMD"] = float(emd_val)
    
    return stats

# ---------- 다중 비교 시각화 ----------
def multi_compare_plot(all_data_for_size, size, type_label, out_dir):
    P_ref = all_data_for_size.get("Ref")
    if P_ref is None: return None

    if USE_BSPLINE:
        C_ref = bspline_curve(P_ref, degree=DEGREE, samples=SAMPLES, closed=CLOSED)
    else:
        C_ref = chordlen_resample(P_ref, SAMPLES)
    
    v1, v2, _ = pca_major_axis(C_ref)
    C_ref_h, _, _ = to_heel_up_frame(C_ref, v1=v1, v2=v2, y0_shift=None)
    
    plot_items = []
    plot_items.append({
        "name": "Ref",
        "data": C_ref_h,
        "metrics": {},
        "style": STYLE_CONFIG["Ref"]
    })
    
    metrics_summary = {}
    model_names = sorted([k for k in all_data_for_size.keys() if k != "Ref"])
    
    for name in model_names:
        P_pred = all_data_for_size[name]
        if P_pred is None: continue
        
        # ==========================================
        # ★ [수정] OUTLINE 데이터에 대한 예외 처리
        # ==========================================
        is_outline_data = ("OUTLINE" in name.upper()) # 이름에 OUTLINE이 있으면 True
        
        if is_outline_data:
            # 외곽선 데이터는 이미 곡선이므로, B-Spline을 태우지 않고
            # 비교를 위해 점 개수만 1500개로 맞춰줍니다 (선형 보간).
            C_pred = chordlen_resample(P_pred, SAMPLES)
            
        elif USE_BSPLINE:
            # 컨트롤 포인트 데이터는 B-Spline 곡선으로 변환합니다.
            C_pred = bspline_curve(P_pred, degree=DEGREE, samples=SAMPLES, closed=CLOSED)
            
        else:
            # B-Spline을 안 쓰기로 설정했다면 그냥 리샘플링
            C_pred = chordlen_resample(P_pred, SAMPLES)
        # ==========================================
        
        # (이하 동일: 정렬 및 메트릭 계산)
        if CLOSED:
            C_pred = cyclic_align(C_ref, C_pred)
        else:
            C_pred = cyclic_align(C_ref, C_pred)
            
        C_pred_h, _, _ = to_heel_up_frame(C_pred, v1=v1, v2=v2, y0_shift=None)
        
        met = calculate_metrics(C_ref_h, C_pred_h)
        metrics_summary[name] = met
        
        style = STYLE_CONFIG.get(name, {"color": "gray", "ls": "-", "lw": 1.0})
        plot_items.append({
            "name": name,
            "data": C_pred_h,
            "metrics": met,
            "style": style
        })

    if SAVE_PLOTS:
        plt.figure(figsize=(7, 10))
        TRIM = 5 
        
        title_lines = [f"Comparision: {type_label} - {int(size)}mm"]
        
        for item in plot_items:
            name = item["name"]
            data = item["data"]
            style = item["style"]
            
            if name == "Ref":
                label_txt = "Origin"
            else:
                rmse = item["metrics"].get("RMSE", 0.0)
                iou = item["metrics"].get("IoU", 0.0)
                label_txt = f"{name}"
                # 타이틀에는 RMSE와 IoU 두 개 정도만 표시 (너무 길어짐 방지)
                title_lines.append(f"{name}: RMSE={rmse:.2f}, IoU={iou:.2f}")
            
            plt.plot(data[TRIM:-TRIM, 0], data[TRIM:-TRIM, 1],
                     color=style["color"], linestyle=style["ls"], linewidth=style["lw"],
                     alpha=style.get("alpha", 0.8), label=label_txt)

        ax = plt.gca()
        ax.set_aspect("equal", "box")
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.set_xlabel("Width Axis (mm)")
        ax.set_ylabel("Length Axis (mm)")
        
        plt.title("\n".join(title_lines), fontsize=10)
        plt.legend(loc='lower right', fontsize=9)
        
        out_png = os.path.join(out_dir, f"compare_{type_label}_{int(size)}.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        
        if SHOW_PLOTS: plt.show()
        else: plt.close()

    return metrics_summary

# ================= Main =================
def main():
    print("--- Unified Comparison Script Started ---")
    
    ref_db = load_unified_csv(MASTER_REF_FILE)
    if not ref_db:
        print("[CRITICAL] Master Reference 로드 실패.")
        return

    model_dbs = {}
    for m_name, m_path in PRED_FILES.items():
        db = load_unified_csv(m_path)
        if db:
            model_dbs[m_name] = db
        else:
            print(f"[WARN] {m_name} 데이터 로드 실패: {m_path}")

    summary_rows = []
    # 추가된 지표 목록
    csv_metrics = [
        "RMSE", "MAE", "Hausdorff_max", "Chamfer_mean", 
        "Length_Error", "Width_Error", "Area_Error", 
        "IoU", "Dice", "EMD"
    ]
    
    header = ["Type", "size"]
    available_models = sorted(model_dbs.keys())
    for m_name in available_models:
        for met in csv_metrics:
            header.append(f"{m_name}_{met}")

    for type_label in TARGET_TYPES:
        print(f"\n[PROCESSING] {type_label} ...")
        
        type_dir = os.path.join(OUT_ROOT, type_label)
        os.makedirs(type_dir, exist_ok=True)
        
        if type_label not in ref_db:
            continue
            
        ref_sizes = set(ref_db[type_label].keys())
        common_sizes = ref_sizes.copy()
        for m_name, db in model_dbs.items():
            if type_label in db:
                common_sizes = common_sizes.intersection(db[type_label].keys())
        
        sorted_sizes = sorted(list(common_sizes))
        target_sizes_only = [s for s in sorted_sizes if 230 <= s <= 280]
        
        if not target_sizes_only:
            print(f"  - No common sizes (230-280). Found: {sorted_sizes}")
            continue
            
        print(f"  - Sizes: {target_sizes_only}")
        
        for size in target_sizes_only:
            data_for_plot = {"Ref": ref_db[type_label][size]}
            
            for m_name in available_models:
                if type_label in model_dbs[m_name] and size in model_dbs[m_name][type_label]:
                    data_for_plot[m_name] = model_dbs[m_name][type_label][size]
            
            metrics = multi_compare_plot(data_for_plot, size, type_label, type_dir)
            
            if metrics:
                row = {"Type": type_label, "size": size}
                for m_name, m_vals in metrics.items():
                    for k, v in m_vals.items():
                        if k in csv_metrics:
                            row[f"{m_name}_{k}"] = v
                summary_rows.append(row)

    summary_csv = os.path.join(OUT_ROOT, "unified_model_comparison_v2.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in summary_rows:
            fmt_row = {k: r[k] for k in ["Type", "size"]}
            for k, v in r.items():
                if k not in ["Type", "size"] and isinstance(v, (float, int)):
                    fmt_row[k] = f"{v:.5f}"
            w.writerow(fmt_row)
            
    print(f"\n[DONE] 완료.")
    print(f"  - 요약 CSV: {summary_csv}")

if __name__ == "__main__":
    main()