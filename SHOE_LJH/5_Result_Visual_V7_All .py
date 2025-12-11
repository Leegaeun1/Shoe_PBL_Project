import os, csv, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.path import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# =========================================================
# [1] 전역 설정 (이 부분만 수정하면 됩니다)
# =========================================================
BASE_ROOT = "20251125"
TARGET_CTRL_DIRS = [f"CTRL{i}" for i in range(10, 80, 10)] # CTRL10 ~ CTRL70
MASTER_FILENAME = "control_points_master_L_20251124.csv"

# 시각화 및 B-Spline 설정
USE_BSPLINE = True
DEGREE = 3
SAMPLES = 1500
CLOSED = False
SAVE_PLOTS = True  # 그래프 이미지 저장 여부

# 평가 지표 목록 (CSV 헤더용)
CSV_METRICS = [
    "RMSE", "MAE", "Hausdorff_max", "Chamfer_mean", 
    "Length_Error", "Width_Error", "Area_Error", 
    "IoU", "Dice", "EMD"
]

# 시각화 스타일 (자동 할당용)
STYLE_RULES = {
    "Ref":  {"color": "black", "ls": "-", "lw": 2.5},
    "PCA":  {"color": "red", "ls": ":", "lw": 2.0},
    "GPR":  {"color": "green", "ls": "-", "lw": 1.5},
    "SVR":  {"color": "purple", "ls": "-.", "lw": 1.5},
    "KRR":  {"color": "blue", "ls": "--", "lw": 1.5},
    "DEFAULT": {"color": "orange", "ls": "--", "lw": 1.5}
}

# =========================================================
# [2] 기하학 및 지표 계산 라이브러리
# =========================================================
_NUM = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][\+\-]?\d+)?$')
def _is_num(x: str) -> bool: return bool(_NUM.match(x))

def load_csv(path):
    """CSV 파일을 읽어 {Type: {Size: Points}} 형태로 반환"""
    if not os.path.exists(path): return {}
    db = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            clean_row = [c.strip() for c in row]
            if len(clean_row) < 5: continue
            t_type = clean_row[0]
            if t_type.lower() in ['type', 'side', 'size']: continue
            try: size = int(float(clean_row[2]))
            except: continue
            points = []
            for i in range(3, len(clean_row)-1, 2):
                if not _is_num(clean_row[i]) or not _is_num(clean_row[i+1]): break
                points.append([float(clean_row[i]), float(clean_row[i+1])])
            if points:
                if t_type not in db: db[t_type] = {}
                db[t_type][size] = np.array(points)
    return db

def chordlen_resample(P, n):
    P = np.asarray(P, float)
    if len(P) <= 1: return np.repeat(P[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.zeros(len(P)); u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-12: return np.repeat(P[:1], n, axis=0)
    u /= L
    s = np.linspace(0,1,n,endpoint=True)
    return np.stack([np.interp(s, u, P[:,0]), np.interp(s, u, P[:,1])], axis=1)

# B-Spline Functions
def open_uniform_knot_vector(n_ctrl, degree):
    kv = np.concatenate([np.zeros(degree+1), np.arange(1, n_ctrl-degree), np.full(degree+1, n_ctrl-degree)])
    return kv / kv[-1]

def bspline_basis(i, k, knots, t):
    t = np.asarray(t)
    if k == 0:
        last = (i+1 == len(knots)-1)
        return np.where((knots[i] <= t) & ((t < knots[i+1]) | (last & np.isclose(t, knots[i+1]))), 1.0, 0.0)
    left_den, right_den = knots[i+k]-knots[i], knots[i+k+1]-knots[i+1]
    left = ((t-knots[i])/left_den)*bspline_basis(i, k-1, knots, t) if left_den > 0 else 0.0
    right = ((knots[i+k+1]-t)/right_den)*bspline_basis(i+1, k-1, knots, t) if right_den > 0 else 0.0
    return left + right

def bspline_curve(ctrl, degree=3, samples=1000, closed=False):
    ctrl = np.asarray(ctrl, float)
    if closed: ctrl = np.concatenate([ctrl, ctrl[:degree]], axis=0)
    n = len(ctrl)
    knots = open_uniform_knot_vector(n, degree)
    t = np.linspace(0, 1, samples, endpoint=True)
    basis = np.stack([bspline_basis(i, degree, knots, t) for i in range(n)], axis=1)
    return basis @ ctrl

def cyclic_align(P, Q):
    n = len(P)
    best = (None, 1e30, 0, False)
    for rev in [False, True]:
        R = Q[::-1].copy() if rev else Q.copy()
        steps = range(0, n, max(1, n//100))
        for k in steps:
            Rk = np.roll(R, -k, axis=0)
            sc = np.sum((P-Rk)**2)
            if sc < best[1]: best = (Rk, sc, k, rev)
            
    if best[0] is not None:
        bk, brev = best[2], best[3]
        R = Q[::-1].copy() if brev else Q.copy()
        for k in range(bk - n//50, bk + n//50):
            real_k = k % n
            Rk = np.roll(R, -real_k, axis=0)
            sc = np.sum((P-Rk)**2)
            if sc < best[1]: best = (Rk, sc, real_k, brev)
    return best[0] if best[0] is not None else Q

def pca_major_axis(P):
    C = P - P.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    v1 = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    v2 = np.array([-v1[1], v1[0]])
    z1 = P @ v1
    return v1, v2

def to_heel_up_frame(P, v1=None, v2=None, y0_shift=None):
    if v1 is None or v2 is None: v1, v2 = pca_major_axis(P)
    R = np.stack([v2, v1], axis=1)
    Pp = P @ R
    if y0_shift is None: y0_shift = Pp[:,1].min()
    Pp[:,1] -= y0_shift
    return Pp, R, y0_shift

def curve_length(C):
    return np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))

# ---------------------------------------------------------
# [지표 계산 함수] 모든 Metric 포함
# ---------------------------------------------------------
def get_metrics(C_ref, C_pred):
    # 1. RMSE & MAE
    ptp = np.linalg.norm(C_ref - C_pred, axis=1)
    rmse = np.sqrt(np.mean(ptp**2))
    mae = np.mean(ptp)
    
    # 2. Length Error
    len_err = np.abs(curve_length(C_ref) - curve_length(C_pred))
    
    # 3. Width Error (X-axis range)
    w_ref = C_ref[:,0].max() - C_ref[:,0].min()
    w_pred = C_pred[:,0].max() - C_pred[:,0].min()
    w_err = np.abs(w_ref - w_pred)
    
    # 4. Area Error (Shoelace)
    def get_area(P): return 0.5*np.abs(np.dot(P[:,0], np.roll(P[:,1],1)) - np.dot(P[:,1], np.roll(P[:,0],1)))
    area_err = np.abs(get_area(C_ref) - get_area(C_pred))
    
    # 5. Geometric Distance (Chamfer/Hausdorff)
    d_mat = cdist(C_ref, C_pred)
    chamfer = 0.5 * (d_mat.min(axis=1).mean() + d_mat.min(axis=0).mean())
    hausdorff = max(d_mat.min(axis=1).max(), d_mat.min(axis=0).max())
    
    # 6. IoU & Dice (Grid Rasterization)
    def rasterize(P, res=200):
        all_pts = np.vstack([C_ref, C_pred])
        min_x, min_y = all_pts.min(axis=0)
        max_x, max_y = all_pts.max(axis=0)
        pad_x, pad_y = (max_x-min_x)*0.1, (max_y-min_y)*0.1
        x_grid = np.linspace(min_x-pad_x, max_x+pad_x, res)
        y_grid = np.linspace(min_y-pad_y, max_y+pad_y, res)
        xv, yv = np.meshgrid(x_grid, y_grid)
        points = np.vstack([xv.flatten(), yv.flatten()]).T
        return Path(P).contains_points(points)
    
    mask1 = rasterize(C_ref)
    mask2 = rasterize(C_pred)
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = inter / union if union > 0 else 0.0
    dice = 2*inter / (mask1.sum() + mask2.sum()) if (mask1.sum()+mask2.sum()) > 0 else 0.0
    
    # 7. EMD (Downsampled Hungarian)
    def get_emd_fast(P1, P2, n=200):
        idx1 = np.linspace(0, len(P1)-1, n).astype(int)
        idx2 = np.linspace(0, len(P2)-1, n).astype(int)
        d = cdist(P1[idx1], P2[idx2])
        row_ind, col_ind = linear_sum_assignment(d)
        return d[row_ind, col_ind].sum() / len(row_ind)
    
    emd = get_emd_fast(C_ref, C_pred)
    
    return {
        "RMSE": rmse, "MAE": mae, "Hausdorff_max": hausdorff, "Chamfer_mean": chamfer,
        "Length_Error": len_err, "Width_Error": w_err, "Area_Error": area_err,
        "IoU": iou, "Dice": dice, "EMD": emd
    }

def plot_comparison(ref_curve, pred_curves, size, type_label, out_dir):
    plt.figure(figsize=(7, 9))
    trim = 5
    
    # Ref 그리기
    plt.plot(ref_curve[trim:-trim,0], ref_curve[trim:-trim,1], 
             label="Reference", **STYLE_RULES["Ref"])
    
    title_lines = [f"Comparison: {type_label} - {size}mm"]
    
    for name, curve, metrics in pred_curves:
        # 스타일 자동 결정
        style = STYLE_RULES["DEFAULT"]
        for k, v in STYLE_RULES.items():
            if k in name.upper(): style = v; break
            
        label = f"{name}"
        rmse = metrics.get("RMSE", 0.0)
        iou = metrics.get("IoU", 0.0)
        title_lines.append(f"{name}: RMSE={rmse:.2f}, IoU={iou:.2f}")
        
        plt.plot(curve[trim:-trim,0], curve[trim:-trim,1], label=label, **style, alpha=0.8)
        
    plt.gca().set_aspect("equal")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Title은 너무 길면 자름
    plt.title("\n".join(title_lines[:6]), fontsize=9)
    plt.legend(loc='lower right', fontsize=8)
    
    plt.xlabel("Width Axis (mm)")
    plt.ylabel("Length Axis (mm)")
    
    save_path = os.path.join(out_dir, f"compare_{type_label}_{size}.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

# =========================================================
# [4] 메인 실행부 (자동 순회)
# =========================================================
def main():
    print(">>> [Auto Evaluation] Start scanning...")
    
    for ctrl_dir in TARGET_CTRL_DIRS:
        base_path = os.path.join(BASE_ROOT, ctrl_dir)
        master_path = os.path.join(base_path, MASTER_FILENAME)
        
        # Master 파일 확인
        if not os.path.exists(master_path):
            print(f"[Skip] No master file in {ctrl_dir}")
            continue
            
        print(f"\n=== Analyzing Folder: {ctrl_dir} ===")
        ref_db = load_csv(master_path)
        
        # 예측 파일 자동 스캔
        pred_root = os.path.join(base_path, "Predictions")
        if not os.path.exists(pred_root):
            print("  [Skip] No Predictions folder.")
            continue
            
        # 각 모델 폴더 순회
        model_dirs = [d for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d))]
        model_dbs = {}
        
        for m_name in model_dirs:
            m_path = os.path.join(pred_root, m_name)
            # 폴더 내 csv 파일 찾기 (summary 제외)
            csvs = [f for f in os.listdir(m_path) if f.endswith(".csv") and "summary" not in f]
            if csvs:
                print(f"  Found Model: {m_name}")
                model_dbs[m_name] = load_csv(os.path.join(m_path, csvs[0]))
                
        if not model_dbs:
            print("  No valid model CSVs found.")
            continue
            
        # 평가 및 결과 저장용 리스트
        summary_rows = [] 
        vis_root = os.path.join(base_path, "Multi_Compare_Output")
        if SAVE_PLOTS: os.makedirs(vis_root, exist_ok=True)
        
        # Type -> Size 순회
        sorted_types = sorted(ref_db.keys())
        for t_type in sorted_types:
            if SAVE_PLOTS:
                os.makedirs(os.path.join(vis_root, t_type), exist_ok=True)
                
            sorted_sizes = sorted(ref_db[t_type].keys())
            # 학습에 안 쓴 230~280 범위만 (옵션)
            target_sizes = [s for s in sorted_sizes if 230 <= s <= 280]
            
            for size in target_sizes:
                P_ref = ref_db[t_type][size]
                
                # Reference Curve
                if USE_BSPLINE: C_ref = bspline_curve(P_ref, degree=DEGREE, samples=SAMPLES, closed=CLOSED)
                else: C_ref = chordlen_resample(P_ref, SAMPLES)
                
                # Ref 기준 정렬 프레임 계산
                v1, v2 = pca_major_axis(C_ref)
                C_ref_h, _, _ = to_heel_up_frame(C_ref, v1, v2)
                
                plot_data_list = []
                
                for m_name, db in model_dbs.items():
                    if t_type in db and size in db[t_type]:
                        P_pred = db[t_type][size]
                        
                        # Outline 예외처리
                        is_outline = ("OUTLINE" in m_name.upper())
                        if is_outline:
                            C_pred = chordlen_resample(P_pred, SAMPLES)
                        elif USE_BSPLINE:
                            C_pred = bspline_curve(P_pred, degree=DEGREE, samples=SAMPLES, closed=CLOSED)
                        else:
                            C_pred = chordlen_resample(P_pred, SAMPLES)
                            
                        # Align & Frame
                        C_pred = cyclic_align(C_ref, C_pred)
                        C_pred_h, _, _ = to_heel_up_frame(C_pred, v1, v2)
                        
                        # 지표 계산
                        metrics = get_metrics(C_ref_h, C_pred_h)
                        
                        # 결과 저장
                        row = {"Type": t_type, "size": size, "Model": m_name}
                        row.update(metrics)
                        summary_rows.append(row)
                        
                        plot_data_list.append((m_name, C_pred_h, metrics))
                        
                # 시각화 저장
                if SAVE_PLOTS and plot_data_list:
                    plot_comparison(C_ref_h, plot_data_list, size, t_type, os.path.join(vis_root, t_type))
                    
        # CSV 저장 1: 전체 Raw Data
        if summary_rows:
            sum_path = os.path.join(base_path, "evaluation_summary_all.csv")
            keys = ["Type", "size", "Model"] + CSV_METRICS
            with open(sum_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in summary_rows:
                    clean_r = {k: r.get(k, "") for k in keys}
                    # 소수점 정리
                    for k in CSV_METRICS:
                        if isinstance(clean_r[k], float): clean_r[k] = f"{clean_r[k]:.5f}"
                    w.writerow(clean_r)
            print(f"  [Saved] Full Summary -> {sum_path}")
            
            # CSV 저장 2: 모델별 평균 (Average Performance)
            model_stats = {}
            for r in summary_rows:
                m = r["Model"]
                if m not in model_stats: model_stats[m] = {k: [] for k in CSV_METRICS}
                for k in CSV_METRICS: model_stats[m][k].append(r[k])
            
            avg_rows = []
            for m, stat in model_stats.items():
                row = {"Model": m}
                for k, v in stat.items(): row[k] = np.mean(v)
                avg_rows.append(row)
                
            avg_path = os.path.join(base_path, "model_performance_average.csv")
            with open(avg_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["Model"] + CSV_METRICS)
                w.writeheader()
                for r in avg_rows:
                    c_r = {k: (f"{v:.5f}" if isinstance(v, float) else v) for k,v in r.items()}
                    w.writerow(c_r)
            print(f"  [Saved] Averages -> {avg_path}")

    print("\n[Done] All tasks finished.")

if __name__ == "__main__":
    main()