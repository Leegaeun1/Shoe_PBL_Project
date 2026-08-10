import os, csv, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.path import Path
from scipy.spatial.distance import cdist

BASE_ROOT = ""
TARGET_CTRL_DIRS = []
MASTER_FILENAME_TEMPLATE = "control_points_master_L_{}.csv"

USE_BSPLINE = True
DEGREE = 3
SAMPLES = 1500
CLOSED = False

SAVE_PLOTS = True 

CSV_METRICS = [
    "RMSE", "MAE", "Hausdorff_max", "Chamfer_mean", 
    "Length_Error", "Width_Error", "Size_Error", "Area_Error", 
    "IoU", "Dice", "EMD"
]

# 그래프 스타일 (점선, 실선, 색상 등)
STYLE_RULES = {
    "Ref":      {"color": "black",     "ls": "-",   "lw": 2.5},
    "RATIO_CTRL": {"color": "magenta", "ls": "-.",  "lw": 2.0},
    "PCA_KRR":  {"color": "cyan",      "ls": ":",   "lw": 2.0}, # 잘 보이라고 두께 키움
    "PCA_SVR":  {"color": "brown",     "ls": ":",   "lw": 1.5},
    "GPR":      {"color": "green",     "ls": "-",   "lw": 1.5},
    "SVR":      {"color": "purple",    "ls": "-.",  "lw": 1.5},
    "KRR":      {"color": "blue",      "ls": "--",  "lw": 1.5},
    "PCA":      {"color": "red",       "ls": ":",   "lw": 2.0},
    "DEFAULT":  {"color": "orange",    "ls": "--",  "lw": 1.5}
}

_NUM = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][\+\-]?\d+)?$')
def _is_num(x: str) -> bool: return bool(_NUM.match(x))

def load_csv(path):
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

def get_metrics(C_ref, C_pred):
    ptp = np.linalg.norm(C_ref - C_pred, axis=1)
    rmse = np.sqrt(np.mean(ptp**2))
    mae = np.mean(ptp)
    len_err = np.abs(curve_length(C_ref) - curve_length(C_pred))
    w_err = np.abs((C_ref[:,0].max()-C_ref[:,0].min()) - (C_pred[:,0].max()-C_pred[:,0].min()))
    s_err = np.abs((C_ref[:,1].max()-C_ref[:,1].min()) - (C_pred[:,1].max()-C_pred[:,1].min()))
    def get_area(P): return 0.5*np.abs(np.dot(P[:,0], np.roll(P[:,1],1)) - np.dot(P[:,1], np.roll(P[:,0],1)))
    area_err = np.abs(get_area(C_ref) - get_area(C_pred))
    d_mat = cdist(C_ref, C_pred)
    chamfer = 0.5 * (d_mat.min(axis=1).mean() + d_mat.min(axis=0).mean())
    hausdorff = max(d_mat.min(axis=1).max(), d_mat.min(axis=0).max())
    
    # Simple IoU approximation
    min_x, min_y = np.vstack([C_ref, C_pred]).min(axis=0)
    max_x, max_y = np.vstack([C_ref, C_pred]).max(axis=0)
    pad = (max_x - min_x) * 0.1
    x_grid = np.linspace(min_x-pad, max_x+pad, 100)
    y_grid = np.linspace(min_y-pad, max_y+pad, 100)
    xv, yv = np.meshgrid(x_grid, y_grid)
    points = np.vstack([xv.flatten(), yv.flatten()]).T
    mask1 = Path(C_ref).contains_points(points)
    mask2 = Path(C_pred).contains_points(points)
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = inter / union if union > 0 else 0.0
    dice = 2*inter / (mask1.sum()+mask2.sum()) if (mask1.sum()+mask2.sum()) > 0 else 0.0
    
    return {
        "RMSE": rmse, "MAE": mae, "Hausdorff_max": hausdorff, "Chamfer_mean": chamfer,
        "Length_Error": len_err, "Width_Error": w_err, "Size_Error": s_err, "Area_Error": area_err,
        "IoU": iou, "Dice": dice, "EMD": 0.0
    }

def plot_comparison(ref_curve, pred_curves, size, type_label, out_dir):
    plt.figure(figsize=(7, 9))
    trim = 5
    # Reference
    plt.plot(ref_curve[trim:-trim,0], ref_curve[trim:-trim,1], label="Ref (Answer)", **STYLE_RULES["Ref"])
    title_lines = [f"{type_label} - {size}mm"]
    
    for name, curve, metrics in pred_curves:
        style = STYLE_RULES["DEFAULT"]
        for k, v in STYLE_RULES.items():
            if k in name.upper(): style = v; break
        label = f"{name}"
        # Predicted
        plt.plot(curve[trim:-trim,0], curve[trim:-trim,1], label=label, **style, alpha=0.8)
        
    plt.gca().set_aspect("equal")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=8)
    plt.title("\n".join(title_lines), fontsize=9)
    
    # 폴더가 없으면 생성
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    plt.savefig(os.path.join(out_dir, f"compare_{type_label}_{size}.png"), dpi=100)
    plt.close()

def run_evaluation():
    for ctrl_dir in TARGET_CTRL_DIRS:
        base_path = os.path.join(BASE_ROOT, ctrl_dir)
        master_path = os.path.join(base_path, MASTER_FILENAME_TEMPLATE.format(BASE_ROOT))
        
        if not os.path.exists(master_path):
            print(f"[Skip] Master CSV not found: {master_path}")
            continue
            
        ref_db = load_csv(master_path)
        pred_root = os.path.join(base_path, "Predictions")
        if not os.path.exists(pred_root): continue
            
        model_dirs = [d for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d))]
        model_dbs = {}
        for m_name in model_dirs:
            csvs = [f for f in os.listdir(os.path.join(pred_root, m_name)) if f.endswith(".csv") and "summary" not in f]
            if csvs: model_dbs[m_name] = load_csv(os.path.join(pred_root, m_name, csvs[0]))
            
        summary_rows = []

        # =========================================================
        # 1. 클러스터 정보 로드 (Type -> Cluster_ID 매핑)
        # =========================================================
        cluster_info_path = os.path.join(base_path, "Cluster_Analysis", "Type_Cluster_Info.csv")
        type_to_cluster = {}
        
        if os.path.exists(cluster_info_path):
            try:
                with open(cluster_info_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # CSV 헤더가 Type, Cluster_ID 라고 가정
                        if "Type" in row and "Cluster_ID" in row:
                            type_to_cluster[row["Type"]] = row["Cluster_ID"]
            except Exception as e:
                print(f"[Warning] Failed to read cluster info: {e}")
        else:
            print(f"[Warning] Cluster info file not found: {cluster_info_path}")

        
        # Multi_Compare_Output 폴더 생성 
        vis_root = os.path.join(base_path, "Multi_Compare_Output")
        if SAVE_PLOTS: os.makedirs(vis_root, exist_ok=True)
        
        sorted_types = sorted(ref_db.keys())
        for t_type in sorted_types:
            if SAVE_PLOTS: os.makedirs(os.path.join(vis_root, t_type), exist_ok=True)
            
            # 현재 Type의 클러스터 ID 찾기 (없으면 'Unknown')
            current_cluster_id = type_to_cluster.get(t_type, "Unknown")
            
            target_sizes = [s for s in sorted(ref_db[t_type].keys()) if 230 <= s <= 280]
            
            for size in target_sizes:
                P_ref = ref_db[t_type][size]
                if USE_BSPLINE: C_ref = bspline_curve(P_ref, degree=DEGREE, samples=SAMPLES, closed=CLOSED)
                else: C_ref = chordlen_resample(P_ref, SAMPLES)
                
                v1, v2 = pca_major_axis(C_ref)
                C_ref_h, _, _ = to_heel_up_frame(C_ref, v1, v2)
                
                plot_data_list = []
                for m_name, db in model_dbs.items():
                    if t_type in db and size in db[t_type]:
                        if "RATIO_CTRL" in m_name.upper():
                            if size == min(db[t_type].keys()): continue
                        
                        P_pred = db[t_type][size]
                        if USE_BSPLINE: C_pred = bspline_curve(P_pred, degree=DEGREE, samples=SAMPLES, closed=CLOSED)
                        else: C_pred = chordlen_resample(P_pred, SAMPLES)
                            
                        C_pred = cyclic_align(C_ref, C_pred)
                        C_pred_h, _, _ = to_heel_up_frame(C_pred, v1, v2)
                        metrics = get_metrics(C_ref_h, C_pred_h)
                        
                        row = {"Type": t_type, "size": size, "Model": m_name}
                        # [NEW] row 데이터에 Cluster_ID 추가
                        row["Cluster_ID"] = current_cluster_id 
                        
                        row.update(metrics)
                        summary_rows.append(row)
                        plot_data_list.append((m_name, C_pred_h, metrics))
                
                if SAVE_PLOTS and plot_data_list:
                    plot_comparison(C_ref_h, plot_data_list, size, t_type, os.path.join(vis_root, t_type))

        # 평가 결과 CSV 저장 로직
        if summary_rows:
            # 1. 전체 상세 결과 저장 (Cluster_ID 컬럼 추가됨)
            sum_path = os.path.join(base_path, "evaluation_summary_all_V2.csv")
            # 헤더에 Cluster_ID 추가
            keys = ["Type", "Cluster_ID", "size", "Model"] + CSV_METRICS 
            with open(sum_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in summary_rows:
                    # 없는 키는 빈칸 처리
                    w.writerow({k: (f"{r[k]:.5f}" if isinstance(r.get(k), float) else r.get(k,"")) for k in keys})
            
            # 2. 모델별 평균 저장 (기존 로직 유지)
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
            
            avg_path = os.path.join(base_path, "model_performance_average_V2.csv")
            with open(avg_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["Model"] + CSV_METRICS)
                w.writeheader()
                for r in avg_rows:
                    w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k,v in r.items()})

            # =========================================================
            # 3. 클러스터별 평균 성능 저장 (Cluster_Performance_Summary.csv)
            # =========================================================
            # (Model, Cluster_ID) 별로 그룹화하여 평균 계산
            cluster_stats = {} 
            
            for r in summary_rows:
                # 키: (모델명, 클러스터ID)
                key = (r["Model"], r.get("Cluster_ID", "Unknown"))
                
                if key not in cluster_stats:
                    cluster_stats[key] = {k: [] for k in CSV_METRICS}
                
                for k in CSV_METRICS:
                    cluster_stats[key][k].append(r[k])
            
            cluster_avg_rows = []
            for (model_name, c_id), stats in cluster_stats.items():
                row = {"Model": model_name, "Cluster_ID": c_id}
                for k, vals in stats.items():
                    row[k] = np.mean(vals) # 평균 계산
                cluster_avg_rows.append(row)
            
            # 보기 좋게 정렬 (모델명 -> 클러스터ID 순)
            cluster_avg_rows.sort(key=lambda x: (x["Model"], str(x["Cluster_ID"])))
            
            cluster_out_path = os.path.join(base_path, "Cluster_Performance_Summary.csv")
            cluster_keys = ["Model", "Cluster_ID"] + CSV_METRICS
            
            try:
                with open(cluster_out_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cluster_keys)
                    w.writeheader()
                    for r in cluster_avg_rows:
                        w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k,v in r.items()})
                print(f"[Save] Cluster Summary: {cluster_out_path}")
            except Exception as e:
                print(f"[Error] Failed to save cluster summary: {e}")
                
def main_process(target_ctrl_num, base_date="20260106"):
    global BASE_ROOT, TARGET_CTRL_DIRS
    BASE_ROOT = base_date
    TARGET_CTRL_DIRS = [f"CTRL{target_ctrl_num}"]
    
    print(f"[Vis] Visualizing & Evaluating for CTRL{target_ctrl_num}...")
    run_evaluation()
    print(f"[Vis] Finished CTRL{target_ctrl_num} (Plots Saved)")