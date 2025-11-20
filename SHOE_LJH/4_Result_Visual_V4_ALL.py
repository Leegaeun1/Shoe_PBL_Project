import os, csv, re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import MultipleLocator

# ========= 전역 설정 (필요에 따라 수정) =========

# 비교할 모든 타입 리스트
TARGET_TYPES = [f"Type{i:02d}" for i in range(8)] # Type00 ~ Type07
# 출력 폴더의 루트 경로
OUT_ROOT = "Multi_Compare_Output_40_3"

# [변경됨] Master Reference 파일 경로 (업로드한 파일)
MASTER_REF_FILE = "Fin_Excel_Data_CTRL40/control_points_master_L_20251118.csv"

# 예측 모델 파일 경로 템플릿
FILE_GPR_TEMPLATE = "Fin_Excel_Data_CTRL40/GPR/pred_Data_230_280_GPR_{}.csv"
FILE_PCA_TEMPLATE = "Fin_Excel_Data_CTRL40/PCA/pred_Data_230_280_PCA_{}.csv"
FILE_RATIO_TEMPLATE = "Fin_Excel_Data_CTRL40/RATIO/pred_Data_230_280_RATIO_{}.csv"

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
    if len(P) <= 1:
        return np.repeat(P[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.zeros(len(P)); u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-12:
        return np.repeat(P[:1], n, axis=0)
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
            if sc < best[1]:
                best = (Rk, sc, k, rev)
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
        "A_to_B_mean": float(d_ab.mean()),
        "A_to_B_median": float(np.median(d_ab)),
        "A_to_B_p90": float(np.percentile(d_ab, 90)),
        "A_to_B_p95": float(np.percentile(d_ab, 95)),
        "A_to_B_max": float(d_ab.max()),
        "B_to_A_mean": float(d_ba.mean()),
        "B_to_A_median": float(np.median(d_ba)),
        "B_to_A_p90": float(np.percentile(d_ba, 90)),
        "B_to_A_p95": float(np.percentile(d_ba, 95)),
        "B_to_A_max": float(d_ba.max()),
    }
    res["Chamfer_mean"] = 0.5*(res["A_to_B_mean"] + res["B_to_A_mean"])
    res["Hausdorff_max"] = float(max(res["A_to_B_max"], res["B_to_A_max"]))
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

# ---------- CSV 파서 (기존) ----------
def load_rows_any(path):
    out = {}
    if not os.path.exists(path):
        return out

    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row: 
                continue
            toks = [t.strip() for t in row if t.strip()]
            nums = []
            for t in toks:
                if _is_num(t):
                    nums.append(float(t))
            if len(nums) < 3:
                continue
            size = int(round(nums[0]))
            xy = np.array(nums[1:], float)
            if xy.size % 2 == 1:
                xy = xy[:-1]
            if xy.size < 4:
                continue
            P = xy.reshape(-1,2)
            out[size] = P
    return out

# ---------- [수정됨] Master CSV 파서 (헤더 없는 파일 대응) ----------
def load_ref_from_master_csv(path):
    """
    Master CSV 파일을 읽어서 Type별, Size별 좌표 데이터를 반환합니다.
    **수정사항**: 헤더가 없는 파일 구조(Type, Side, Size, x1, y1...)를 인덱스로 읽습니다.
    """
    print(f"[INFO] Master Ref 파일을 로드합니다: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Master CSV 파일을 찾을 수 없습니다: {path}")

    full_db = {}
    
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f) # DictReader가 아닌 일반 reader 사용
        
        for row_idx, row in enumerate(reader):
            if not row: continue
            
            # 공백 제거
            clean_row = [c.strip() for c in row]
            
            # 최소한 Type, Side, Size, x1, y1 (5개)은 있어야 함
            if len(clean_row) < 5:
                continue
                
            # 1열: Type (예: Type00)
            t_type = clean_row[0]
            # 만약 1열이 'Type00' 형식이 아니라 제목행('type' 등)이라면 건너뜀
            if t_type.lower() == 'type':
                continue
                
            # 3열: Size (예: 230)
            try:
                size = int(float(clean_row[2]))
            except ValueError:
                continue
                
            # 4열부터: 좌표 데이터 (x1, y1, x2, y2...)
            # clean_row[3]부터 끝까지
            coord_data = clean_row[3:]
            points = []
            
            # 2개씩 묶어서 처리
            for i in range(0, len(coord_data) - 1, 2):
                x_str = coord_data[i]
                y_str = coord_data[i+1]
                
                if not x_str or not y_str: # 값이 비어있으면 중단
                    break
                if not _is_num(x_str) or not _is_num(y_str): # 숫자가 아니면 중단
                    break
                    
                points.append([float(x_str), float(y_str)])
            
            if not points:
                continue
                
            pts_arr = np.array(points)
            
            # DB에 저장
            if t_type not in full_db:
                full_db[t_type] = {}
            full_db[t_type][size] = pts_arr

    print(f"[INFO] 로드 완료. 총 {len(full_db)}개의 Type이 발견되었습니다: {list(full_db.keys())}")
    return full_db


# ---------- 오차 계산 함수 ----------
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


# ---------- 다중 비교 함수 ----------
def multi_compare_plot(all_data, size, type_label, out_dir):
    
    # 1. 원본(Ref) 데이터 확인
    P_ref = all_data.get("Ref")
    if P_ref is None:
        return None 
    
    # 2. Resampling
    if USE_BSPLINE:
        C_ref = bspline_curve(P_ref, degree=DEGREE, samples=SAMPLES, closed=CLOSED)
    else:
        C_ref = chordlen_resample(P_ref, SAMPLES)

    # 3. 정렬 기준
    v1, v2, _ = pca_major_axis(C_ref)
    C_ref_h, R, y0 = to_heel_up_frame(C_ref, v1=v1, v2=v2, y0_shift=None)
    
    plots = {
        "Ref (Original)": {"color": 'blue', "data": C_ref_h, "metrics": {}}
    }
    all_metrics = {} 
    
    # 4. 나머지 데이터셋 처리
    for name, P_pred in all_data.items():
        if name == "Ref" or P_pred is None:
            continue
        
        if USE_BSPLINE:
            C_pred = bspline_curve(P_pred, degree=DEGREE, samples=SAMPLES, closed=CLOSED)
        else:
            C_pred = chordlen_resample(P_pred, SAMPLES)
            
        if CLOSED:
            C_pred = cyclic_align(C_ref, C_pred)
            
        C_pred_h, _, _ = to_heel_up_frame(C_pred, v1=v1, v2=v2, y0_shift=None)
        
        metrics = calculate_metrics(C_ref_h, C_pred_h)
        all_metrics[name] = metrics
        
        color = {'GPR': 'green', 'PCA': 'red', 'RATIO': 'orange'}.get(name, 'black')
        plots[name] = {"color": color, "data": C_pred_h, "metrics": metrics}

    # 5. 플롯
    if SAVE_PLOTS:
        plt.figure(figsize=(6, 10))
        TRIM_ENDS = 1 
        
        title_lines = [f"Multi-Compare: {type_label} (Size {int(size)} mm)"]
        
        for label, p_data in plots.items():
            C_h_plot = p_data["data"][TRIM_ENDS:-TRIM_ENDS]
            
            if label != "Ref (Original)":
                rmse = p_data["metrics"]["RMSE"]
                label_with_rmse = f"{label} (RMSE: {rmse:.3f})"
                title_lines.append(f" {label}: RMSE={rmse:.3f} mm")
            else:
                label_with_rmse = label
            line_style = '-'
            if label.startswith("RATIO"):
                line_style = '--'  # RATIO는 점선으로
            plt.plot(C_h_plot[:, 0], C_h_plot[:, 1], lw=2, 
                     linestyle=line_style,  # 스타일 적용
                     color=p_data["color"], label=label_with_rmse, alpha=0.9)

        ax = plt.gca() 
        ax.set_aspect("equal", "box")
        
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=-10, right=120)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_minor_locator(MultipleLocator(5))

        ax.grid(True, which='major', linestyle='--', alpha=0.5)

        plt.title("\n".join(title_lines), fontsize=10)
        plt.legend(loc='lower right')
        
        out_png = os.path.join(out_dir, f"compare_multi_{type_label}_{int(size)}mm.png")
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
            
    return all_metrics


def main():
    print("--- 다중 모델 비교 시각화 및 오차 계산 스크립트 시작 ---")
    
    # 1. Master Ref 파일 로드
    try:
        ref_db_master = load_ref_from_master_csv(MASTER_REF_FILE)
    except FileNotFoundError as e:
        print(f"[CRITICAL] {e}")
        return

    # 예측 모델 템플릿
    pred_file_templates = {
        "GPR": FILE_GPR_TEMPLATE,
        "PCA": FILE_PCA_TEMPLATE,
        "RATIO": FILE_RATIO_TEMPLATE
    }
    
    metric_header_suffixes = ["RMSE", "Hausdorff_max", "Length_Error", "Chamfer_mean",
                              "A_to_B_mean", "A_to_B_median", "A_to_B_p90", "A_to_B_p95", "A_to_B_max",
                              "B_to_A_mean", "B_to_A_median", "B_to_A_p90", "B_to_A_p95", "B_to_A_max"]
    final_header = ["Type", "size"]
    for name in ["GPR", "PCA", "RATIO"]:
        for suffix in metric_header_suffixes:
            final_header.append(f"{name}_{suffix}")

    summary_rows = []
    
    # 2. 메인 루프
    for type_label in TARGET_TYPES:
        print(f"\n[INFO] Processing {type_label}...")
        
        type_out_dir = os.path.join(OUT_ROOT, type_label)
        os.makedirs(type_out_dir, exist_ok=True)
        
        all_loaded_data = {}
        
        # (1) Ref 데이터
        if type_label in ref_db_master:
            all_loaded_data["Ref"] = ref_db_master[type_label]
        else:
            print(f"  [WARN] Ref: Master CSV에 {type_label} 데이터가 없습니다.")
            all_loaded_data["Ref"] = None

        # (2) 예측 모델 데이터
        valid_files = True
        if all_loaded_data["Ref"] is None:
            print(f"  [SKIP] {type_label}: Ref 데이터가 없어 스킵합니다.")
            valid_files = False
        
        if valid_files:
            for name, template in pred_file_templates.items():
                path = template.format(type_label)
                data = load_rows_any(path)
                if not data:
                     print(f"  [WARN] {type_label} - {name} 파일이 없거나 비어있습니다: {path}")
                all_loaded_data[name] = data
            
            # 2.3. 공통 사이즈
            ref_sizes = set(all_loaded_data['Ref'].keys())
            size_sets = [ref_sizes]
            for name in ["GPR", "PCA", "RATIO"]:
                if all_loaded_data.get(name):
                    size_sets.append(set(all_loaded_data[name].keys()))
            
            common_sizes = sorted(list(set.intersection(*size_sets)))
            
            if not common_sizes:
                print(f"  [WARN] {type_label}: 공통 사이즈가 발견되지 않았습니다. (Ref Size: {sorted(list(ref_sizes))})")
                continue
                
            print(f"  [INFO] 공통 비교 사이즈: {common_sizes}")

            # 2.4. 플롯 및 메트릭
            for size in common_sizes:
                size_data = {
                    name: data.get(size) for name, data in all_loaded_data.items() if data
                }
                
                metrics_by_model = multi_compare_plot(size_data, size, type_label, type_out_dir)
                
                row = {"Type": type_label, "size": size}
                for name in ["GPR", "PCA", "RATIO"]:
                    if name in metrics_by_model:
                        for suffix in metric_header_suffixes:
                            row[f"{name}_{suffix}"] = metrics_by_model[name][suffix]
                summary_rows.append(row)
            
    # 3. 저장
    summary_csv = os.path.join(OUT_ROOT, "multi_summary_all_models.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=final_header)
        w.writeheader()
        
        for r in summary_rows:
            output_row = {k: v for k, v in r.items() if k in ["Type", "size"]}
            for key, value in r.items():
                if key not in ["Type", "size"]:
                    output_row[key] = f"{value:.6f}"
            w.writerow(output_row)

    print(f"\n[OK] 스크립트 완료: {os.path.abspath(OUT_ROOT)} 폴더를 확인하세요.")
    print(f"[OK] 종합 오차 CSV 저장 완료: {os.path.abspath(summary_csv)}")


if __name__ == "__main__":
    main()