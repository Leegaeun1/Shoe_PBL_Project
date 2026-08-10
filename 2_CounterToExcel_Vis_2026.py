import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from matplotlib.ticker import MultipleLocator
import re

# --- Matplotlib 백엔드 설정 ---
preferred_backends = ["QtAgg", "Qt5Agg", "TkAgg"]
for be in preferred_backends:
    try:
        matplotlib.use(be, force=True)
        break
    except Exception:
        pass

# 전역 변수 선언
OUTPUT_DIR = ""
MASTER_CSV = ""
DEGREE = 3

def save_plot(filepath, title, data_plots, align_to_heel=False):
    """(기존 코드와 동일)"""
    try:
        fig, ax = plt.subplots(figsize=(8, 10))
        all_data_for_bounds = []
        v1, v2, y0_shift = None, None, None

        if align_to_heel and data_plots:
            ref_data = None
            for item in data_plots:
                d = item.get('data')
                if d is not None and d.ndim == 2 and d.shape[1] == 2 and len(d) > 1:
                    ref_data = d
                    break
            if ref_data is not None:
                try:
                    v1, v2, _ = pca_major_axis(ref_data)
                    _, _, y0_shift = to_heel_up_frame(ref_data, v1, v2)
                except Exception:
                    align_to_heel = False
            else:
                align_to_heel = False

        for item in data_plots:
            points = item.get('data')
            if points is None: continue
            style = item.get('style', '.-')
            label = item.get('label', '')
            zorder = item.get('zorder', 1)
            
            if points.ndim == 2 and points.shape[1] == 2 and len(points) > 0:
                if align_to_heel:
                    points_to_plot, _, _ = to_heel_up_frame(points, v1, v2, y0_shift)
                else:
                    points_to_plot = points
                ax.plot(points_to_plot[:, 0], points_to_plot[:, 1], style, label=label, zorder=zorder, markersize=2)
                all_data_for_bounds.append(points_to_plot)

        if not all_data_for_bounds:
            plt.close(fig); return

        ax.set_aspect("equal", "box")
        ax.set_title(title, fontsize=10)
        
        if align_to_heel:
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5))
            ax.set_ylim(bottom=-5) 
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
        else:
            ax.invert_yaxis() 
            ax.grid(True, alpha=0.3)

        if any(item.get('label') and not item['label'].startswith('_') for item in data_plots):
            if align_to_heel: ax.legend(loc='center right') 
            else: ax.legend(loc="best")

        all_pts = np.vstack(all_data_for_bounds)
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        pad_x = 0.1 * (xmax - xmin + 1e-9)
        pad_y = 0.1 * (ymax - ymin + 1e-9)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        if not align_to_heel: ax.set_ylim(ymax + pad_y, ymin - pad_y) 

        fig.savefig(filepath)
        plt.close(fig)
    except Exception as e:
        print(f"[ERROR] Failed to save plot '{title}': {e}")
        if 'fig' in locals(): plt.close(fig)

def pca_major_axis(P):
    C = P - P.mean(axis=0, keepdims=True)
    try: _, _, Vt = np.linalg.svd(C, full_matrices=False)
    except: Vt = np.array([[1.0, 0.0], [0.0, 1.0]])
    v1 = Vt[0]; v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = np.array([-v1[1], v1[0]])
    z1 = P @ v1
    heel_idx = int(np.argmin(z1))
    return v1, v2, heel_idx

def to_heel_up_frame(P, v1=None, v2=None, y0_shift=None):
    if v1 is None or v2 is None: v1, v2, _ = pca_major_axis(P)
    R = np.stack([v2, v1], axis=1)
    Pp = P @ R
    if y0_shift is None: y0_shift = Pp[:,1].min()
    Pp[:,1] -= y0_shift
    return Pp, R, y0_shift

# --- 이미지 처리 함수들 (간략화) ---
def load_binary_outline(image_path, thresh=0.8):
    import matplotlib.image as mpimg
    img = mpimg.imread(image_path)
    if img.ndim == 3: gray = img[..., :3].mean(axis=2)
    else: gray = img.astype(float)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)
    mask = (gray < thresh).astype(np.uint8)
    try:
        from scipy.ndimage import binary_dilation, binary_closing
        mask = binary_closing(mask, iterations=1)
        mask = binary_dilation(mask, iterations=1)
    except: pass
    return mask

def largest_contour(mask):
    from skimage import measure
    contours = measure.find_contours(mask, level=0.5)
    if not contours: raise RuntimeError("No contour found.")
    return np.fliplr(max(contours, key=len))

def resample_polyline(points, n_samples):
    P = np.vstack([points, points[0]])
    seg_lengths = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cumulative_lengths = np.hstack(np.cumsum(seg_lengths))
    total_length = cumulative_lengths[-1]
    u = np.linspace(0, total_length, n_samples, endpoint=False)
    indices = np.searchsorted(cumulative_lengths, u, side="right") - 1
    indices = np.clip(indices, 0, len(P) - 2)
    t = (u - cumulative_lengths[indices]) / (seg_lengths[indices] + 1e-9)
    return (1 - t)[:, None] * P[indices] + t[:, None] * P[indices + 1]

# --- B-스플라인 함수 ---
def open_uniform_knot_vector(n_ctrl, degree):
    knots = np.concatenate([np.zeros(degree + 1), np.arange(1, n_ctrl - degree), np.full(degree + 1, n_ctrl - degree)])
    return knots / np.max(knots)

def bspline_basis(i, degree, knots, t):
    if degree == 0:
        is_last_knot = (i + 1 == len(knots) - 1)
        if (knots[i] <= t < knots[i+1]) or (is_last_knot and np.isclose(t, knots[i+1])): return 1.0
        return 0.0
    term1 = 0.0
    den1 = knots[i+degree] - knots[i]
    if den1 > 1e-9: term1 = (t - knots[i]) / den1 * bspline_basis(i, degree - 1, knots, t)
    term2 = 0.0
    den2 = knots[i+degree+1] - knots[i+1]
    if den2 > 1e-9: term2 = (knots[i+degree+1] - t) / den2 * bspline_basis(i + 1, degree - 1, knots, t)
    return term1 + term2

def bspline_curve(ctrl_points, degree, knots, t_values):
    n_ctrl = len(ctrl_points)
    curve = np.zeros((len(t_values), ctrl_points.shape[1]))
    for j, t in enumerate(t_values):
        point = np.zeros(ctrl_points.shape[1])
        for i in range(n_ctrl):
            w = bspline_basis(i, degree, knots, t)
            if w > 1e-9: point += w * ctrl_points[i]
        curve[j] = point
    return curve

def fit_open_bspline_least_squares(points, n_ctrl=20, degree=3, lam=1e-5):
    M = len(points)
    knots = open_uniform_knot_vector(n_ctrl, degree)
    ts = np.linspace(0, 1, M, endpoint=True)
    A = np.zeros((M, n_ctrl))
    for j, t in enumerate(ts):
        for i in range(n_ctrl):
            A[j, i] = bspline_basis(i, degree, knots, t)
    if lam > 0:
        D = np.eye(n_ctrl, k=0) * -2 + np.eye(n_ctrl, k=1) * 1 + np.eye(n_ctrl, k=-1) * 1
        D = D[1:-1]
        ATA = A.T @ A + lam * (D.T @ D)
    else:
        ATA = A.T @ A
    ATYx = A.T @ points[:, 0]
    ATYy = A.T @ points[:, 1]
    ctrl_x = np.linalg.solve(ATA + 1e-9 * np.eye(n_ctrl), ATYx)
    ctrl_y = np.linalg.solve(ATA + 1e-9 * np.eye(n_ctrl), ATYy)
    return np.stack([ctrl_x, ctrl_y], axis=1), knots

# --- CSV & Parsing ---
def parse_size_mm_from_filename(path, fallback=250.0):
    m = re.search(r'(\d{2,3})(?=[^\d]|$)', os.path.basename(path))
    return float(m.group(1)) if m else float(fallback)

def parse_type_from_filename(path, default="unknown"):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    if "_" in name:
        t = name.split("_", 1)[0].strip()
        return t if t else default
    return default

def get_or_assign_type_label(type_key):
    global TYPE_LABELS
    if type_key not in TYPE_LABELS:
        TYPE_LABELS[type_key] = f"Type{len(TYPE_LABELS):02d}"
    return TYPE_LABELS[type_key]

def parse_side_from_filename(path, default=""):
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    matches = list(re.finditer(r'(\d{2,3})(?!\d)', base))
    if not matches: return default
    m = matches[-1]
    suffix = base[m.end():]
    tokens = re.findall(r'[^\W\d_]+', suffix, flags=re.UNICODE)
    if not tokens: return default
    return "_".join(tokens).upper()

def _ensure_master_header(path, n_ctrl, sep=", "):
    if os.path.exists(path) and os.path.getsize(path) > 0: return
    cols = ["type", "side", "size"]
    for i in range(1, n_ctrl + 1): cols += [f"x{i}", f"y{i}"]
    with open(path, "w", encoding="utf-8") as f: f.write(sep.join(cols) + "\n")

def save_ctrl_to_master(image_path, ctrl_points, sep=", ", type_label=None):
    n_ctrl = ctrl_points.shape[0]
    _ensure_master_header(MASTER_CSV, n_ctrl, sep=sep)
    coords = ctrl_points.copy()
    min_x, min_y = coords.min(axis=0)
    coords[:, 0] -= min_x
    coords[:, 1] -= min_y
    size_mm = parse_size_mm_from_filename(image_path, fallback=250.0)
    side = parse_side_from_filename(image_path, default="")
    if type_label is None:
        type_key = parse_type_from_filename(image_path, default="unknown")
        type_label = get_or_assign_type_label(type_key)
    flat = coords.flatten()
    line = sep.join([type_label, side, f"{size_mm:.0f}"] + [f"{v:.6f}" for v in flat])
    with open(MASTER_CSV, "a", encoding="utf-8") as f: f.write(line + "\n")
    print(f"[APPEND] -> {MASTER_CSV} (type='{type_label}', size={size_mm:.0f})")

def run_open_fit_DIR(DIR_path, n_contour_points=200, n_ctrl_points=25, degree=3):
    print("Creating debug output directories...")
    debug_dir_base = os.path.join(OUTPUT_DIR, "debug_outputs") 
    dir_1_mask = os.path.join(debug_dir_base, "1_masks")
    dir_2_contour = os.path.join(debug_dir_base, "2_contours_pixel")
    dir_3_polygon = os.path.join(debug_dir_base, "3_polygon_mm")
    dir_4_bspline = os.path.join(debug_dir_base, "4_bspline_fit_mm")
    
    for d in [dir_1_mask, dir_2_contour, dir_3_polygon, dir_4_bspline]:
        os.makedirs(d, exist_ok=True)
    
    jpgs = sorted(glob.glob(os.path.join(DIR_path, "*.jpg")))
    if not jpgs:
        print(f"[ERROR] No .jpg files found in: {DIR_path}")
        return

    for img in jpgs:
        try:
            filename_base = os.path.basename(img)
            print(f"\n[PREP] {filename_base}")

            # 1. Masking
            mask = load_binary_outline(img)
            plt.imsave(os.path.join(dir_1_mask, filename_base + ".png"), mask, cmap='gray')

            # 2. Contour
            contour = largest_contour(mask)
            save_plot(os.path.join(dir_2_contour, filename_base + ".png"), 
                      title=f"2. Pixel Contour\n{filename_base}", 
                      data_plots=[{'data': contour, 'style': 'b.', 'label': 'Raw'}], align_to_heel=False)

            # 3. Resample
            contour_resampled = resample_polyline(contour, n_contour_points)
            toe_index = np.argmax(contour_resampled[:, 1])
            contour_rolled = np.roll(contour_resampled, toe_index, axis=0)
            contour_path = np.vstack([contour_rolled, contour_rolled[0]])

            # 4. Scale to mm
            size_mm = parse_size_mm_from_filename(img, fallback=250.0)
            contour_mm = (contour_path - [contour_path[:,0].min(), contour_path[:,1].min()]) * (size_mm / (contour_path[:,1].max() - contour_path[:,1].min() + 1e-9))
            
            save_plot(os.path.join(dir_3_polygon, filename_base + ".png"),
                      title=f"3. Resampled Polygon\n{filename_base}",
                      data_plots=[{'data': contour_mm, 'style': 'k.-', 'label': f'Polygon'}], align_to_heel=True) 

            # 5. Fit B-Spline
            ctrl, knots = fit_open_bspline_least_squares(contour_mm, n_ctrl=n_ctrl_points, degree=degree)
            
            # 6. Save Curve Plot
            t_values = np.linspace(0, 1, 400, endpoint=True)
            final_curve_raw = bspline_curve(ctrl, degree, knots, t_values)
            final_curve = np.vstack([final_curve_raw, final_curve_raw[0]])
            ctrl_closed = np.vstack([ctrl, ctrl[0]])
            
            save_plot(os.path.join(dir_4_bspline, filename_base + ".png"),
                      title=f"4. B-Spline Fit\n{filename_base}",
                      data_plots=[
                          {'data': contour_mm, 'style': 'k-', 'label': 'Target', 'zorder': 1},
                          {'data': final_curve, 'style': 'r-', 'label': 'Curve', 'zorder': 3}, 
                          {'data': ctrl, 'style': 'bo', 'label': f'Ctrl ({n_ctrl_points})', 'zorder': 4},
                          {'data': ctrl_closed, 'style': 'b--', 'label': 'Poly', 'zorder': 2}
                      ], align_to_heel=True) 
            
            # 7. Append to CSV
            save_ctrl_to_master(img, ctrl)
            
        except Exception as e:
            print(f"[SKIP] {os.path.basename(img)}: {e}")

TYPE_LABELS = {} 

# 외부에서 호출 가능한 메인 함수
def main_process(dir_path, n_ctrl, base_date):
    global OUTPUT_DIR, MASTER_CSV, TYPE_LABELS
    TYPE_LABELS = {} # 초기화
    
    # 동적 경로 설정
    OUTPUT_DIR = f"{base_date}/CTRL{n_ctrl}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    MASTER_CSV = os.path.join(OUTPUT_DIR, f"control_points_master_L_{base_date}.csv")
    
    # 기존 파일이 있다면 삭제
    if os.path.exists(MASTER_CSV):
        os.remove(MASTER_CSV)
        
    print(f"[Start] Count={n_ctrl}, SaveTo={OUTPUT_DIR}")
    run_open_fit_DIR(DIR_path=dir_path, n_contour_points=200, n_ctrl_points=n_ctrl, degree=3)

# if __name__ == "__main__":
    # main_process(r"C:\Test\Images", 30, "20260106")