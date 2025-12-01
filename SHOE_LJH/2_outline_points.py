import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import re
from skimage import measure
import matplotlib.image as mpimg

from datetime import datetime

# --- Matplotlib 백엔드 설정 ---
preferred_backends = ["QtAgg", "Qt5Agg", "TkAgg"]
for be in preferred_backends:
    try:
        matplotlib.use(be, force=True)
        break
    except Exception:
        pass
print("Matplotlib backend:", matplotlib.get_backend())

Dir_Name = "L"
currentTime = datetime.now().strftime("%Y%m%d")

# --- 설정값 ---
N_CONTOUR_POINTS = 10 # 윤곽선 포인트 개수 (제어점 아님)

OUTPUT_DIR = "20251125/CTRL10" # 출력 디렉토리 이름 변경
os.makedirs(OUTPUT_DIR, exist_ok=True)


MASTER_CSV = os.path.join(
    OUTPUT_DIR,
    f"outline_points_master_{Dir_Name}_{currentTime}.csv" # 파일명도 변경
)


# --- 이미지 및 윤곽선 처리 함수 ---
# 1.이미지 이진 마스킹
def load_binary_outline(image_path, thresh=0.8):
    import matplotlib.image as mpimg
    try:
        img = mpimg.imread(image_path) 
    except Exception:
        return None

    if img.ndim == 3: 
        gray = img[..., :3].mean(axis=2) 
    else: 
        gray = img.astype(float)

    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)

    mask = (gray < thresh).astype(np.uint8)
    
    try:
        from scipy.ndimage import binary_dilation, binary_closing
        mask = binary_closing(mask, iterations=1)
        mask = binary_dilation(mask, iterations=1)
    except Exception: pass

    return mask 


# 2.윤곽선 추출 
def largest_contour(mask):
    from skimage import measure
    if mask is None: return None
    contours = measure.find_contours(mask, level=0.5)
    if not contours: raise RuntimeError("No contour found.")
    return np.fliplr(max(contours, key=len)) 

# 3.윤곽선을 다각형으로 만듬
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

# --- B-스플라인 관련 함수는 제어점 저장을 위해 사용되지 않으므로 제거 ---
# (DraggableCtrl, Navigator 등 GUI 클래스 및 B-스플라인 피팅 관련 함수는 윤곽선 포인트 저장에 필요 없으므로 제거했습니다.)

# --- 유틸리티 함수 ---
def bbox_minmax(points): 
    xmin, ymin = points.min(axis=0) 
    xmax, ymax = points.max(axis=0) 
    return xmin, ymin, xmax, ymax

def scale_to_mm(points, height_mm, ref_axis="y"): 
    xmin, ymin, xmax, ymax = bbox_minmax(points)
    if ref_axis.lower() == "y": 
        scale = height_mm / (ymax - ymin + 1e-9) 
    else:
        scale = height_mm / (xmax - xmin + 1e-9)
    # 원점 정렬(좌상단을 0,0) 후 스케일링
    return (points - [xmin, ymin]) * scale 

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

def parse_side_from_filename(path, default=""):
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)

    matches = list(re.finditer(r'(\d{2,3})(?!\d)', base))
    if not matches:
        return default

    m = matches[-1]
    suffix = base[m.end():] 

    tokens = re.findall(r'[^\W\d_]+', suffix, flags=re.UNICODE)
    if not tokens:
        return default

    out = "_".join(tokens)
    return out.upper()

# --- CSV 저장 함수 (윤곽선 포인트 저장용으로 수정) ---

def _ensure_master_header(path, n_points, sep=", "):
    """마스터 CSV에 헤더가 없으면 생성: type,size,side,x1,y1,...,xN,yN"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    cols = ["type", "side", "size", ] 
    for i in range(1, n_points + 1):
        cols += [f"x{i}", f"y{i}"]
    header = sep.join(cols)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")


def save_contour_to_master(image_path, contour_points, master_csv=MASTER_CSV, sep=", ", type_label=None):
    """
    하나의 마스터 CSV 파일에 윤곽선 포인트를 '추가'합니다.
    """
    n_points = contour_points.shape[0]
    _ensure_master_header(master_csv, n_points, sep=sep)

    coords = contour_points.copy()
    # 이미 scale_to_mm에서 좌상단 정렬되었지만, 다시 한번 정렬 (최솟값 기준 정렬)
    min_x, min_y = coords.min(axis=0)
    coords[:, 0] -= min_x
    coords[:, 1] -= min_y

    size_mm = parse_size_mm_from_filename(image_path, fallback=250.0)
    side = parse_side_from_filename(image_path, default="")

    # 타입 라벨 결정 (전역 맵핑 사용)
    if type_label is None:
        type_key = parse_type_from_filename(image_path, default="unknown")
        global TYPE_MAPPING
        type_label = TYPE_MAPPING.get(type_key, "unknown_unmapped")

    flat = coords.flatten()
    # 형식: type, side, size, x1,y1,...,xN,yN
    line = sep.join([type_label, side, f"{size_mm:.0f}"] + [f"{v:.6f}" for v in flat])

    with open(master_csv, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    print(f"[APPEND] -> {master_csv} (type='{type_label}', size≈{size_mm:.0f}, side='{side}', {n_points} contour pts)")

# --- 메인 실행 함수 (윤곽선 포인트 저장용으로 변경) ---

TYPE_MAPPING = {} 

def run_contour_dir_and_save(DIR_path, n_contour_points=200):
    """
    1) DIR_path 내 모든 .jpg 파일에서 Type을 추출 및 매핑
    2) 윤곽선 포인트를 추출하고 CSV에 저장
    """
    global TYPE_MAPPING
    # 파일 목록을 알파벳 순으로 가져옵니다.
    all_files = sorted(glob.glob(os.path.join(DIR_path, "*.jpg"))) 

    if not all_files:
        print(f"[ERROR] No .jpg files found in: {DIR_path}")
        return

    # 1. Type Mapping 사전 구축 (알파벳 순으로 Type00, Type01...)
    unique_types = set()
    for img in all_files:
        type_key = parse_type_from_filename(img, default="unknown")
        unique_types.add(type_key)

    sorted_unique_types = sorted(list(unique_types))
    TYPE_MAPPING = {
        original_type: f"Type{i:02d}" 
        for i, original_type in enumerate(sorted_unique_types)
    }
    print(f"\n[TYPE MAPPING] Original Name -> Mapped Type (Sorted by name): {TYPE_MAPPING}")
    

    # 2. 파일 처리 및 저장
    for img in all_files:
        try:
            type_key = parse_type_from_filename(img, default="unknown")
            type_label = TYPE_MAPPING.get(type_key, "unknown")
            
            print(f"\n[PREP] {os.path.basename(img)} (Mapped Type: {type_label})")

            mask = load_binary_outline(img)
            if mask is None: raise RuntimeError("Image load/mask failed.")
            contour = largest_contour(mask)
            contour_resampled = resample_polyline(contour, n_contour_points)

            # 윤곽선 시작점 정렬 (발가락 끝)
            toe_index = np.argmax(contour_resampled[:, 1])
            contour_rolled = np.roll(contour_resampled, toe_index, axis=0)
            
            # 여기서 중요한 점: 제어점이 아닌 윤곽선 포인트를 mm 스케일로 변환
            size_mm = parse_size_mm_from_filename(img, fallback=250.0)
            # contour_rolled (N_CONTOUR_POINTS개)를 mm 스케일로 변환
            contour_mm = scale_to_mm(contour_rolled, height_mm=size_mm, ref_axis="y")

            # CSV에 윤곽선 포인트 저장
            save_contour_to_master(img, contour_mm, type_label=type_label)

        except Exception as e:
            print(f"[SKIP] {os.path.basename(img)}: {e}")

    print(f"\n[FINAL SAVE LOCATION] Results appended to {MASTER_CSV}")


if __name__ == "__main__":
    
    DIR_PATH = r"C:\Users\user\Documents\GitHub\Shoe_PBL_Project\SHOE_LJH\output_outlines_1030\L" # <- 여기에 대상 디렉토리 경로 입력
    
    # N_CONTOUR_POINTS는 위에서 200으로 설정됨
    # N_CTRL_POINTS는 더 이상 사용되지 않음

    run_contour_dir_and_save(
        DIR_path = DIR_PATH, 
        n_contour_points=N_CONTOUR_POINTS
    )