import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob

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

Dir_Name = "R"
N_CTRL_POINTS = 70  


currentTime = datetime.now().strftime("%Y%m%d")

DIR_PATH = fr"C:\Users\user\Documents\GitHub\Shoe_PBL_Project\SHOE_LJH\output_outlines_1030\{Dir_Name}"  # <- 여기에 대상 디렉토리 경로 입력
OUTPUT_DIR = f"20251126/CTRL{N_CTRL_POINTS}" 
os.makedirs(OUTPUT_DIR, exist_ok=True)


MASTER_CSV = os.path.join(
    OUTPUT_DIR,
    f"control_points_master_{Dir_Name}_{currentTime}.csv"
)


# --- 이미지 및 윤곽선 처리 함수 ---
# 1.이미지 이진 마스킹
def load_binary_outline(image_path, thresh=0.8):
    import matplotlib.image as mpimg
    img = mpimg.imread(image_path) # GUI 띄어줄 이미지를 읽음

    if img.ndim == 3: # 색 이미지임 ㅇㅇ 
        gray = img[..., :3].mean(axis=2) # 해당 이미지  >>> 3색 평균값으로 만들고 이걸  흑백으로 변환시킴

    else: # 아님!
        gray = img.astype(float)

    # 흑백 사진에 대해서 정규화 시킴
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)

    # 흑백 사진에 대해서 이진 마스킹 진행 >> 그냥 흑/백 부분으로 나누는거 
    #thresh는 0.8로 밝기가 0.8보다 어두우면 1로 나옴 ㅇㅇ
    mask = (gray < thresh).astype(np.uint8)
    
    try:# 마스크를 좀 더 다듬는 구간 (너무 삐뚤빼뚤해서 그럼) 생각 ㄴㄴ
        from scipy.ndimage import binary_dilation, binary_closing
        mask = binary_closing(mask, iterations=1)
        mask = binary_dilation(mask, iterations=1)
    except Exception: print("Scipy not found.")

    return mask # 마스킹 완성!


# 2.윤곽선 추출 
def largest_contour(mask):
    from skimage import measure
    # 앞에서 마스킹 한거 ㅇㅇ 그거가지고 윤곽선 추출 level은 강도임 
    contours = measure.find_contours(mask, level=0.5)
    if not contours: raise RuntimeError("No contour found.")
    return np.fliplr(max(contours, key=len))  # >> 가장 큰 외곽선 선택   >> 그리고 나서 좌우 반전함 --------  contour 좌표는 (row, col) = (y, x) 순서라서 우리가 아는 x/y좌표에 너으려면 그래야한뎅

# 3.윤곽선을 다각형으로 만듬
def resample_polyline(points, n_samples):# points - 윤곽선 전체를 n_samples개의 점으로 다각형을 그림!
    P = np.vstack([points, points[0]]) ## 시작점과 끝점을 일단 연결함 >> 끊어져있을수도 있으니께

    seg_lengths = np.linalg.norm(np.diff(P, axis=0), axis=1) # 전체 변의 길이 배열.    
    # np.diff(P, axis=0) >> 이게 각 점의 벡터차이
    # np.linalg.norm >> 벡터의 norm을 구함  >> 거리임
    # cumulative_lengths = np.hstack([[0], np.cumsum(seg_lengths)])
    cumulative_lengths = np.hstack(np.cumsum(seg_lengths)) # 각 선분 길이를 누적합해서 경로를 따라 진행한 거리 누적.  >> 1,1+2, 1+2+3 ... 이렇게 되어있고, 
    total_length = cumulative_lengths[-1] ## 그럼 젤 끝에건 전체 폐곡선의 길이, 폐곡선이란... 한 곡선상에서 한 점이 한 방향으로 움직여, 출발점으로 되돌아오는 곡선.이라네여

    u = np.linspace(0, total_length, n_samples, endpoint=False) # 전체 구간에서 균등 간격의 n_samples 생성  >> 일정한 거리간격으로 생성하는거징

    # 각 샘플 u가 어느 선분 구간에 속하는지 찾아서 그 인덱스를 반환
    indices = np.searchsorted(cumulative_lengths, u, side="right") - 1
    indices = np.clip(indices, 0, len(P) - 2) # 배열 범위 조정 


    # t는 각 샘플이 속한 선분에서의 상대 위치 (0~1).  >> 이건....모르겠당
    t = (u - cumulative_lengths[indices]) / (seg_lengths[indices] + 1e-9)

    return (1 - t)[:, None] * P[indices] + t[:, None] * P[indices + 1] # 선분의 양 끝점을 선형 보간.
    # >> n_samples × 2 배열 = 등간격 아크길이로 다시 샘플링된 곡선 좌표가 결과로 출력됨!

# --- B-스플라인 핵심 함수 -------------------------------------------------------------------------------------

# --- 1.Knot Vector 생성 함수 ---
def open_uniform_knot_vector(n_ctrl, degree):
    # 스플라인에서 degree = 차수란??

    # 선형(linear, p=1): C⁰ 연속 → 꺾이는 다각형 같은 곡선
    # 이차(quadratic, p=2): C¹ 연속 → 기울기까지 연속 (부드럽지만 약간 각진 느낌)
    # 삼차(cubic, p=3): C² 연속 → 곡률까지 연속, 가장 많이 쓰임 (CAD, 그래픽스)
    # *********차수가 더 커질수록 곡선이 매끄러워지고, 제어점 변화가 더 넓게 퍼짐


    # Knot Vector란    >> 매개변수 t가 어느 구간에서 어떤 제어점 기저 함수가 활성화되는지를 결정
    # Knot Vector의 총 개수는 제어점 개수(n_ctrl) + 차수(degree) + 1 

    # 
    knots = np.concatenate([
        np.zeros(degree + 1),  # >> 곡선이 첫 제어점에서 시작하도록 보장
        np.arange(1, n_ctrl - degree),
        np.full(degree + 1, n_ctrl - degree) # >>곡선이 마지막 제어점에서 끝나도록 보장
    ])

    # 예시!

    # 제어점 5개 (n_ctrl=5), 차수 3 (degree=3)일 때:

    #knots = np.concatenate([
        # 앞부분: np.zeros(4) → [0,0,0,0]
        # 중간: np.arange(1, 2) → [1]
        # 끝부분: np.full(4, 2) → [2,2,2,2]
    #])


    # 합치면 → [0,0,0,0,1,2,2,2,2]
    # 정규화하면 [0,0,0,0,0.5,1,1,1,1]  >> 0에서 시작해서 1에서 끝나는 knot vector.

    return knots / np.max(knots) # 0과 1 사이로 정규화

# Cox-de Boor 재귀 공식을 사용한 B-스플라인 기저 함수 계산   >> 이건 정해져있는 방정식같은거임!
def bspline_basis(i, degree, knots, t):
   # 기저함수란?
   # B-스플라인 곡선은 제어점(control points)과 기저 함수(basis functions) 의 선형 결합으로 표현됨

   # 여기서는 이를 이용해서
   # 특정 점 t에서 각 컨트롤 포인트들이 얼마나 기여하는지(가중치) 를 구하는 함수


    if degree == 0:# >> 이건 차수가 0일때임 거의 직선으로 그려짐
        is_last_knot = (i + 1 == len(knots) - 1)
        if (knots[i] <= t < knots[i+1]) or (is_last_knot and np.isclose(t, knots[i+1])):
            return 1.0
        return 0.0
    
    
    # i번째 컨트롤 포인트, 왼쪽 구간에서의 기여
    term1 = 0.0
    den1 = knots[i+degree] - knots[i]
    if den1 > 1e-9: 
        term1 = (t - knots[i]) / den1 * bspline_basis(i, degree - 1, knots, t) # 차수 낮춰가면서 재귀적으로 돌림
    

    #i번째 컨트롤 포인트, 오른쪽 구간에서의 기여
    term2 = 0.0
    den2 = knots[i+degree+1] - knots[i+1]
    if den2 > 1e-9: 
        term2 = (knots[i+degree+1] - t) / den2 * bspline_basis(i + 1, degree - 1, knots, t) # 차수 낮춰가면서 재귀적으로 돌림
        
    # 각 지점 기여도 조사끝!
    return term1 + term2

# ctrl_points + 차수(degree) + 노드 벡터(knots) + 매개변수 값들(t_values)을 받아서, 곡선 위의 좌표들을 계산
def bspline_curve(ctrl_points, degree, knots, t_values):
    """B-스플라인 곡선 계산"""
    n_ctrl = len(ctrl_points)
    curve = np.zeros((len(t_values), ctrl_points.shape[1]))
    
    
    for j, t in enumerate(t_values):
        point = np.zeros(ctrl_points.shape[1])
        for i in range(n_ctrl):
            w = bspline_basis(i, degree, knots, t)
            if w > 1e-9: 
                point += w * ctrl_points[i]
        curve[j] = point
    return curve

# --- 메인 B-스플라인을 위한 피팅 함수 -------------------------------------------------------
def fit_open_bspline_least_squares(points, n_ctrl=20, degree=3, lam=1e-5):
    """'열린' B-스플라인을 데이터 점들에 최소자승법으로 피팅합니다."""

    M = len(points) # 아까 만든 다각형 ㅇㅇ 점 갯수

    knots = open_uniform_knot_vector(n_ctrl, degree) # knot 벡터

    
    # 0과 1 사이에서 M개의 균등 간격 숫자 생성   // endpoint =True >> 1을 포함함!
    ts = np.linspace(0, 1, M, endpoint=True)

    A = np.zeros((M, n_ctrl)) # 초기화된 n_ctrl갯수 행렬


    for j, t in enumerate(ts):# 데이터 점 × 제어점 크기의 기저함수 행렬 만들기,  
        for i in range(n_ctrl):
            A[j, i] = bspline_basis(i, degree, knots, t)  #> > i번째 컨트롤 포인트가 t점(윤곽선 위 점)에 대해 미치는 영향 계산 


    # 정규화(Regularization) 행렬 D >> 스무딩(regularization) 을 위해 >> 제어점이 많아지면, 너무 울룩블룩해짐, 이걸 방지하게끔 하는 부분
    if lam > 0:
        # np.eye(n_ctrl, k=0) >>> 대각선이 1인 단위 행렬, np.eye(n_ctrl, k=1) >> 위 대각선,  np.eye(n_ctrl, k=-1) >> 아래대각선
        D = np.eye(n_ctrl, k=0) * -2 + np.eye(n_ctrl, k=1) * 1 + np.eye(n_ctrl, k=-1) * 1
        
        # 
        D = D[1:-1] #  곡선 시작·끝을 강제로 지나야 하므로, 양 끝점은 제외해서 조정 X

        # >> 보통 최소자승 정규방정식  lam이 클수록 더 매끈한 곡선이 됨 
        ATA = A.T @ A + lam * (D.T @ D) # @ => 행렬 곱 연산자
    else:
        ATA = A.T @ A
    

    # 최소자승법으로 컨트롤 포인트를 풀어버림
    ATYx = A.T @ points[:, 0] # X좌표용 근사
    ATYy = A.T @ points[:, 1] # Y좌표용 근사

    # np.linalg.solve >> 선형 방정식을 푸는 데 사용되는 NumPy의 함수, 주어진 선형 방정식의 행렬 형태의 계수와 상수 벡터를 입력으로 받아, 선형 방정식의 해를 구함
    ctrl_x = np.linalg.solve(ATA + 1e-9 * np.eye(n_ctrl), ATYx)# 컨트롤포인트들의 x좌표 
    ctrl_y = np.linalg.solve(ATA + 1e-9 * np.eye(n_ctrl), ATYy)# 컨트롤포인트들의 y좌표 

    return np.stack([ctrl_x, ctrl_y], axis=1), knots # 제어점 x좌표와 y좌표를 합쳐서 (x,y) 쌍으로 만듦.

class DraggableCtrl:
    def __init__(self, entry, degree, pick_radius_px=12):
        """
        entry = {
          'image_path': str,
          'contour_mm': (N,2) np.array,
          'ctrl': (K,2) np.array
        }
        """
        self.degree = degree
        self.pick_radius = pick_radius_px

        self.fig, self.ax = plt.subplots(figsize=(8, 10))
        self.bg = entry['contour_mm']
        self.image_path = entry['image_path']
        self.ctrl = entry['ctrl'].copy()

        # 보기 안정화를 위해 첫 점 기준 오프셋 제거
        offset = self.ctrl[0].copy()
        self.ctrl -= offset
        bg = self.bg - offset

        self.knots = open_uniform_knot_vector(len(self.ctrl), self.degree)

        self.ax.plot(bg[:, 0], bg[:, 1], color="k", lw=1.2, alpha=0.8, label="Original Contour")
        self.scat = self.ax.scatter(self.ctrl[:, 0], self.ctrl[:, 1], s=60, zorder=4, label="Control Points")
        (self.line_ctrl,) = self.ax.plot([], [], "--", color="grey", zorder=3, label="Control Polygon")
        (self.curve_line,) = self.ax.plot([], [], linewidth=2.5, zorder=2, color="#d9534f",
                                          label=f"Open B-spline (degree={self.degree})")

        self.ax.set_aspect("equal", "box")
        self.ax.grid(True, alpha=0.3)
        self.ax.invert_yaxis()

        # 범위 세팅
        all_pts = np.vstack([self.ctrl, bg])
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        pad_x = 0.6 * (xmax - xmin + 1e-9)
        pad_y = 0.2 * (ymax - ymin + 1e-9)
        self.ax.set_xlim(xmin - pad_x, xmax + pad_x)
        self.ax.set_ylim(ymax + pad_y, ymin - pad_y)

        size_mm = parse_size_mm_from_filename(self.image_path, fallback=250.0)
        self.ax.set_title(f"[{os.path.basename(self.image_path)}] Drag Control Points — Real-size (≈{size_mm:.0f} mm)")

        self.update_curve(rescale=False)
        self.ax.legend(loc="best")

        self.dragging_idx = None
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        #self.fig.canvas.mpl_connect("close_event", self.on_close)

    def eval_curve(self, n=400):
        t = np.linspace(0, 1, n, endpoint=False)
        C = bspline_curve(self.ctrl, self.degree, self.knots, t)
        return np.vstack([C, C[0]])

    def update_curve(self, rescale=False):
        curve = self.eval_curve()
        self.curve_line.set_data(curve[:, 0], curve[:, 1])
        self.scat.set_offsets(self.ctrl)
        self.line_ctrl.set_data(self.ctrl[:, 0], self.ctrl[:, 1])
        if rescale:
            self.ax.relim(); self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

    def pick_point(self, event):
        if event.xdata is None:
            return None
        ctrl_disp = self.ax.transData.transform(self.ctrl)
        mouse = np.array([event.x, event.y])
        dists = np.hypot(ctrl_disp[:, 0] - mouse[0], ctrl_disp[:, 1] - mouse[1])
        i = np.argmin(dists)
        return i if dists[i] <= self.pick_radius else None

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.dragging_idx = self.pick_point(event)

    def on_release(self, event):
        self.dragging_idx = None

    def on_motion(self, event):
        if self.dragging_idx is None or event.inaxes != self.ax:
            return
        self.ctrl[self.dragging_idx, 0] = float(event.xdata)
        self.ctrl[self.dragging_idx, 1] = float(event.ydata)
        self.update_curve()

    def on_key(self, event):
        key = (event.key or "").lower()
        if key == "s":
            # 현재 파일 덮어쓰기 저장
            self.save_current()
        # 방향키는 Navigator가 관리하므로 여기서는 패스

    # def on_close(self, event):
    #     # 창 닫을 때 자동 저장
    #     try:
    #         self.save_current()
    #     except Exception as e:
    #         print(f"[WARN] auto-save failed: {e}")

    def save_current(self):
        tkey = parse_type_from_filename(self.image_path, default="unknown")
        tlabel = get_or_assign_type_label(tkey)
        save_ctrl_to_master(self.image_path, self.ctrl, type_label=tlabel)

    # 외부에서 이미지/데이터를 갱신할 때 쓸 메서드
    def load_entry(self, entry):
        self.bg = entry['contour_mm']
        self.image_path = entry['image_path']
        self.ctrl = entry['ctrl'].copy()

        offset = self.ctrl[0].copy()
        self.ctrl -= offset
        bg = self.bg - offset

        self.knots = open_uniform_knot_vector(len(self.ctrl), self.degree)

        self.ax.clear()
        self.ax.plot(bg[:, 0], bg[:, 1], color="k", lw=1.2, alpha=0.8, label="Original Contour")
        self.scat = self.ax.scatter(self.ctrl[:, 0], self.ctrl[:, 1], s=60, zorder=4, label="Control Points")
        (self.line_ctrl,) = self.ax.plot([], [], "--", color="grey", zorder=3, label="Control Polygon")
        (self.curve_line,) = self.ax.plot([], [], linewidth=2.5, zorder=2, color="#d9534f",
                                          label=f"Open B-spline (degree={self.degree})")
        self.ax.set_aspect("equal", "box")
        self.ax.grid(True, alpha=0.3)
        self.ax.invert_yaxis()

        all_pts = np.vstack([self.ctrl, bg])
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        pad_x = 0.6 * (xmax - xmin + 1e-9)
        pad_y = 0.2 * (ymax - ymin + 1e-9)
        self.ax.set_xlim(xmin - pad_x, xmax + pad_x)
        self.ax.set_ylim(ymax + pad_y, ymin - pad_y)

        size_mm = parse_size_mm_from_filename(self.image_path, fallback=250.0)
        self.ax.set_title(f"[{os.path.basename(self.image_path)}] Drag Control Points — Real-size (≈{size_mm:.0f} mm)")
        self.update_curve(rescale=False)
        self.ax.legend(loc="best")

class Navigator:
    """
    좌/우 화살표로 파일 탐색, 각 파일 내용 저장 관리.
    """
    def __init__(self, entries, degree, pick_radius_px=12):
        assert len(entries) > 0, "No entries to browse."
        self.entries = entries
        self.idx = 0
        self.viewer = DraggableCtrl(entries[0], degree=degree, pick_radius_px=pick_radius_px)
        self.cid = self.viewer.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_status()

    def on_key(self, event):
        key = (event.key or "").lower()
        if key in ("right",):
            self.next()
        elif key in ("left",):
            self.prev()

    def next(self):
        # 현재 파일 먼저 저장
        #self.viewer.save_current()
        self.idx = (self.idx + 1) % len(self.entries)
        self.viewer.load_entry(self.entries[self.idx])
        self.update_status()

    def prev(self):
        #self.viewer.save_current()
        self.idx = (self.idx - 1) % len(self.entries)
        self.viewer.load_entry(self.entries[self.idx])
        self.update_status()

    def update_status(self):
        print(f"[BROWSE] {self.idx+1}/{len(self.entries)} -> {os.path.basename(self.entries[self.idx]['image_path'])}")

import re

def bbox_minmax(points): # 바운딩박스 만듬 --> 끝부분 기준으로 네모난 박스를 만드는것 
    xmin, ymin = points.min(axis=0) # 좌우 끝 
    xmax, ymax = points.max(axis=0) # 위아래 끝
    return xmin, ymin, xmax, ymax

def scale_to_mm(points, height_mm, ref_axis="y"): # 픽셀 좌표 -> 실제 mm단위로 변환 
    xmin, ymin, xmax, ymax = bbox_minmax(points)
    if ref_axis.lower() == "y": # 높이 기준
        scale = height_mm / (ymax - ymin + 1e-9) # 목표mm/실제 픽셀 길이. 1e-9는 0으로 나누기 방지
    else:
        scale = height_mm / (xmax - xmin + 1e-9)
    return (points - [xmin, ymin]) * scale # 원점 정렬(좌상단을 0,0)

def parse_size_mm_from_filename(path, fallback=250.0):
    m = re.search(r'(\d{2,3})(?=[^\d]|$)', os.path.basename(path)) # 2~3자리 숫자를 캡쳐. 230~290, 숫자 다음에 숫자가 아닌 문자/문자열 끝이 와야함
    return float(m.group(1)) if m else float(fallback) # 못찾으면 250반환함, 숫자반환


def _ensure_master_header(path, n_ctrl, sep=", "):
    """마스터 CSV에 헤더가 없으면 생성: type,size,side,x1,y1,...,xN,yN"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    cols = ["type", "side", "size", ]  # ← type을 맨 앞에 추가
    for i in range(1, n_ctrl + 1):
        cols += [f"x{i}", f"y{i}"]
    header = sep.join(cols)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")


def save_ctrl_to_master(image_path, ctrl_points, master_csv=MASTER_CSV, sep=", ", type_label=None):
    """
    하나의 마스터 CSV 파일에 행을 '추가'합니다.
    형식:
      type, size, side, x1,y1,...,xN,yN
    """
    n_ctrl = ctrl_points.shape[0]
    _ensure_master_header(master_csv, n_ctrl, sep=sep)

    coords = ctrl_points.copy()
    # 좌상단(최솟값) 기준 정렬 (기존 유지)
    min_x, min_y = coords.min(axis=0)
    coords[:, 0] -= min_x
    coords[:, 1] -= min_y

    size_mm = parse_size_mm_from_filename(image_path, fallback=250.0)
    side = parse_side_from_filename(image_path, default="")

    # 타입 라벨 결정
    if type_label is None:
        type_key = parse_type_from_filename(image_path, default="unknown")
        type_label = get_or_assign_type_label(type_key)

    flat = coords.flatten()
    # 맨 앞에 type 추가
    line = sep.join([type_label, side, f"{size_mm:.0f}"] + [f"{v:.6f}" for v in flat])

    with open(master_csv, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    print(f"[APPEND] -> {master_csv}  (type='{type_label}', size≈{size_mm:.0f}, side='{side}', {n_ctrl} ctrl pts)")


# --- MODIFIED: 메인 실행 함수 ---
def run_open_fit(image_path, n_contour_points=200, n_ctrl_points=25):
    print("1. Loading image and extracting contour...")
    mask = load_binary_outline(image_path) # 이진 마스킹
    contour = largest_contour(mask) # 윤곽선 추출 
    
    print(f"2. Resampling contour to {n_contour_points} points...")
    contour_resampled = resample_polyline(contour, n_contour_points) # 윤곽선 전체를 n_samples개의 점으로 만들수 있는 다각형을 그림!

    print("3. Re-ordering contour data to start and end at the toe...")
    # --- 데이터 재구성 (핵심) ---
    # 발가락 끝점(y값이 가장 큰 점)의 인덱스를 찾음 
    toe_index = np.argmax(contour_resampled[:, 1])
    # np.argmax >> rkwkd zms dnjsthdml dlseprtm qksghks
    # [:,0] → x좌표 (좌우 방향)
    # [:,1] → y좌표 (상하 방향)

    # 발가락 끝점이 배열의 시작이 되도록 데이터를 회전시킴    >> 동작은 다음과 같이 -- [shift = toe_index → 배열을 앞으로 toe_index만큼 밀어줌.]
    contour_rolled = np.roll(contour_resampled, toe_index, axis=0)
    # 시작점을 마지막에 추가하여 데이터를 [toe, ..., toe] 형태로 만듦  >> 완전히 닫기!
    contour_path = np.vstack([contour_rolled, contour_rolled[0]])

    # ★ 추가: 파일명에서 사이즈(mm) 읽어서 mm 스케일로 변환
    size_mm = parse_size_mm_from_filename(image_path, fallback=250.0)
    contour_mm = scale_to_mm(contour_path, height_mm=size_mm, ref_axis="y")

    # 저장 파일명 베이스: 예) "C:/.../230Size2_ctrl_mm"
    save_stem = os.path.splitext(image_path)[0] + "_ctrl_mm"

    print(f"4. Fitting an open B-spline with {n_ctrl_points} control points (mm space)...")
    # 제어점을 사용하여 B-스플라인을 맞춤
    initial_ctrl_points, _ = fit_open_bspline_least_squares(
        contour_mm, n_ctrl=n_ctrl_points, degree=DEGREE
    )

    print("5. Launching interactive editor (mm mode)...")
    # 마지막 GUI창 구성
    app = DraggableCtrl(initial_ctrl_points, background_contour=contour_mm,save_stem=save_stem,)
    app.ax.set_title(f"Drag Control Points to Edit the Spline (Open) — Real-size (≈{size_mm:.0f} mm)")
    plt.show()


def run_open_fit_DIR(DIR_path, n_contour_points=200, n_ctrl_points=25):
    """
    1) DIR_path 내 모든 .jpg 파일에서 컨트롤 포인트 추출
    2) 추출 즉시 CSV 저장 (파일별 덮어쓰기)
    3) GUI에서 좌/우 키로 파일 전환 가능
    4) 수정 후 저장(s 키) 또는 전환/종료 시 자동 저장
    """
    jpgs = sorted(glob.glob(os.path.join(DIR_path, "*.jpg")))
    if not jpgs:
        print(f"[ERROR] No .jpg files found in: {DIR_path}")
        return

    entries = []
    for img in jpgs:
        try:
            print(f"\n[PREP] {os.path.basename(img)}")
            mask = load_binary_outline(img)
            contour = largest_contour(mask)
            contour_resampled = resample_polyline(contour, n_contour_points)

            toe_index = np.argmax(contour_resampled[:, 1])
            contour_rolled = np.roll(contour_resampled, toe_index, axis=0)
            contour_path = np.vstack([contour_rolled, contour_rolled[0]])

            size_mm = parse_size_mm_from_filename(img, fallback=250.0)
            contour_mm = scale_to_mm(contour_path, height_mm=size_mm, ref_axis="y")

            ctrl, _ = fit_open_bspline_least_squares(
                contour_mm, n_ctrl=n_ctrl_points, degree=DEGREE
            )

            # (2) 즉시 저장
            save_ctrl_to_master(img, ctrl)

            # GUI 탐색용 엔트리 보관
            entries.append({"image_path": img, "contour_mm": contour_mm, "ctrl": ctrl})
        except Exception as e:
            print(f"[SKIP] {os.path.basename(img)}: {e}")

    if not entries:
        print("[ERROR] No valid entries to show.")
        return

    print("\n[GUI] Use ← / → to switch files. Press 's' to save current.")
    nav = Navigator(entries, degree=DEGREE, pick_radius_px=PICK_RADIUS_PX)
    plt.show()


import os, re

TYPE_LABELS = {} 

def parse_type_from_filename(path, default="unknown"):
    """
    파일명에서 '타입_사이즈(숫자)측면.jpg' 형태의 '타입'을 추출.
    예: 'adidas_230L.jpg' -> 'adidas'
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    # 첫 '_' 이전을 타입으로 간주
    if "_" in name:
        t = name.split("_", 1)[0].strip()
        return t if t else default
    return default

def get_or_assign_type_label(type_key):
    """
    전역 TYPE_LABELS에서 라벨을 얻거나 새로 배정.
    """
    if type_key not in TYPE_LABELS:
        TYPE_LABELS[type_key] = f"Type{len(TYPE_LABELS):02d}"
    return TYPE_LABELS[type_key]

def parse_side_from_filename(path, default=""):
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)

    # base에서 '마지막 2~3자리 숫자(뒤에 또 숫자 없는)' 위치 찾기
    matches = list(re.finditer(r'(\d{2,3})(?!\d)', base))
    if not matches:
        return default

    m = matches[-1]
    suffix = base[m.end():]  # 사이즈 숫자 뒤부터 확장자 앞까지

    # 숫자 뒤 구간에서 '문자(한글/영문 등)만'을 단어로 인식해서 모두 수집
    # [^\W\d_]+  == 글자(letter)만 (숫자/언더스코어/공백/기호 제외), 유니코드 대응
    tokens = re.findall(r'[^\W\d_]+', suffix, flags=re.UNICODE)
    if not tokens:
        return default

    # 여러 단어면 '_'로 연결. 영문은 대문자 처리(한글은 영향 없음)
    out = "_".join(tokens)
    return out.upper()


if __name__ == "__main__":
    # --- MODIFIED: '열린' B-스플라인을 위한 Interactive Editor ---
    DEGREE = 3
    PICK_RADIUS_PX = 12


    #IMAGE_FILE = "260LB.jpg"
    N_CONTOUR_POINTS = 200



    # if not os.path.exists(IMAGE_FILE):
    #     print(f"Error: Image file not found at '{IMAGE_FILE}'")
    # else:
    #     # run_open_fit(
    #     #     image_path=IMAGE_FILE, 
    #     #     n_contour_points=N_CONTOUR_POINTS, 
    #     #     n_ctrl_points=N_CTRL_POINTS
    #     # )


        
    run_open_fit_DIR(DIR_path = DIR_PATH, 
                        n_contour_points=N_CONTOUR_POINTS, 
                        n_ctrl_points=N_CTRL_POINTS)