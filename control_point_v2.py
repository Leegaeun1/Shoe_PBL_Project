import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --- Matplotlib 백엔드 설정 ---
preferred_backends = ["QtAgg", "Qt5Agg", "TkAgg"]
for be in preferred_backends:
    try:
        matplotlib.use(be, force=True)
        break
    except Exception:
        pass
print("Matplotlib backend:", matplotlib.get_backend())




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
    def __init__(self, ctrl_points, background_contour=None, save_stem=None, title_suffix=""):
        self.ctrl = ctrl_points.copy()
        self.knots = open_uniform_knot_vector(len(self.ctrl), DEGREE)
        self.fig, self.ax = plt.subplots(figsize=(8, 10))

        # 저장 베이스 경로 (예: ".../230Size2_ctrl_mm")
        self.save_stem = save_stem

        # 보기 안정화를 위해 첫 컨트롤 포인트를 원점으로 이동
        offset = self.ctrl[0].copy()
        self.ctrl = self.ctrl - offset
        bg = None
        if background_contour is not None:
            bg = background_contour - offset
            self.ax.plot(bg[:, 0], bg[:, 1], color="k", lw=1.2, alpha=0.8, label="Original Contour")

        self.scat = self.ax.scatter(self.ctrl[:, 0], self.ctrl[:, 1], s=60, zorder=4, label="Control Points")
        (self.line_ctrl,) = self.ax.plot([], [], "--", color="grey", zorder=3, label="Control Polygon")
        (self.curve_line,) = self.ax.plot([], [], linewidth=2.5, zorder=2, color="#d9534f",
                                          label=f"Open B-spline (degree={DEGREE})")

        self.ax.set_title(f"Drag Control Points to Edit the Spline (Open) — Real-size (mm) {title_suffix}")
        self.ax.set_aspect("equal", "box")
        self.ax.grid(True, alpha=0.3)
        self.ax.invert_yaxis()
        # ★ (B) 데이터 기반으로 축 범위 크게 잡기 (패딩 포함)
        #     - 배경 윤곽선이 있으면 같이 고려
        all_pts = self.ctrl if bg is None else np.vstack([self.ctrl, bg])
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        pad_x = 0.6 * (xmax - xmin + 1e-9)
        pad_y = 0.2 * (ymax - ymin + 1e-9)
        self.ax.set_xlim(xmin - pad_x, xmax + pad_x)
         # invert_yaxis()를 썼으니, y축은 (ymax+pad) -> (ymin-pad)의 순서로 그려야 화면이 “아래로 증가”
        self.ax.set_ylim(ymax + pad_y, ymin - pad_y)

        self.update_curve(rescale=False)
        self.ax.legend(loc="best")

        self.dragging_idx = None
        # 이벤트 연결(드래그)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        # ★ 저장 단축키 & 창 닫힘(auto-save)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def eval_curve(self, n=400):# 스플라인 그리기  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 이것만 보기
        t = np.linspace(0, 1, n, endpoint=False)
        C = bspline_curve(self.ctrl, DEGREE, self.knots, t)
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
        if event.xdata is None: return None
        ctrl_disp = self.ax.transData.transform(self.ctrl)
        mouse = np.array([event.x, event.y])
        dists = np.hypot(ctrl_disp[:, 0] - mouse[0], ctrl_disp[:, 1] - mouse[1])
        i = np.argmin(dists)
        return i if dists[i] <= PICK_RADIUS_PX else None

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self.dragging_idx = self.pick_point(event)

    def on_release(self, event):
        self.dragging_idx = None

    def on_motion(self, event):
        if self.dragging_idx is None or event.inaxes != self.ax: return
        self.ctrl[self.dragging_idx, 0] = float(event.xdata)
        self.ctrl[self.dragging_idx, 1] = float(event.ydata)
        self.update_curve()

    # ---------- 저장 관련 ----------
    def _ensure_stem(self):
        # 저장 경로 베이스가 없으면 임시 이름 부여
        if not self.save_stem:
            self.save_stem = os.path.join(os.getcwd(), "control_points_mm")
    def _default_save_path(self):
        # 전역 IMAGE_FILE을 이용해 저장 파일명 구성: <원본이름>_ctrl_points.csv
        base, _ = os.path.splitext(IMAGE_FILE)
        return "ctrl_points.csv"
    
    def save_csv(self, path=None, sep=","):
        """
        첫 행: x1 y1 x2 y2 ...
        둘째 행: <값들이 같은 구분자(sep)로> 한 줄
        """
        path = path or self._default_save_path()
        n = len(self.ctrl)
        # 헤더 만들기 (x1 y1 x2 y2 ...)
        # cols = []
        # for i in range(n):
        #     cols.append(f"x{i+1}")
        #     cols.append(f"y{i+1}")
        # header_line = sep.join(cols)

        # 값(한 줄) 만들기
        coords = self.ctrl.copy()
        min_x, min_y = coords.min(axis=0)
        coords[:, 0] -= min_x
        coords[:, 1] -= min_y
        size_mm = parse_size_mm_from_filename(IMAGE_FILE, fallback=250.0)

        flat = coords.flatten()
        value_line = sep.join(f"{v:.6f}" for v in flat)


        # 파일로 쓰기
        with open(path, "a", encoding="utf-8") as f:
            #f.write(header_line + "\n")
            f.write(f"{size_mm:.0f}, {value_line}\n")

        print(f"[SAVE] Control points saved to {path} (separator='{sep}')")

    def on_key(self, event):
        key = (event.key or "").lower()
        if key == "s":
            self.save_csv()   # CSV 저장

    def on_close(self, event):
        # 창 닫을 때 마지막 상태를 CSV로 자동 저장(덮어쓰기)
        try:
            self.save_csv()
        except Exception as e:
            print(f"[WARN] auto-save failed: {e}")


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


if __name__ == "__main__":
    # --- MODIFIED: '열린' B-스플라인을 위한 Interactive Editor ---
    DEGREE = 3
    PICK_RADIUS_PX = 12

    IMAGE_FILE = "290RB.jpg"
    N_CONTOUR_POINTS = 200
    N_CTRL_POINTS = 25

    '''if not os.path.exists(IMAGE_FILE):
        print(f"Error: Image file not found at '{IMAGE_FILE}'")
    else:'''
    run_open_fit(
            image_path=IMAGE_FILE, 
            n_contour_points=N_CONTOUR_POINTS, 
            n_ctrl_points=N_CTRL_POINTS
        )