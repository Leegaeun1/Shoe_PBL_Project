#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive Open B-spline Editor (+ Arrow-key CSV row navigator)

추가된 기능
- ←/→ : CSV '행' 기준 이전/다음으로 이동 (순환)
- ↑/↓ : 5mm 단위 사이즈 증감 (슬라이더와 동기화)
- 좌상단 상태표시: "Row i/N — size=XXXmm"
- S : 현재 컨트롤 포인트를 CSV에 append 저장 (행 목록 즉시 갱신)
- E : 현재 스플라인을 조밀 샘플로 별도 CSV 내보내기

CSV 포맷(행 단위, 공백/쉼표 혼용): size_mm, x1, y1, x2, y2, ...
"""

import os, csv, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ========================= 사용자 설정 =========================
CSV_PATH = os.path.abspath("pred_series_230_250_270.csv")   # 컨트롤 포인트 CSV 경로
EXPORT_DIR = os.path.dirname(CSV_PATH)               # 스플라인 샘플 내보내기 폴더
SIZES = list(range(230, 305, 5))                     # 230,235,...,300
DEGREE = 3                                           # B-스플라인 차수
SPLINE_SAMPLES = 400                                 # 곡선 샘플 수
PICK_RADIUS_PX = 12                                  # 마우스 픽업 반경(px)
# =============================================================

# ---------------- B-스플라인 핵심 ----------------
def open_uniform_knot_vector(n_ctrl, degree):
    knots = np.concatenate([
        np.zeros(degree + 1),
        np.arange(1, n_ctrl - degree),
        np.full(degree + 1, n_ctrl - degree),
    ])
    return knots / np.max(knots)

def bspline_basis(i, degree, knots, t):
    if degree == 0:
        is_last = (i + 1 == len(knots) - 1)
        if (knots[i] <= t < knots[i+1]) or (is_last and np.isclose(t, knots[i+1])):
            return 1.0
        return 0.0
    term1 = 0.0
    den1 = knots[i+degree] - knots[i]
    if den1 > 1e-9:
        term1 = (t - knots[i]) / den1 * bspline_basis(i, degree-1, knots, t)
    term2 = 0.0
    den2 = knots[i+degree+1] - knots[i+1]
    if den2 > 1e-9:
        term2 = (knots[i+degree+1] - t) / den2 * bspline_basis(i+1, degree-1, knots, t)
    return term1 + term2

def bspline_curve(ctrl_points, degree, knots, t_values):
    n_ctrl = len(ctrl_points)
    curve = np.zeros((len(t_values), 2), dtype=float)
    for j, t in enumerate(t_values):
        p = np.zeros(2, dtype=float)
        for i in range(n_ctrl):
            w = bspline_basis(i, degree, knots, t)
            if w > 1e-9:
                p += w * ctrl_points[i]
        curve[j] = p
    return curve
# -------------------------------------------------

# -------------------- 입출력 유틸 --------------------
def _parse_line_to_size_pts(line: str):
    """한 줄을 (size:int, pts:(N,2) ndarray)로 파싱. 실패 시 None."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    toks = [t for t in line.replace(",", " ").split() if t]
    try:
        vals = list(map(float, toks))
    except ValueError:
        return None
    if len(vals) < 3:
        return None
    size = int(round(vals[0]))
    xy = np.array(vals[1:], dtype=float)
    if len(xy) % 2 == 1:
        xy = xy[:-1]
    if xy.size == 0:
        return None
    pts = xy.reshape(-1, 2)
    return size, pts

def read_ctrl_points_rows(path):
    """CSV 전체를 '행 단위'로 읽어 [(size, pts), ...] 반환. (중복 사이즈 보존)"""
    rows = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                parsed = _parse_line_to_size_pts(raw)
                if parsed:
                    rows.append(parsed)
    return rows

def append_ctrl_points_csv(path, size_mm, pts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = pts.reshape(-1)
    row = [f"{size_mm:d}"] + [f"{v:.6f}" for v in flat]
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(row) + "\n")

def export_spline_csv(dirpath, size_mm, xy):
    os.makedirs(dirpath, exist_ok=True)
    out = os.path.join(dirpath, f"spline_size_{size_mm}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x","y"])
        for x, y in xy:
            w.writerow([f"{x:.6f}", f"{y:.6f}"])
    return out
# ----------------------------------------------------

# ------- 초기 컨트롤 포인트(없을 때) 생성 -------
def default_points():
    # 가벼운 발바닥형 곡선(열린)
    theta = np.linspace(math.pi*0.15, math.pi*1.85, 18)
    rx, ry = 60, 150
    cx, cy = 0, -150
    xs = cx + rx * np.sin(theta)
    ys = cy + ry * np.cos(theta)
    ys[:4] += np.linspace(0, -35, 4)  # 발끝 살짝 평평
    return np.stack([xs, ys], axis=1)

# -------------- 드래그 가능 위젯 --------------
class DraggablePoints:
    def __init__(self, ax, init_pts, degree=3, on_change=None):
        self.ax = ax
        self.on_change = on_change
        self.degree = degree
        self.knots = open_uniform_knot_vector(len(init_pts), degree)
        self.set_pts(init_pts, redraw=False)

        self.cid_press = ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.drag_idx = None

    def set_pts(self, pts, redraw=True):
        self.ctrl = np.array(pts, dtype=float)
        self.knots = open_uniform_knot_vector(len(self.ctrl), self.degree)
        if hasattr(self, "sc"):
            self.sc.remove(); self.poly.remove()
        self.sc = self.ax.scatter(self.ctrl[:,0], self.ctrl[:,1], s=64, zorder=5, label="Control Points")
        (self.poly,) = self.ax.plot(self.ctrl[:,0], self.ctrl[:,1], "--", lw=1.2, color="0.4", zorder=4, label="Control Polygon")
        if redraw and self.on_change:
            self.on_change(self.ctrl, self.knots)

    def current(self):
        return self.ctrl.copy(), self.knots.copy()

    def _pick(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        disp = self.ax.transData.transform(self.ctrl)
        mouse = np.array([event.x, event.y])
        d = np.hypot(disp[:,0]-mouse[0], disp[:,1]-mouse[1])
        i = int(np.argmin(d))
        return i if d[i] <= PICK_RADIUS_PX else None

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self.drag_idx = self._pick(event)

    def on_motion(self, event):
        if self.drag_idx is None or event.inaxes != self.ax: return
        if (event.xdata is None) or (event.ydata is None): return
        self.ctrl[self.drag_idx] = [event.xdata, event.ydata]
        self.sc.set_offsets(self.ctrl)
        self.poly.set_data(self.ctrl[:,0], self.ctrl[:,1])
        if self.on_change:
            self.on_change(self.ctrl, self.knots)
        self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        self.drag_idx = None

# -------------- 메인 --------------
def main():
    # 1) CSV '행' 단위로 로드
    rows = read_ctrl_points_rows(CSV_PATH)  # [(size, pts), ...]
    size2latest = {size: pts for size, pts in rows}  # 같은 size가 여러 번 있어도 마지막이 최신

    # 2) 시작 행/사이즈 결정
    if rows:
        start_row_idx = len(rows) - 1  # 마지막 행부터 보기
        start_size, start_pts = rows[start_row_idx]
    else:
        start_size = 290
        start_pts = default_points()
        rows = [(start_size, start_pts.copy())]
        size2latest[start_size] = start_pts.copy()
        start_row_idx = 0

    # 3) Figure/Axes
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    ax.set_aspect("equal", "datalim")
    ax.grid(True, alpha=0.25)
    ax.invert_yaxis()

    state = {
        "size": int(start_size),
        "curve": None,
        "curve_line": None,
        "row_idx": start_row_idx,
        "suspend_slider": False,
    }

    # 상태 텍스트 
    status_text = fig.text(0.5, 0.92, "", ha="center", va="top", fontsize=14)


    def update_status():
        status_text.set_text(f"Row {state['row_idx']+1}/{len(rows)} — size={state['size']} mm")

    # 4) 곡선 업데이트
    def update_curve(ctrl, knots):
        t = np.linspace(0, 1, SPLINE_SAMPLES, endpoint=False)
        xy = bspline_curve(ctrl, DEGREE, knots, t)
        state["curve"] = np.vstack([xy, xy[-1]])  # 오픈 곡선 표시 (마지막 점 한 번 더)
        if state["curve_line"] is None:
            (ln,) = ax.plot(state["curve"][:,0], state["curve"][:,1], "-", lw=2.2, zorder=3,
                            label=f"Open B-spline (degree={DEGREE})")
            state["curve_line"] = ln
        else:
            state["curve_line"].set_data(state["curve"][:,0], state["curve"][:,1])

    # 5) 드래그 위젯 초기화
    dp = DraggablePoints(ax, start_pts, degree=DEGREE, on_change=update_curve)
    update_curve(*dp.current())

    # 보기 범위 여유
    pts0, _ = dp.current()
    xmin, xmax = pts0[:,0].min()-40, pts0[:,0].max()+40
    ymin, ymax = pts0[:,1].min()-60, pts0[:,1].max()+60
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)

    ax.legend(loc="upper left")
    ax.set_title(f"Drag Control Points — Open B-spline (≈{start_size} mm)")
    update_status()

    # 6) 사이즈 슬라이더
    #s_ax = fig.add_axes([0.18, 0.03, 0.65, 0.03])
    #slider = Slider(s_ax, "Size (mm)", SIZES[0], SIZES[-1], valinit=int(start_size), valstep=5)

    def on_size(val):
        # 슬라이더에서 직접 바꿀 때만 동작 (화살표 네비게이션이 set_val 할 때는 무시)
        if state["suspend_slider"]:
            return
        size = int(round(val))
        state["size"] = size
        # 해당 사이즈가 존재하면 최신값, 없으면 가장 가까운 사이즈/디폴트
        if size in size2latest:
            pts = size2latest[size].copy()
        else:
            if size2latest:
                near = min(size2latest, key=lambda k: abs(k-size))
                pts = size2latest[near].copy()
            else:
                pts = default_points()
        dp.set_pts(pts, redraw=True)
        ax.set_title(f"Drag Control Points — Open B-spline (≈{size} mm)")
        update_status()
        fig.canvas.draw_idle()

    #slider.on_changed(on_size)

    # 행으로 직접 이동
    def goto_row(new_idx):
        if not rows:
            return
        new_idx %= len(rows)  # 순환
        size, pts = rows[new_idx]
        state["row_idx"] = new_idx
        state["size"] = int(size)
        # 슬라이더 값만 동기화(콜백은 잠시 막기)
        state["suspend_slider"] = True
        # try:
        #     slider.set_val(state["size"])
        # finally:
        #     state["suspend_slider"] = False
        # 해당 행의 pts로 직접 세팅 (중복 size여도 "행" 기준으로 보여주기)
        dp.set_pts(pts.copy(), redraw=True)
        ax.set_title(f"Drag Control Points — Open B-spline (≈{state['size']} mm)")
        update_status()
        fig.canvas.draw_idle()

    # 7) 단축키: 저장/내보내기/네비게이션
    def on_key(event):
        k = (event.key or "").lower()

        if k == "s":
            # 현재 컨트롤 포인트 저장 (CSV append + 메모리 갱신)
            ctrl, _ = dp.current()
            append_ctrl_points_csv(CSV_PATH, state["size"], ctrl)
            # 최신 맵/행 목록 갱신
            size2latest[state["size"]] = ctrl.copy()
            rows.append((state["size"], ctrl.copy()))
            state["row_idx"] = len(rows) - 1  # 방금 저장한 행으로 인덱스 업데이트
            print(f"[Saved] size {state['size']} → {CSV_PATH}")
            update_status()

        elif k == "e":
            if state["curve"] is not None:
                out = export_spline_csv(EXPORT_DIR, state["size"], state["curve"])
                print(f"[Exported] spline → {out}")

        elif k in ("left", "right"):
            step = -1 if k == "left" else 1
            goto_row(state["row_idx"] + step)

        elif k in ("up", "down"):
            # 5mm 단위 사이즈 증감 (슬라이더 경유)
            if state["size"] in SIZES:
                i = SIZES.index(state["size"])
            else:
                # 가장 가까운 인덱스로 스냅
                i = min(range(len(SIZES)), key=lambda j: abs(SIZES[j]-state["size"]))
            if k == "up":
                i = min(i + 1, len(SIZES) - 1)
            else:
                i = max(i - 1, 0)
            #slider.set_val(SIZES[i])  # on_size가 처리

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("조작법: ←/→=CSV 이전/다음 행, ↑/↓=사이즈 증감(5mm), 마우스 드래그=컨트롤 포인트 이동")
    print("       S=컨트롤포인트 append 저장, E=스플라인 CSV로 내보내기")
    plt.show()

if __name__ == "__main__":
    main()
