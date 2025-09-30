#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive Open B-spline Editor (control_point_v2.py 스타일)
- 230~290mm (10mm 간격) 사이즈용 컨트롤 포인트를 바탕으로 '정통 B-스플라인' 윤곽선을 표시.
- 점(컨트롤 포인트)을 드래그하면 즉시 곡선 갱신.
- 상단 슬라이더로 사이즈 전환.
- 단축키:  S=현재 사이즈 컨트롤 포인트 CSV에 append 저장
           E=현재 스플라인을 조밀 샘플로 별도 CSV 내보내기
- CSV 포맷(행 단위, 공백/쉼표 모두 허용): size_mm, x1, y1, x2, y2, ...
"""

import os, csv, math, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ========================= 사용자 설정 =========================
CSV_PATH = os.path.abspath("ctrl_points_pred.csv")  # 컨트롤 포인트 저장/불러오기 경로
EXPORT_DIR = os.path.dirname(CSV_PATH)                    # 스플라인 샘플 내보내기 폴더
SIZES = list(range(230, 305, 5))                          # 230,235,...,290
DEGREE = 3                                                # B-스플라인 차수
SPLINE_SAMPLES = 400                                      # 곡선 샘플 수
PICK_RADIUS_PX = 12                                       # 마우스 픽업 반경(px)
# =============================================================

# ---------------- B-스플라인 핵심 (control_point_v2.py 방식) ----------------
# Open uniform knot vector
def open_uniform_knot_vector(n_ctrl, degree):
    knots = np.concatenate([
        np.zeros(degree + 1),
        np.arange(1, n_ctrl - degree),
        np.full(degree + 1, n_ctrl - degree)
    ])
    return knots / np.max(knots)

# Cox–de Boor B-spline basis
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

# Evaluate open B-spline curve given control points
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
# --------------------------------------------------------------------------

# -------------------- 입출력 유틸 --------------------
def read_ctrl_points_csv(path):
    """CSV에서 {size: (N,2) ndarray} 로 읽기. 공백/쉼표 혼용 지원."""
    data = {}
    if not os.path.isfile(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"): 
                continue
            toks = [t for t in line.replace(",", " ").split() if t]
            try:
                vals = list(map(float, toks))
            except ValueError:
                continue
            if len(vals) < 3:
                continue
            size = int(round(vals[0]))
            xy = np.array(vals[1:], dtype=float)
            if len(xy) % 2 == 1:
                xy = xy[:-1]
            data[size] = xy.reshape(-1, 2)
    return data

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
    # 1) 사이즈별 컨트롤 포인트 로드
    size2pts = read_ctrl_points_csv(CSV_PATH)

    # 2) 시작 사이즈 결정
    start = 290 if 290 in size2pts else (max(size2pts) if size2pts else 290)
    if start not in size2pts:
        size2pts[start] = default_points()

    # 3) Figure/Axes
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    ax.set_aspect("equal", "datalim")
    ax.grid(True, alpha=0.25)
    ax.invert_yaxis()

    state = {"size": start, "curve": None, "curve_line": None}

    # 4) 곡선 업데이트
    def update_curve(ctrl, knots):
        t = np.linspace(0, 1, SPLINE_SAMPLES, endpoint=False)
        xy = bspline_curve(ctrl, DEGREE, knots, t)
        state["curve"] = np.vstack([xy, xy[-1]])  # 오픈 곡선 표시
        if state["curve_line"] is None:
            (ln,) = ax.plot(state["curve"][:,0], state["curve"][:,1], "-", lw=2.2, zorder=3,
                            label=f"Open B-spline (degree={DEGREE})")
            state["curve_line"] = ln
        else:
            state["curve_line"].set_data(state["curve"][:,0], state["curve"][:,1])

    # 5) 드래그 위젯
    dp = DraggablePoints(ax, size2pts[start], degree=DEGREE, on_change=update_curve)
    update_curve(*dp.current())

    # 범위 여유
    pts0, _ = dp.current()
    xmin, xmax = pts0[:,0].min()-40, pts0[:,0].max()+40
    ymin, ymax = pts0[:,1].min()-60, pts0[:,1].max()+60
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)

    ax.legend(loc="upper left")
    ax.set_title(f"Drag Control Points to Edit the Spline (Open) — Real-size (≈{start} mm)")

    # 6) 사이즈 슬라이더
    s_ax = fig.add_axes([0.18, 0.03, 0.65, 0.03])
    slider = Slider(s_ax, "Size (mm)", SIZES[0], SIZES[-1], valinit=start, valstep=5)

    def on_size(val):
        size = int(round(val))
        state["size"] = size
        if size in size2pts:
            pts = size2pts[size].copy()
        else:
            # 가장 가까운 사이즈 복사
            if size2pts:
                near = min(size2pts, key=lambda k: abs(k-size))
                pts = size2pts[near].copy()
            else:
                pts = default_points()
        dp.set_pts(pts, redraw=True)
        ax.set_title(f"Drag Control Points to Edit the Spline (Open) — Real-size (≈{size} mm)")
        fig.canvas.draw_idle()

    slider.on_changed(on_size)

    # 7) 단축키: 저장/내보내기
    def on_key(event):
        k = (event.key or "").lower()
        if k == "s":
            ctrl, _ = dp.current()
            size2pts[state["size"]] = ctrl.copy()
            append_ctrl_points_csv(CSV_PATH, state["size"], ctrl)
            print(f"[Saved] size {state['size']} → {CSV_PATH}")
        elif k == "e":
            if state["curve"] is None: return
            out = export_spline_csv(EXPORT_DIR, state["size"], state["curve"])
            print(f"[Exported] spline → {out}")

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("조작법: 점 드래그, 슬라이더로 사이즈 변경, S=컨트롤포인트 저장, E=스플라인 내보내기")
    plt.show()

if __name__ == "__main__":
    main()
