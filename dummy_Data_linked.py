#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dummy_Data_linked.py — heel-last enforced, robust heel-align, optional --no_resample
- Always place the heel as the LAST vertex (for both 220 & 270)
- Always heel-align BOTH curves to y=0 and same heel x
- Supports 'positive_down' plotting (screen-down is +y) without flipping shapes
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

# ----------------- I/O helpers ----------------- #
def row_to_xy(row, positive_down: bool) -> np.ndarray:
    """Read one row (x1..x40,y1..y40, Size) into Nx2.
    If positive_down=False, convert y -> y - Size (range ~ [-Size,0]).
    If positive_down=True, trust CSV's y as-is (already negative-down in your data)."""
    xs = [row.get(f"x{i}") for i in range(1, 41)]
    ys = [row.get(f"y{i}") for i in range(1, 41)]
    size = float(row.get("Size", 0))
    ys = [y - size for y in ys] if not positive_down else ys
    return np.column_stack([xs, ys]).astype(float)

def xy_to_row_dict(pts: np.ndarray, size_mm: int | float, extra: dict | None = None) -> dict:
    d = {}
    for i in range(1, 41):
        d[f"x{i}"] = float(pts[i-1, 0])
        d[f"y{i}"] = float(pts[i-1, 1])
    d["Size"] = size_mm
    if extra:
        d.update(extra)
    return d

# ----------------- Geometry utils ----------------- #
# --- replace heel_index() with width-aware auto detection ---
def heel_index(pts: np.ndarray, positive_down: bool) -> int:
    y = pts[:,1]
    i_min, i_max = int(np.argmin(y)), int(np.argmax(y))

    def local_width(i, w=4):
        # 이웃 일부 점들의 x 분산(폭)으로 뒤꿈치/발가락 판별
        N = len(pts)
        idx = [(i+j) % N for j in range(-w, w+1)]
        return float(np.std(pts[idx,0]))

    return i_min if local_width(i_min) < local_width(i_max) else i_max


def rotate_last_is_index(pts: np.ndarray, last_idx: int) -> np.ndarray:
    if last_idx == len(pts) - 1:
        return pts.copy()
    return np.vstack([pts[last_idx + 1 :], pts[: last_idx + 1]])

def closed_bspline_resample(pts: np.ndarray, n_out: int = 40, s: float = 8.0) -> np.ndarray:
    """Closed B-spline smoothing + resample to n_out points (no duplicate)."""
    x, y = pts[:, 0], pts[:, 1]
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]
    tck, _ = interpolate.splprep([x, y], s=s, per=True)
    u = np.linspace(0, 1, n_out, endpoint=False)
    xf, yf = interpolate.splev(u, tck)
    return np.column_stack([xf, yf])

def heel_align(pts, set_x=None):
    """Snap the heel (last vertex) to y=0, and optionally to a given x."""
    out = pts.copy()
    heel = out[-1]
    target_y = 0.0                 # 항상 뒤꿈치 y=0
    out[:, 1] += target_y - heel[1]
    if set_x is not None:
        out[:, 0] += set_x - out[-1, 0]
    return out

def ensure_heel_last(pts: np.ndarray, positive_down: bool) -> np.ndarray:
    """Rotate so that the heel point becomes the LAST vertex."""
    idx = heel_index(pts, positive_down)
    return rotate_last_is_index(pts, idx)

# ----------------- High-level ops ----------------- #
def preprocess_curve(row, smooth_s=10.0, positive_down=False, no_resample=False) -> np.ndarray:
    """Return Nx2 with heel-last & heel-align (y=0)."""
    C = row_to_xy(row, positive_down=positive_down)
    if not no_resample:
        C = rotate_last_is_index(C, heel_index(C, positive_down))
        C = closed_bspline_resample(C, n_out=40, s=smooth_s)
    C = ensure_heel_last(C, positive_down)
    C = heel_align(C, set_x=None)  # y=0 스냅
    return C

def process_pair(row_220, row_270, smooth_s=10.0, positive_down=False, no_resample=False):
    """Prepare two curves (220, 270) with identical conventions and shared heel x."""
    c220 = preprocess_curve(row_220, smooth_s, positive_down, no_resample)
    c270 = preprocess_curve(row_270, smooth_s, positive_down, no_resample)
    # 두 사이즈 동일 heel x 사용(220 기준)
    heel_x = c220[-1, 0]
    c220 = heel_align(c220, set_x=heel_x)
    c270 = heel_align(c270, set_x=heel_x)
    return c220, c270

def parse_range(s: str):
    a, b = [float(t.strip()) for t in s.split(",")]
    return a, b

def plot_closed(ax, P, *args, **kwargs):
    Q = np.vstack([P, P[0]])
    ax.plot(Q[:,0], Q[:,1], *args, **kwargs)

# ----------------- CLI ----------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="insole_ctrl_40pts_2_with270.csv")
    ap.add_argument("--smooth", type=float, default=10.0)
    ap.add_argument("--no_resample", action="store_true",
                    help="Use original CSV sampling (skip B-spline resample); heel-last & heel-align still applied.")
    ap.add_argument("--positive_down", action="store_true",
                    help="If set, treat screen-down as +y; DO NOT shift y by Size in row_to_xy.")
    ap.add_argument("--xlim", type=str, default="")
    ap.add_argument("--ylim", type=str, default="")
    ap.add_argument("--save_csv", type=str, default="")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "Size" not in df.columns:
        raise RuntimeError("CSV must contain 'Size' with 220 and/or 270 rows.")

    row_220 = df[df["Size"] == 220].iloc[0]
    row_270 = df[df["Size"] == 270].iloc[0]

    s220, s270 = process_pair(
        row_220, row_270,
        smooth_s=args.smooth,
        positive_down=args.positive_down,
        no_resample=args.no_resample
    )

    fig, ax = plt.subplots(figsize=(5.5, 9))
    plot_closed(ax, s220, "o-", label="220mm")
    plot_closed(ax, s270, "o-", label="270mm")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    # Axis ranges
    if args.xlim:
        xmin, xmax = parse_range(args.xlim); ax.set_xlim(xmin, xmax)
    if args.ylim:
        ytop, ybot = parse_range(args.ylim); ax.set_ylim(ytop, ybot)
    elif args.positive_down:
        ax.set_ylim(-290, 0)  # 기본 보기 좋게

    plt.show()

    if args.save_csv:
        rows = [xy_to_row_dict(s220, 220), xy_to_row_dict(s270, 270)]
        pd.DataFrame(rows).to_csv(args.save_csv, index=False)
        print(f"[OK] Saved: {args.save_csv}")

if __name__ == "__main__":
    main()
