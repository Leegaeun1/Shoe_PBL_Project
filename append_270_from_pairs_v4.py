#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append 270mm predictions to a CSV (currently 220-only) using donor 220↔270 pairs.

This v4 adds a y-direction switch (--y_dir {auto,up,down}) so outputs never look vertically flipped.
- Heel auto-detect (width-based) and ensure heel is LAST even after resample
- Snap heel to y=0
- Unify y-direction per --y_dir at every stage (donors, input 220, predicted 270)

Typical run (positive-down datasets):
python append_270_from_pairs_v4.py \
  --pairs_csv insole_dummy_220_270_pairs.csv \
  --input_csv insole_ctrl_40pts_2.csv \
  --mode pca_knn --k 7 --var_thresh 0.97 --residual_weight 0.4 \
  --area_snap --perimeter_snap \
  --y_dir down \
  --plot_overlays preds_v4 \
  --out_csv insole_ctrl_40pts_2_with270_v4.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

# ---------- Column utilities ---------- #
def detect_xy_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    xs = [c for c in df.columns if str(c).lower().startswith("x")]
    ys = [c for c in df.columns if str(c).lower().startswith("y")]
    def key(c):
        num = ''.join([ch for ch in str(c) if ch.isdigit()])
        return int(num) if num else 0
    xs = sorted(xs, key=key); ys = sorted(ys, key=key)
    assert len(xs) == len(ys) and len(xs) >= 10, "Need paired x*/y* columns (>=10 points)."
    return xs, ys

def row_to_xy(row: pd.Series, xs: list[str], ys: list[str]) -> np.ndarray:
    return np.stack([row[xs].to_numpy(float), row[ys].to_numpy(float)], axis=1)

def xy_to_row_dict(pts: np.ndarray, xs: list[str], ys: list[str],
                   base_row: pd.Series | None, size_mm: int | float):
    d = {}
    for i,(cx,cy) in enumerate(zip(xs,ys)):
        d[cx] = float(pts[i,0]); d[cy] = float(pts[i,1])
    if base_row is not None:
        for c in base_row.index:
            if c in xs or c in ys or c == "Size":
                continue
            d[c] = base_row[c]
    d["Size"] = int(size_mm)
    return d

# ---------- Geometry helpers ---------- #
def rotate_last_is_index(pts: np.ndarray, last_idx: int) -> np.ndarray:
    if last_idx == len(pts) - 1:
        return pts.copy()
    return np.vstack([pts[last_idx + 1 :], pts[: last_idx + 1]])

def closed_bspline_resample(pts: np.ndarray, n_out: int, s: float = 8.0) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]
    tck, _ = interpolate.splprep([x, y], s=s, per=True)
    u = np.linspace(0, 1, n_out, endpoint=False)
    xf, yf = interpolate.splev(u, tck)
    return np.column_stack([xf, yf])

def heel_align(pts: np.ndarray, set_x: float | None = None) -> np.ndarray:
    """Snap heel (last vertex) to y=0; optionally also set heel x."""
    out = pts.copy()
    heel = out[-1]
    out[:, 1] += 0.0 - heel[1]
    if set_x is not None:
        out[:, 0] += set_x - out[-1, 0]
    return out

def scale_about_point(pts: np.ndarray, anchor: np.ndarray, scale: float) -> np.ndarray:
    return anchor + (pts - anchor) * scale

def polygon_area(P: np.ndarray) -> float:
    Q = np.vstack([P,P[0]])
    x = Q[:,0]; y = Q[:,1]
    return 0.5*float(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))

def polygon_perimeter(P: np.ndarray) -> float:
    Q = np.vstack([P,P[0]])
    return float(np.sum(np.linalg.norm(Q[1:] - Q[:-1], axis=1)))

def compute_normals(poly: np.ndarray) -> np.ndarray:
    N = len(poly)
    T = np.zeros_like(poly)
    if N >= 3:
        T[1:-1] = poly[2:] - poly[:-2]
    T[0] = poly[1] - poly[0]
    T[-1] = poly[-1] - poly[-2]
    nrm = np.stack([-T[:,1], T[:,0]], axis=1)
    n = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    return nrm / n

# ---------- Heel detection & y-direction ---------- #
def heel_index_auto(pts: np.ndarray) -> int:
    """Choose heel among min_y/max_y by narrower local width (std of x)."""
    y = pts[:,1]
    i_min, i_max = int(np.argmin(y)), int(np.argmax(y))
    def local_width(i, w=4):
        N = len(pts)
        idx = [(i+j) % N for j in range(-w, w+1)]
        return float(np.std(pts[idx,0]))
    return i_min if local_width(i_min) < local_width(i_max) else i_max

def ensure_heel_last(pts: np.ndarray) -> np.ndarray:
    return rotate_last_is_index(pts, heel_index_auto(pts))

def unify_y_dir(pts: np.ndarray, target: str) -> np.ndarray:
    """
    After heel-align(y=0), force y-direction to:
    - 'down': toe negative (positive-down plotting)
    - 'up'  : toe positive (positive-up plotting)
    - 'auto': keep as-is if already dominant in target direction; else flip
    """
    out = pts.copy()
    y = out[:,1]
    # current: up if positive span >= negative span
    cur = "up" if y.max() >= -y.min() else "down"
    if target == "auto":
        return out
    if target != cur:
        out[:,1] = -out[:,1]
    return out

# ---------- Models ---------- #
def umeyama_similarity(X: np.ndarray, Y: np.ndarray, with_scaling=True):
    Xc = X.mean(axis=0); Yc = Y.mean(axis=0)
    X0 = X - Xc; Y0 = Y - Yc
    C = (Y0.T @ X0) / X.shape[0]
    U, S, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    if with_scaling:
        varX = (X0**2).sum() / X.shape[0]
        s = (S @ np.ones_like(S)) / (varX + 1e-12)
    else:
        s = 1.0
    t = Yc - s * (R @ Xc)
    return float(s), R, t

def fit_pair_model(P220: np.ndarray, P270: np.ndarray) -> dict:
    s_g, R_g, t_g = umeyama_similarity(P220, P270, with_scaling=True)
    P220_fit = (s_g * (R_g @ P220.T)).T + t_g
    nrm = compute_normals(P220_fit)
    E = P270 - P220_fit
    s_res = (E * nrm).sum(axis=1)
    return {"s": s_g, "R": R_g, "t": t_g, "s_res": s_res}

def apply_pair_model(model: dict, P220_new: np.ndarray, residual_lambda_vec: np.ndarray) -> np.ndarray:
    s_g, R_g, t_g = model["s"], model["R"], model["t"]
    P_base = (s_g * (R_g @ P220_new.T)).T + t_g
    nrm = compute_normals(P_base)
    P_full = P_base + nrm * model["s_res"][:, None]
    return P_base + (P_full - P_base) * residual_lambda_vec[:, None]

def procrustes_distance(A: np.ndarray, B: np.ndarray) -> float:
    s, R, t = umeyama_similarity(A, B, with_scaling=True)  # B ≈ s R A + t
    A2B = (s * (R @ A.T)).T + t
    return float(np.mean(np.sum((A2B - B)**2, axis=1)))

# ---------- Preprocessing ---------- #
def preprocess_220_only(row_220: pd.Series, xs: list[str], ys: list[str], smooth_s=10.0, y_dir="down") -> np.ndarray:
    """Return Nx2 with heel-last & heel-align(y=0) & unified y-direction."""
    C = row_to_xy(row_220, xs, ys)
    C = rotate_last_is_index(C, heel_index_auto(C))          # first pass
    C = closed_bspline_resample(C, n_out=len(xs), s=smooth_s)
    C = ensure_heel_last(C)                                   # robust after resample
    C = heel_align(C, set_x=C[-1,0])                          # y=0
    C = unify_y_dir(C, target=y_dir)                          # y-direction
    return C

def prepare_donors(df_pairs: pd.DataFrame, xs: list[str], ys: list[str], smooth_s: float, y_dir="down") -> tuple[list[np.ndarray], list[dict]]:
    donors_220 = []
    donors_models = []
    for pid, g in df_pairs.groupby("PairID"):
        sizes = set(map(int, g["Size"].unique()))
        if not ({220,270} <= sizes):
            continue
        r220 = g[g["Size"]==220].iloc[0]
        r270 = g[g["Size"]==270].iloc[0]
        P220 = preprocess_220_only(r220, xs, ys, smooth_s=smooth_s, y_dir=y_dir)
        P270 = preprocess_220_only(r270, xs, ys, smooth_s=smooth_s, y_dir=y_dir)
        heel_x = P270[-1,0]
        P270 = heel_align(P270, set_x=heel_x)
        P220 = unify_y_dir(P220, target=y_dir)
        P270 = unify_y_dir(P270, target=y_dir)
        donors_220.append(P220)
        donors_models.append(fit_pair_model(P220, P270))
    if not donors_models:
        raise RuntimeError("No valid donors in pairs CSV (need PairID with both Size=220 and 270).")
    return donors_220, donors_models

# ---------- Predictors ---------- #
def predict_270_for(P220_t: np.ndarray, donors_220: list[np.ndarray], donors_models: list[dict],
                    mode: str, k: int, residual_lambda_vec: np.ndarray, var_thresh: float,
                    pca_weight_vec: np.ndarray):
    # choose neighbors
    dists = np.array([procrustes_distance(P220_t, D) for D in donors_220])
    idx = np.argsort(dists)[:max(1,k)]
    w = 1.0 / (dists[idx] + 1e-8); w = w / w.sum()

    if mode == "scale_only":
        anchor = P220_t[-1].copy()
        P_scale = scale_about_point(P220_t, anchor, 270.0/220.0)
        P_scale[-1,1] = 0.0
        return P_scale, {"mode":"scale_only"}

    if mode == "donor_avg":
        preds = []
        for j, wj in zip(idx, w):
            preds.append(wj * apply_pair_model(donors_models[j], P220_t, residual_lambda_vec))
        return np.sum(preds, axis=0), {"mode":"donor_avg","k":k}

    # pca_knn
    S = np.stack([m["s_res"] for m in donors_models], axis=0)  # (M,N)
    mu = S.mean(axis=0, keepdims=True)
    S0 = S - mu
    Wcol = pca_weight_vec
    S0w = S0 * Wcol[None, :]
    U, Sigma, Vt_w = np.linalg.svd(S0w, full_matrices=False)
    var = (Sigma**2) / max(S.shape[0]-1, 1)
    ratio = var / max(var.sum(), 1e-12)
    K = int(np.searchsorted(np.cumsum(ratio), var_thresh) + 1)
    K = max(1, min(K, Vt_w.shape[0]))
    Vt_K_w = Vt_w[:K, :]
    Z = S0w @ Vt_K_w.T
    z_bar = np.sum((w[:,None]) * Z[idx, :], axis=0)
    s_res_pred_w = (z_bar @ Vt_K_w) + (mu * Wcol[None,:]).ravel()
    eps = 1e-8
    s_res_pred = s_res_pred_w / (Wcol + eps)

    bases = []
    for j, wj in zip(idx, w):
        m = donors_models[j]
        P_base = (m["s"] * (m["R"] @ P220_t.T)).T + m["t"]
        bases.append(wj * P_base)
    P_base_bar = np.sum(bases, axis=0)
    nrm = compute_normals(P_base_bar)
    P_pred = P_base_bar + (nrm * s_res_pred[:, None]) * residual_lambda_vec[:, None]
    return P_pred, {"mode":"pca_knn","k":k,"K":K}

# ---------- Snaps ---------- #
def heel_resnap(P270: np.ndarray, P220_ref: np.ndarray) -> np.ndarray:
    return heel_align(P270, set_x=P220_ref[-1,0])

def area_resnap(P270: np.ndarray, P220_ref: np.ndarray) -> np.ndarray:
    area_220 = abs(polygon_area(P220_ref))
    target = area_220 * (270.0/220.0)**2
    area_pred = abs(polygon_area(P270))
    if area_pred < 1e-9:
        return P270
    gamma = np.sqrt(max(target,1e-12) / area_pred)
    anchor = P220_ref[-1].copy()
    P_adj = scale_about_point(P270, anchor, gamma)
    return heel_resnap(P_adj, P220_ref)

def perimeter_resnap(P270: np.ndarray, P220_ref: np.ndarray) -> np.ndarray:
    per_220 = polygon_perimeter(P220_ref)
    target = per_220 * (270.0/220.0)
    per_pred = polygon_perimeter(P270)
    if per_pred < 1e-12:
        return P270
    gamma = max(target,1e-12) / per_pred
    anchor = P220_ref[-1].copy()
    P_adj = scale_about_point(P270, anchor, gamma)
    return heel_resnap(P_adj, P220_ref)

# ---------- Main append ---------- #
def append_270(pairs_csv: str, input_csv: str, out_csv: str | None,
               mode: str, k: int, residual_weight: float, var_thresh: float,
               smooth_s: float, plot_overlays: str | None, area_snap: bool, perimeter_snap_flag: bool,
               y_dir: str, pca_weight_profile: str):

    df_in = pd.read_csv(input_csv)
    xs, ys = detect_xy_cols(df_in)
    df_pairs = pd.read_csv(pairs_csv)

    N = len(xs)
    # simple profile: constant 1.0 unless provided
    def parse_profile(profile: str, N: int) -> np.ndarray:
        if not profile: return np.ones(N, dtype=float)
        pairs = []
        for tok in profile.split(","):
            s, v = tok.split(":"); pairs.append((float(s), float(v)))
        pairs = sorted(pairs, key=lambda x: x[0])
        if pairs[0][0] > 0.0: pairs = [(0.0, pairs[0][1])] + pairs
        if pairs[-1][0] < 1.0: pairs = pairs + [(1.0, pairs[0][1])]
        u = np.linspace(0.0, 1.0, N, endpoint=False)
        return np.interp(u, [p[0] for p in pairs], [p[1] for p in pairs])

    pca_weight_vec = parse_profile(pca_weight_profile, N)
    pca_weight_vec = pca_weight_vec / (pca_weight_vec.mean() + 1e-12)

    donors_220, donors_models = prepare_donors(df_pairs, xs, ys, smooth_s=smooth_s, y_dir=y_dir)

    rows_out = []
    rows_out.extend(df_in.to_dict(orient="records"))
    to_plot = []

    lam_vec = np.ones(N, dtype=float) * residual_weight

    for _, r in df_in.iterrows():
        if int(r.get("Size", 0)) != 220:
            continue
        P220_t = preprocess_220_only(r, xs, ys, smooth_s=smooth_s, y_dir=y_dir)

        P270_pred, meta = predict_270_for(
            P220_t, donors_220, donors_models,
            mode=mode, k=k, residual_lambda_vec=lam_vec, var_thresh=var_thresh,
            pca_weight_vec=pca_weight_vec
        )
        # snaps + final y-direction unify
        P270_pred = heel_resnap(P270_pred, P220_t)
        if area_snap:
            P270_pred = area_resnap(P270_pred, P220_t)
        if perimeter_snap_flag:
            P270_pred = perimeter_resnap(P270_pred, P220_t)
        P270_pred = unify_y_dir(P270_pred, target=y_dir)

        row270 = xy_to_row_dict(P270_pred, xs, ys, base_row=r, size_mm=270)
        rows_out.append(row270)
        to_plot.append((P220_t, P270_pred))

    df_out = pd.DataFrame(rows_out)
    out_path = Path(out_csv) if out_csv else Path(input_csv).with_name(Path(input_csv).stem + "_with270_v4.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Wrote: {out_path} (appended {len(to_plot)} predicted 270 rows)")

    if plot_overlays:
        pdir = Path(plot_overlays); pdir.mkdir(parents=True, exist_ok=True)
        for i,(P220_t, P270_pred) in enumerate(to_plot):
            fig, ax = plt.subplots(figsize=(3.2,9))
            def plot_closed(ax, P, label):
                Q = np.vstack([P, P[0]]); ax.plot(Q[:,0], Q[:,1], "o-", ms=3, label=label)
            plot_closed(ax, P220_t, "220mm")
            plot_closed(ax, P270_pred, "270mm")
            ax.axhline(0, lw=0.5, c="k"); ax.axvline(P220_t[-1,0], lw=0.5, c="k")
            ax.set_aspect("equal", adjustable="box")
            ax.legend(); ax.set_title(f"Pred 270 — row {i} [{mode}]")
            fig.tight_layout()
            fig.savefig(pdir / f"pred_{i:03d}.png", dpi=150)
            plt.close(fig)

# ---------- CLI ---------- #
def parse_list_of_numbers(s: str, typ=float):
    return [typ(x.strip()) for x in s.split(",") if x.strip()!=""]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--mode", choices=["scale_only","donor_avg","pca_knn"], default="pca_knn")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--residual_weight", type=float, default=0.4)
    ap.add_argument("--var_thresh", type=float, default=0.97)
    ap.add_argument("--smooth_s", type=float, default=10.0)
    ap.add_argument("--area_snap", action="store_true")
    ap.add_argument("--perimeter_snap", action="store_true")
    ap.add_argument("--y_dir", choices=["auto","up","down"], default="down")
    ap.add_argument("--pca_weight_profile", type=str, default="")
    ap.add_argument("--plot_overlays", default="")
    args = ap.parse_args()

    append_270(
        pairs_csv=args.pairs_csv,
        input_csv=args.input_csv,
        out_csv=(args.out_csv or None),
        mode=args.mode, k=args.k, residual_weight=args.residual_weight,
        var_thresh=args.var_thresh, smooth_s=args.smooth_s,
        plot_overlays=(args.plot_overlays or None),
        area_snap=args.area_snap, perimeter_snap_flag=args.perimeter_snap,
        y_dir=args.y_dir, pca_weight_profile=args.pca_weight_profile
    )

if __name__ == "__main__":
    main()
