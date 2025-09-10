#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append 270mm predictions to a CSV (currently 220-only) using donor 220↔270 pairs,
with strong emphasis on minimizing error.

v3 adds:
- Per-row adaptive residual weight: --residual_weight_auto with --auto_lambda_grid
  * chooses λ that minimizes donor-consensus spread (mm) among k neighbors
- Consensus-based fallback: if spread > --consensus_mm, fall back to scale_only
- Heel re-snap (always) + optional area_snap (uniform scale to (270/220)^2 area)
- CV-based auto_tune (dataset-level) retained

Usage (best safety):
python append_270_from_pairs_v3.py \
  --pairs_csv /mnt/data/insole_dummy_220_270_pairs.csv \
  --input_csv insole_ctrl_40pts_2.csv \
  --auto_tune --area_snap \
  --residual_weight_auto --auto_lambda_grid 0.0,0.2,0.4 \
  --consensus_mm 6.0 \
  --out_csv insole_ctrl_40pts_2_with270.csv
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

def xy_to_row_dict(pts: np.ndarray, xs: list[str], ys: list[str], base_row: pd.Series | None, size_mm: int | float):
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
def heel_index_min_y(pts: np.ndarray) -> int:
    return int(np.argmin(pts[:, 1]))

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

def heel_align(pts: np.ndarray, size_mm: float, set_x: float | None = None) -> np.ndarray:
    out = pts.copy()
    heel = out[-1]
    out[:, 1] += (0.0) - heel[1]
    if set_x is not None:
        out[:, 0] += set_x - out[-1, 0]
    return out

def scale_about_point(pts: np.ndarray, anchor: np.ndarray, scale: float) -> np.ndarray:
    return anchor + (pts - anchor) * scale

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

def ensure_heel_last(pts: np.ndarray) -> np.ndarray:
    return rotate_last_is_index(pts, heel_index_min_y(pts))

def polygon_area(P: np.ndarray) -> float:
    Q = np.vstack([P,P[0]])
    x = Q[:,0]; y = Q[:,1]
    return 0.5*float(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))

def polygon_perimeter(P: np.ndarray) -> float:
    Q = np.vstack([P,P[0]])
    return float(np.sum(np.linalg.norm(Q[1:] - Q[:-1], axis=1)))

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

def apply_pair_model(model: dict, P220_new: np.ndarray, residual_weight: float = 1.0) -> np.ndarray:
    s_g, R_g, t_g = model["s"], model["R"], model["t"]
    P_base = (s_g * (R_g @ P220_new.T)).T + t_g
    if residual_weight <= 1e-9:
        return P_base
    nrm = compute_normals(P_base)
    P_full = P_base + nrm * model["s_res"][:, None]
    return P_base + residual_weight * (P_full - P_base)

def procrustes_distance(A: np.ndarray, B: np.ndarray) -> float:
    s, R, t = umeyama_similarity(A, B, with_scaling=True)  # B ≈ s R A + t
    A2B = (s * (R @ A.T)).T + t
    return float(np.mean(np.sum((A2B - B)**2, axis=1)))

# ---------- Preprocessing ---------- #
def preprocess_220_only(row_220: pd.Series, xs: list[str], ys: list[str], smooth_s=10.0) -> np.ndarray:
    C = row_to_xy(row_220, xs, ys)
    C = rotate_last_is_index(C, heel_index_min_y(C))
    C = closed_bspline_resample(C, n_out=len(xs), s=smooth_s)
    C = ensure_heel_last(C)
    C = heel_align(C, size_mm=220.0, set_x=C[-1,0])
    return C

def prepare_donors(df_pairs: pd.DataFrame, xs: list[str], ys: list[str], smooth_s: float) -> tuple[list[np.ndarray], list[dict]]:
    donors_220 = []
    donors_models = []
    for pid, g in df_pairs.groupby("PairID"):
        if not ({220,270} <= set(g["Size"].unique())):
            continue
        r220 = g[g["Size"]==220].iloc[0]
        r270 = g[g["Size"]==270].iloc[0]
        P220 = preprocess_220_only(r220, xs, ys, smooth_s=smooth_s)
        P270 = preprocess_220_only(r270, xs, ys, smooth_s=smooth_s)
        heel_x = P270[-1,0]
        P270 = heel_align(P270, size_mm=270.0, set_x=heel_x)
        donors_220.append(P220)
        donors_models.append(fit_pair_model(P220, P270))
    if not donors_models:
        raise RuntimeError("No valid donors in pairs CSV (need PairID with both Size=220 and 270).")
    return donors_220, donors_models

# ---------- Adaptive residual (per-row) ---------- #
def consensus_spread_mm(preds: list[np.ndarray]) -> float:
    """Average pointwise std (Euclidean) across donor predictions."""
    P = np.stack(preds, axis=0)  # (D,N,2)
    mean = P.mean(axis=0, keepdims=True)
    diffs = np.linalg.norm(P - mean, axis=2)  # (D,N)
    std = diffs.std(axis=0)  # (N,)
    return float(std.mean())

def choose_lambda_by_consensus(P220_t: np.ndarray, neighbors_idx: np.ndarray, w: np.ndarray,
                               donors_models: list[dict], lam_grid: list[float]) -> tuple[float, np.ndarray]:
    best = (lam_grid[0], None, 1e18)
    for lam in lam_grid:
        donor_preds = [apply_pair_model(donors_models[j], P220_t, residual_weight=lam) for j in neighbors_idx]
        spread = consensus_spread_mm(donor_preds)
        if spread < best[2]:
            best = (lam, donor_preds, spread)
    lam_star, donor_preds_star, spread_star = best
    # weighted sum with the same w
    P_pred = np.sum([wj * Pj for wj, Pj in zip(w, donor_preds_star)], axis=0)
    return lam_star, P_pred, spread_star

# ---------- Predictors ---------- #
def predict_270_for(P220_t: np.ndarray, donors_220: list[np.ndarray], donors_models: list[dict],
                    mode: str, k: int, residual_weight: float, var_thresh: float,
                    residual_weight_auto: bool, auto_lambda_grid: list[float],
                    consensus_mm: float) -> tuple[np.ndarray, dict]:
    # A: scale_only
    if mode == "scale_only":
        anchor = P220_t[-1].copy()
        P270 = scale_about_point(P220_t, anchor, 270.0/220.0)
        P270[-1,1] = 0.0
        return P270, {"mode":"scale_only"}

    # choose k donors by Procrustes distance
    dists = np.array([procrustes_distance(P220_t, D) for D in donors_220])
    idx = np.argsort(dists)[:max(1,k)]
    w = 1.0 / (dists[idx] + 1e-8); w = w / w.sum()

    if mode == "donor_avg":
        if residual_weight_auto:
            lam_star, P_pred, spread = choose_lambda_by_consensus(P220_t, idx, w, donors_models, auto_lambda_grid)
            meta = {"mode":"donor_avg","k":k,"lambda":lam_star,"spread":spread}
            # fallback if spread too large
            if spread > consensus_mm:
                anchor = P220_t[-1].copy()
                P270 = scale_about_point(P220_t, anchor, 270.0/220.0)
                P270[-1,1] = -270.0
                meta["fallback"]="scale_only_due_to_spread"
                return P270, meta
            return P_pred, meta
        else:
            preds = []
            for j, wj in zip(idx, w):
                Pj = apply_pair_model(donors_models[j], P220_t, residual_weight=residual_weight)
                preds.append(wj * Pj)
            return np.sum(preds, axis=0), {"mode":"donor_avg","k":k,"lambda":residual_weight}

    if mode == "pca_knn":
        # PCA on all donor residuals
        S = np.stack([m["s_res"] for m in donors_models], axis=0)  # (M,N)
        mu = S.mean(axis=0, keepdims=True)
        S0 = S - mu
        U, Sigma, Vt = np.linalg.svd(S0, full_matrices=False)
        var = (Sigma**2) / max(S.shape[0]-1, 1)
        ratio = var / max(var.sum(), 1e-12)
        K = int(np.searchsorted(np.cumsum(ratio), var_thresh) + 1)
        K = max(1, min(K, Vt.shape[0]))
        Vt_K = Vt[:K, :]
        Z = S0 @ Vt_K.T  # donor scores
        z_bar = np.sum((w[:,None]) * Z[idx, :], axis=0)  # (K,)
        s_res_pred = (z_bar @ Vt_K) + mu.ravel()  # (N,)
        bases = []
        for j, wj in zip(idx, w):
            m = donors_models[j]
            s_g, R_g, t_g = m["s"], m["R"], m["t"]
            P_base = (s_g * (R_g @ P220_t.T)).T + t_g
            bases.append(wj * P_base)
        P_base_bar = np.sum(bases, axis=0)
        nrm = compute_normals(P_base_bar)
        P_pred = P_base_bar + residual_weight * (nrm * s_res_pred[:, None])
        return P_pred, {"mode":"pca_knn","k":k,"lambda":residual_weight,"K":K}

    raise ValueError(f"Unknown mode: {mode}")

# ---------- Snaps ---------- #
def heel_resnap(P270: np.ndarray, P220_ref: np.ndarray) -> np.ndarray:
    return heel_align(P270, size_mm=270.0, set_x=P220_ref[-1,0])

def area_resnap(P270: np.ndarray, P220_ref: np.ndarray) -> np.ndarray:
    def poly_area(P):
        Q = np.vstack([P,P[0]]); x=Q[:,0]; y=Q[:,1]
        return 0.5*float(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))
    area_220 = abs(poly_area(P220_ref))
    target = area_220 * (270.0/220.0)**2
    area_pred = abs(poly_area(P270))
    if area_pred < 1e-9:
        return P270
    gamma = np.sqrt(max(target,1e-12) / area_pred)
    anchor = P220_ref[-1].copy()
    P_adj = scale_about_point(P270, anchor, gamma)
    P_adj = heel_resnap(P_adj, P220_ref)
    return P_adj

# ---------- CV auto-tune (dataset-level) ---------- #
def rmse(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((P - Q)**2, axis=1))))

def cv_auto_tune(df_pairs: pd.DataFrame, xs: list[str], ys: list[str], smooth_s: float,
                 grid_modes, grid_k, grid_lambda, grid_var) -> dict:
    pairs = []
    for pid, g in df_pairs.groupby("PairID"):
        if not ({220,270} <= set(g["Size"].unique())):
            continue
        r220 = g[g["Size"]==220].iloc[0]
        r270 = g[g["Size"]==270].iloc[0]
        P220 = preprocess_220_only(r220, xs, ys, smooth_s=smooth_s)
        P270 = preprocess_220_only(r270, xs, ys, smooth_s=smooth_s)
        P270 = heel_align(P270, size_mm=270.0, set_x=P270[-1,0])
        pairs.append({"pid": pid, "P220": P220, "P270": P270})
    assert len(pairs) >= 3, "Need at least 3 donor pairs for CV."

    best = {"rmse": 1e18}
    donors_220_all = [p["P220"] for p in pairs]
    donors_models_all = [fit_pair_model(p["P220"], p["P270"]) for p in pairs]

    for mode in grid_modes:
        ks = grid_k if mode != "scale_only" else [0]
        lambdas = grid_lambda if mode != "scale_only" else [0.0]
        vars_ = grid_var if mode == "pca_knn" else [0.0]
        for k in ks:
            for lam in lambdas:
                for vt in vars_:
                    errs = []
                    for i, test in enumerate(pairs):
                        donors_220 = donors_220_all[:i] + donors_220_all[i+1:]
                        donors_models = donors_models_all[:i] + donors_models_all[i+1:]
                        P220_t = test["P220"]; P270_true = test["P270"]
                        P270_pred, _ = predict_270_for(P220_t, donors_220, donors_models,
                                                       mode=mode, k=k if k>0 else 1,
                                                       residual_weight=lam, var_thresh=vt if vt>0 else 0.95,
                                                       residual_weight_auto=False, auto_lambda_grid=[0.0],
                                                       consensus_mm=1e9)
                        P270_pred = heel_resnap(P270_pred, P220_t)
                        P270_pred = area_resnap(P270_pred, P220_t)
                        errs.append(rmse(P270_pred, P270_true))
                    avg_rmse = float(np.mean(errs))
                    if avg_rmse < best["rmse"]:
                        best = {"mode": mode, "k": k, "lambda": lam, "var_thresh": vt, "rmse": avg_rmse}
    return best

# ---------- Main append ---------- #
def append_270(pairs_csv: str, input_csv: str, out_csv: str | None,
               mode: str, k: int, residual_weight: float, var_thresh: float,
               smooth_s: float, plot_overlays: str | None, area_snap: bool,
               auto_tune: bool, grid_k, grid_lambda, grid_var, inplace: bool,
               residual_weight_auto: bool, auto_lambda_grid, consensus_mm: float):
    df_in = pd.read_csv(input_csv)
    xs, ys = detect_xy_cols(df_in)
    df_pairs = pd.read_csv(pairs_csv)

    if auto_tune:
        best = cv_auto_tune(
            df_pairs=df_pairs, xs=xs, ys=ys, smooth_s=smooth_s,
            grid_modes=["scale_only","donor_avg","pca_knn"],
            grid_k=grid_k, grid_lambda=grid_lambda, grid_var=grid_var
        )
        print(f"[AUTO-TUNE] Best config by CV RMSE: {best}")
        mode = best["mode"]; k = best["k"]; residual_weight = best["lambda"]; var_thresh = best["var_thresh"]

    donors_220, donors_models = prepare_donors(df_pairs, xs, ys, smooth_s=smooth_s)

    rows_out = []
    rows_out.extend(df_in.to_dict(orient="records"))

    to_plot = []
    metas = []
    for idx_row, r in df_in.iterrows():
        if int(r.get("Size", 0)) != 220:
            continue
        P220_t = preprocess_220_only(r, xs, ys, smooth_s=smooth_s)
        P270_pred, meta = predict_270_for(P220_t, donors_220, donors_models,
                                          mode=mode, k=k if k>0 else 1,
                                          residual_weight=residual_weight, var_thresh=var_thresh,
                                          residual_weight_auto=residual_weight_auto,
                                          auto_lambda_grid=auto_lambda_grid,
                                          consensus_mm=consensus_mm)
        P270_pred = heel_resnap(P270_pred, P220_t)
        if area_snap:
            P270_pred = area_resnap(P270_pred, P220_t)

        row270 = xy_to_row_dict(P270_pred, xs, ys, base_row=r, size_mm=270)
        rows_out.append(row270)
        to_plot.append((P220_t, P270_pred, r))
        metas.append(meta)

    df_out = pd.DataFrame(rows_out)

    if out_csv is None or out_csv == "":
        out_path = Path(input_csv).with_name(Path(input_csv).stem + "_with270.csv")
    else:
        out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if inplace:
        out_path = Path(input_csv)
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Wrote: {out_path} (appended {len(to_plot)} predicted 270 rows)")
    if metas:
        print("[DETAIL] metas for rows:", metas)

    if plot_overlays:
        pdir = Path(plot_overlays); pdir.mkdir(parents=True, exist_ok=True)
        for i,(P220_t, P270_pred, r) in enumerate(to_plot):
            fig, ax = plt.subplots(figsize=(5.5, 8))
            def plot_closed(ax, P, label):
                Q = np.vstack([P, P[0]]); ax.plot(Q[:,0], Q[:,1], label=label)
            plot_closed(ax, P220_t, "220")
            plot_closed(ax, P270_pred, "270 (pred)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Pred 270 — row {i} [{mode}]")
            ax.legend()
            fig.tight_layout()
            fig.savefig(pdir / f"pred_{i:03d}.png", dpi=160)
            plt.close(fig)

def parse_list_of_numbers(s: str, typ=float):
    return [typ(x.strip()) for x in s.split(",") if x.strip()!=""]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--mode", choices=["scale_only","donor_avg","pca_knn"], default="donor_avg")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--residual_weight", type=float, default=0.4)
    ap.add_argument("--var_thresh", type=float, default=0.95)
    ap.add_argument("--smooth", type=float, default=10.0)
    ap.add_argument("--plot_overlays", default="")
    ap.add_argument("--area_snap", action="store_true")
    ap.add_argument("--auto_tune", action="store_true")
    ap.add_argument("--grid_k", type=str, default="3,5,7")
    ap.add_argument("--grid_lambda", type=str, default="0.0,0.2,0.4,0.6")
    ap.add_argument("--grid_var", type=str, default="0.90,0.95,0.98")
    ap.add_argument("--inplace", action="store_true")
    # v3 additions
    ap.add_argument("--residual_weight_auto", action="store_true")
    ap.add_argument("--auto_lambda_grid", type=str, default="0.0,0.2,0.4")
    ap.add_argument("--consensus_mm", type=float, default=6.0)
    args = ap.parse_args()

    append_270(
        pairs_csv=args.pairs_csv,
        input_csv=args.input_csv,
        out_csv=args.out_csv if args.out_csv else None,
        mode=args.mode,
        k=args.k,
        residual_weight=args.residual_weight,
        var_thresh=args.var_thresh,
        smooth_s=args.smooth,
        plot_overlays=args.plot_overlays if args.plot_overlays else None,
        area_snap=args.area_snap,
        auto_tune=args.auto_tune,
        grid_k=parse_list_of_numbers(args.grid_k, int),
        grid_lambda=parse_list_of_numbers(args.grid_lambda, float),
        grid_var=parse_list_of_numbers(args.grid_var, float),
        inplace=args.inplace,
        residual_weight_auto=args.residual_weight_auto,
        auto_lambda_grid=parse_list_of_numbers(args.auto_lambda_grid, float),
        consensus_mm=args.consensus_mm
    )

if __name__ == "__main__":
    main()
