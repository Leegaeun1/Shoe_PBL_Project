#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_insole_270_nocli.py

- 명령줄 인자 없이 파일 자체를 실행
- 맨 위 CONFIG 블록만 수정해서 경로/모드/하이퍼파라미터를 제어
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# CONFIG: 여기만 바꿔서 사용하세요
# =========================
CONFIG = {
    # 파일 경로
    "pairs_csv": "insole_viewer_output.csv",          # 같은 model의 220/270 도너 쌍 CSV
    "input_csv": "insole2_220_40pts_Real.csv",        # 270을 예측할 220만 있는 CSV
    "output_csv": "insole_ctrl_40pts_2_with270_3.csv",  # 예측 결과를 append하여 저장할 CSV

    # 포인트 수(권장 40)

    # 포인트 수(권장 40)
    "n_points": 40,

    # 데이터 상하반전 보정
    "invert_y_220": False,     # 도너 220
    "invert_y_270": False,     # 도너 270
    "invert_y_input": False,   # 입력 220

    # 예측 모드: "scale_only" | "donor_avg" | "pca_knn"
    "mode": "pca_knn",
    "k": 7,
    "var_thresh": 0.97,
    "residual_weight": 1.0,
    "residual_weight_auto": True,
    "auto_lambda_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
    "consensus_mm": 1.2,

    # 출력 Size 값(예: 숫자 270로 저장)
    "pred_size_value": 270,

    # 자동 튜닝 (기본 끔)
    "auto_tune": False,
    "tune_mode_grid": ("scale_only", "donor_avg", "pca_knn"),
    "tune_k_grid": (3, 5, 7),
    "tune_var_grid": (0.90, 0.95, 0.97),
    "tune_lam_grid": (0.25, 0.5, 0.75, 1.0),
}


# =========================
# 공통 유틸
# =========================
def norm_col_map(columns: List[str]) -> Dict[str, str]:
    return {c.lower(): c for c in columns}

def get_existing_name(normmap: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        key = cand.lower()
        if key in normmap:
            return normmap[key]
    return None

def detect_point_count(columns: List[str]) -> int:
    lowers = [c.lower() for c in columns]
    xcount = len([c for c in lowers if c.startswith("x") and c[1:].isdigit()])
    ycount = len([c for c in lowers if c.startswith("y") and c[1:].isdigit()])
    return max(0, min(xcount, ycount))

def row_to_xy_descending(row: pd.Series, n: int, normmap: Dict[str, str]) -> np.ndarray:
    """
    점 순서: 40,39,...,1 식으로 역순(시계방향)으로 배열 생성.
    CSV 컬럼은 x1..xN 순서라도 내부에서 [pN, pN-1, ..., p1]로 바꾼다.
    """
    pts = []
    for i in range(n, 1-1, -1):  # n, n-1, ..., 1
        xname = get_existing_name(normmap, [f"x{i}", f"X{i}"])
        yname = get_existing_name(normmap, [f"y{i}", f"Y{i}"])
        if xname is None or yname is None:
            raise KeyError(f"좌표 컬럼 누락: x{i}/y{i}")
        x = float(row[xname]); y = float(row[yname])
        pts.append([x, y])
    return np.asarray(pts, dtype=np.float64)

def xy_to_row_dict(xy: np.ndarray) -> Dict[str, float]:
    # 출력은 X1..Y40(오름차순)로 저장
    n = xy.shape[0]

    out = {}
    for j in range(n):                  # j = 0..n-1
        col = n - j                     # 40,39,...,1
        out[f"x{col}"] = float(xy[j,0])
        out[f"y{col}"] = float(xy[j,1])
    return out


# =========================
# 전처리 / 정규화
# =========================
def heel_index_min_y(xy: np.ndarray) -> int:
    # 뒤꿈치 = y가 가장 작은 점
    return int(np.argmin(xy[:, 1]))

def rotate_to_make_heel_first(xy: np.ndarray, heel_idx: Optional[int]=None) -> np.ndarray:
    # 뒤꿈치가 첫 번째(인덱스 0)가 되도록 회전
    if heel_idx is None:
        heel_idx = heel_index_min_y(xy)
    return np.concatenate([xy[heel_idx:], xy[:heel_idx]], axis=0)

def cumulative_arclen(xy: np.ndarray) -> np.ndarray:
    diffs = np.diff(np.vstack([xy, xy[0]]), axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    s = np.cumsum(seg)
    s = np.insert(s, 0, 0.0)
    return s[:-1]

def closed_resample(xy: np.ndarray, n_out: int, smooth_win: int = 3) -> np.ndarray:
    s = cumulative_arclen(xy)
    total = s[-1] + np.linalg.norm(xy[0] - xy[-1])
    target = np.linspace(0, total, n_out+1)[:-1]
    xy2 = np.vstack([xy, xy[0]]); s2 = np.append(s, total)
    x_interp = np.interp(target, s2, xy2[:, 0])
    y_interp = np.interp(target, s2, xy2[:, 1])
    out = np.stack([x_interp, y_interp], axis=1)

    if smooth_win and smooth_win > 1:
        k = smooth_win; pad = k // 2
        ext = np.vstack([out[-pad:], out, out[:pad]])
        win = []
        for i in range(len(out)):
            sl = ext[i:i+k]
            win.append(np.mean(sl, axis=0))
        out = np.asarray(win)
    return out

def polygon_area(xy: np.ndarray) -> float:
    x = xy[:, 0]; y = xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def enforce_clockwise_if_needed(xy: np.ndarray) -> np.ndarray:
    """
    시계방향이면 면적<0. 데이터가 시계방향이라고 했으므로
    면적이 양수(반시계)일 때만 뒤집어서 시계로 통일.
    """
    area = polygon_area(xy)
    if area > 0:
        return xy[::-1].copy()
    return xy

def heel_align_origin(xy: np.ndarray) -> np.ndarray:
    heel = xy[0]  # 이미 heel-first 상태
    return xy - heel

def shape_preprocess(xy_raw: np.ndarray, n_points: int, invert_y: bool=False) -> np.ndarray:
    pts = xy_raw.copy()
    if invert_y:
        pts[:, 1] = -pts[:, 1]
    # 뒤꿈치 = min y → 첫 번째로 회전
    pts = rotate_to_make_heel_first(pts, heel_idx=heel_index_min_y(pts))
    # 곡선 재샘플 (방향 보존)
    pts = closed_resample(pts, n_points, smooth_win=3)
    # 뒤꿈치 원점 정렬
    pts = heel_align_origin(pts)
    # 방향 시계방향 강제(필요 시만)
    pts = enforce_clockwise_if_needed(pts)
    return pts


# =========================
# 전역 정합(Umeyama)
# =========================
def umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scale: bool=True) -> Tuple[float, np.ndarray, np.ndarray]:
    mu_src = src.mean(axis=0); mu_dst = dst.mean(axis=0)
    src_demean = src - mu_src; dst_demean = dst - mu_dst
    cov = (dst_demean.T @ src_demean) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    if with_scale:
        var_src = np.sum(src_demean ** 2) / src.shape[0]
        s = np.trace(np.diag(D) @ S) / (var_src + 1e-12)
    else:
        s = 1.0
    t = mu_dst - s * (R @ mu_src)
    return float(s), R, t

def apply_similarity(xy: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (xy @ R.T)) + t

def procrustes_distance(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean(axis=0); b0 = b - b.mean(axis=0)
    sa = np.sqrt(np.sum(a0**2)); sb = np.sqrt(np.sum(b0**2))
    if sa < 1e-8 or sb < 1e-8: return 1e9
    a0 /= sa; b0 /= sb
    return float(np.mean(np.linalg.norm(a0 - b0, axis=1)))


# =========================
# 도너 모델
# =========================
class PairModel:
    def __init__(self, s: float, R: np.ndarray, t: np.ndarray, residual: np.ndarray,
                 xy220_norm: np.ndarray, xy270_norm: np.ndarray, model_id: str):
        self.s = float(s); self.R = R.astype(np.float64); self.t = t.astype(np.float64)
        self.residual = residual.astype(np.float64)
        self.xy220_norm = xy220_norm.astype(np.float64)
        self.xy270_norm = xy270_norm.astype(np.float64)
        self.model_id = model_id

    def predict_on(self, xy220_norm: np.ndarray, residual_weight: float=1.0) -> np.ndarray:
        base = apply_similarity(xy220_norm, self.s, self.R, self.t)
        return base + residual_weight * self.residual

def fit_pair_model(xy220: np.ndarray, xy270: np.ndarray, model_id: str) -> PairModel:
    s, R, t = umeyama_similarity(xy220, xy270, with_scale=True)
    base = apply_similarity(xy220, s, R, t)
    residual = xy270 - base
    return PairModel(s, R, t, residual, xy220, xy270, model_id)


# =========================
# PCA (잔차 압축/복원)
# =========================
class ResidualPCA:
    def __init__(self, mean_vec: np.ndarray, Vt: np.ndarray, var_ratio: np.ndarray):
        self.mean_vec = mean_vec
        self.Vt = Vt
        self.var_ratio = var_ratio

    @staticmethod
    def from_models(models: List[PairModel], var_thresh: float=0.97) -> "ResidualPCA":
        mats = [m.residual.reshape(-1) for m in models]
        X = np.vstack(mats)
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        U, S, Vt_full = np.linalg.svd(Xc, full_matrices=False)
        var = (S**2); var_ratio_all = var / (var.sum() + 1e-12)
        cum = np.cumsum(var_ratio_all)
        K = int(np.searchsorted(cum, var_thresh) + 1)
        K = max(1, min(K, Vt_full.shape[0]))
        Vt = Vt_full[:K]
        return ResidualPCA(mean_vec=mu.reshape(-1), Vt=Vt, var_ratio=var_ratio_all[:K])

    def encode(self, residual: np.ndarray) -> np.ndarray:
        r = residual.reshape(-1)
        z = (r - self.mean_vec) @ self.Vt.T
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        r = self.mean_vec + z @ self.Vt
        return r.reshape(-1, 2)


# =========================
# 예측 파이프라인
# =========================
def consensus_spread(preds: List[np.ndarray]) -> float:
    arr = np.stack(preds, axis=0)
    std = arr.std(axis=0)
    return float(std.mean())

def polygon_area_abs(xy: np.ndarray) -> float:
    return abs(polygon_area(xy))

def area_scale_to_target(xy: np.ndarray, target_area: float) -> np.ndarray:
    cur_area = polygon_area_abs(xy)
    if cur_area < 1e-12: return xy.copy()
    s = math.sqrt(target_area / cur_area)
    return xy * s

def heel_resnap(xy: np.ndarray) -> np.ndarray:
    # heel-first 상태이므로 index 0이 뒤꿈치여야 함
    return xy - xy[0]

def area_resnap(xy: np.ndarray, area_ratio: float) -> np.ndarray:
    target = polygon_area_abs(xy) * area_ratio
    return area_scale_to_target(xy, target)

def pick_knn_models(models: List[PairModel], query_xy220: np.ndarray, k: int) -> List[Tuple[float, PairModel]]:
    scored = [(procrustes_distance(query_xy220, m.xy220_norm), m) for m in models]
    scored.sort(key=lambda x: x[0])
    return scored[:max(1, min(k, len(scored)))]

def predict_270_for(
    xy220_norm: np.ndarray,
    models: List[PairModel],
    mode: str = "pca_knn",
    k: int = 7,
    residual_weight: float = 1.0,
    residual_weight_auto: bool = False,
    auto_lambda_grid: Optional[List[float]] = None,
    consensus_mm: float = 1.5,
    var_thresh: float = 0.97
) -> Tuple[np.ndarray, Dict]:
    diag = {}

    if mode == "scale_only" or len(models) == 0:
        s_avg = float(np.mean([m.s for m in models])) if len(models) else 270.0/220.0
        pred = apply_similarity(xy220_norm, s_avg, np.eye(2), np.zeros(2))
        diag.update(dict(mode="scale_only", s=s_avg))
        return pred, diag

    knn = pick_knn_models(models, xy220_norm, k)
    diag["knn"] = [(float(d), m.model_id) for d, m in knn]

    if mode == "donor_avg":
        lambdas = [residual_weight] if not (residual_weight_auto and auto_lambda_grid) else auto_lambda_grid
        best_pred, best_spread, best_lambda = None, 1e18, None
        for lam in lambdas:
            preds = [m.predict_on(xy220_norm, residual_weight=lam) for _, m in knn]
            eps = 1e-9
            ws = np.array([1.0 / (d + eps) for d, _ in knn], dtype=np.float64); ws /= ws.sum()
            agg = np.tensordot(ws, np.stack(preds, axis=0), axes=(0, 0))
            spr = consensus_spread(preds)
            if spr < best_spread:
                best_spread, best_pred, best_lambda = spr, agg, lam
        diag.update(dict(mode="donor_avg", best_lambda=best_lambda, consensus_spread=best_spread))
        if residual_weight_auto and best_spread > consensus_mm:
            s_avg = float(np.mean([m.s for m in models]))
            pred = apply_similarity(xy220_norm, s_avg, np.eye(2), np.zeros(2))
            diag["fallback"] = "scale_only_by_consensus"
            return pred, diag
        return best_pred, diag

    # pca_knn
    pca = ResidualPCA.from_models([m for _, m in knn], var_thresh=var_thresh)
    diag["pca_K"] = int(pca.Vt.shape[0])
    eps = 1e-9
    ws = np.array([1.0 / (d + eps) for d, _ in knn], dtype=np.float64); ws /= ws.sum()

    Z = np.vstack([pca.encode(m.residual) for (d, m) in knn])
    z_bar = ws @ Z
    res_pred = pca.decode(z_bar)

    if residual_weight_auto and auto_lambda_grid:
        best_pred, best_spread, best_lambda = None, 1e18, None
        for lam in auto_lambda_grid:
            preds = []
            for (d, m) in knn:
                base = apply_similarity(xy220_norm, m.s, m.R, m.t)
                preds.append(base + lam * res_pred)
            spr = consensus_spread(preds)
            if spr < best_spread:
                best_spread, best_lambda = spr, lam
                bases = [apply_similarity(xy220_norm, m.s, m.R, m.t) for _, m in knn]
                base_agg = np.tensordot(ws, np.stack(bases, axis=0), axes=(0, 0))
                candidate = base_agg + lam * res_pred
                best_pred = candidate
        diag.update(dict(mode="pca_knn", pca_consensus=dict(best_lambda=best_lambda, consensus_spread=best_spread)))
        if best_spread > consensus_mm:
            s_avg = float(np.mean([m.s for m in models]))
            pred = apply_similarity(xy220_norm, s_avg, np.eye(2), np.zeros(2))
            diag["fallback"] = "scale_only_by_consensus"
            return pred, diag
        return best_pred, diag
    else:
        bases = [apply_similarity(xy220_norm, m.s, m.R, m.t) for _, m in knn]
        base_agg = np.tensordot(ws, np.stack(bases, axis=0), axes=(0, 0))
        pred = base_agg + residual_weight * res_pred
        diag.update(dict(mode="pca_knn", residual_weight=residual_weight))
        return pred, diag



def length_snap_relative(xy: np.ndarray, input_len: float, scale_ratio: float) -> np.ndarray:
    cur_len = xy[:,1].max() - xy[:,1].min()
    target = input_len * scale_ratio
    if cur_len <= 1e-12: 
        return xy.copy()
    s = target / cur_len
    return xy * s


# =========================
# 빌드/튜닝
# =========================
def build_models_from_pairs(
    df_pairs: pd.DataFrame,
    n_points: int,
    invert_y_220: bool = False,
    invert_y_270: bool = False
) -> List[PairModel]:
    cols = list(df_pairs.columns)
    normmap = norm_col_map(cols)

    n_detect = detect_point_count(cols)
    n = min(n_points, n_detect)
    if n == 0:
        raise RuntimeError("좌표 컬럼(x1..yN 또는 X1..YN)이 없습니다.")

    size_col = get_existing_name(normmap, ["size", "Size"])
    if size_col is None:
        raise RuntimeError("사이즈 컬럼('size' 또는 'Size')이 필요합니다. (220/270)")

    model_col = get_existing_name(normmap, ["model", "PairID", "shape", "Shape"])  # 도너 식별자
    by_model = defaultdict(dict)

    for idx, row in df_pairs.iterrows():
        size_val = str(row[size_col]).strip()
        try:
            size_num = int(float(size_val)); size_str = str(size_num)
        except:
            size_str = size_val

        model_id = str(row[model_col]) if model_col is not None else f"m{idx}"

        # *** 중요: 40,39,...,1 역순으로 읽기 ***
        xy_raw = row_to_xy_descending(row, n_detect, normmap)

        if size_str == "220":
            xy220 = shape_preprocess(xy_raw if not invert_y_220 else np.column_stack([xy_raw[:,0], -xy_raw[:,1]]), n_points=n, invert_y=False)
            by_model[model_id]["220"] = xy220
        elif size_str == "270":
            xy270 = shape_preprocess(xy_raw if not invert_y_270 else np.column_stack([xy_raw[:,0], -xy_raw[:,1]]), n_points=n, invert_y=False)
            by_model[model_id]["270"] = xy270

    models = []
    for mid, d in by_model.items():
        if "220" in d and "270" in d:
            models.append(fit_pair_model(d["220"], d["270"], mid))
    return models

def auto_tune_loo(
    models: List[PairModel],
    mode_grid: List[str],
    k_grid: List[int],
    var_grid: List[float],
    lam_grid: List[float],
    consensus_mm: float = 1.5
) -> Dict:
    best = {"rmse": 1e18}
    M = len(models)
    if M <= 2:
        return dict(mode="pca_knn", k=5, var_thresh=0.95, residual_weight=0.75, rmse=float("nan"))

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

    for mode in mode_grid:
        if mode == "scale_only":
            cfgs = [dict(mode=mode)]
        elif mode == "donor_avg":
            cfgs = [dict(mode=mode, k=k, residual_weight=lam) for k in k_grid for lam in lam_grid]
        else:
            cfgs = [dict(mode=mode, k=k, var_thresh=vt, residual_weight=lam) for k in k_grid for vt in var_grid for lam in lam_grid]

        for cfg in cfgs:
            rmses = []
            for i in range(M):
                test = models[i]
                train = [models[j] for j in range(M) if j != i]
                pred, _ = predict_270_for(
                    test.xy220_norm, train,
                    mode=cfg.get("mode", "pca_knn"),
                    k=cfg.get("k", 5),
                    residual_weight=cfg.get("residual_weight", 1.0),
                    residual_weight_auto=False,
                    auto_lambda_grid=None,
                    consensus_mm=consensus_mm,
                    var_thresh=cfg.get("var_thresh", 0.95),
                )
                pred = heel_resnap(pred)
                pred = area_resnap(pred, area_ratio=(270.0/220.0)**2)
                rmses.append(rmse(pred, test.xy270_norm))
            avg = float(np.mean(rmses))
            if avg < best["rmse"]:
                best = dict(cfg); best["rmse"] = avg
    return best


# =========================
# 메인
# =========================
def main():
    cfg = CONFIG.copy()
    print("[CONFIG]", cfg)

    # 1) 도너 모델
    df_pairs = pd.read_csv(cfg["pairs_csv"])
    models = build_models_from_pairs(
        df_pairs,
        n_points=cfg["n_points"],
        invert_y_220=cfg["invert_y_220"],
        invert_y_270=cfg["invert_y_270"],
    )
    if len(models) == 0:
        raise RuntimeError("도너 쌍(같은 식별자에서 220/270)이 없습니다.")

    # 2) (옵션) 자동 튜닝
    mode = cfg["mode"]; k = cfg["k"]; var_thresh = cfg["var_thresh"]; residual_weight = cfg["residual_weight"]
    if cfg["auto_tune"]:
        best = auto_tune_loo(
            models,
            mode_grid=list(cfg["tune_mode_grid"]),
            k_grid=list(cfg["tune_k_grid"]),
            var_grid=list(cfg["tune_var_grid"]),
            lam_grid=list(cfg["tune_lam_grid"]),
            consensus_mm=cfg["consensus_mm"],
        )
        print("[AUTO-TUNE] best:", best)
        mode = best.get("mode", mode)
        k = best.get("k", k)
        var_thresh = best.get("var_thresh", var_thresh)
        residual_weight = best.get("residual_weight", residual_weight)

    # 3) 입력 220 읽기 (동일한 역순 규칙 적용)
    df_in = pd.read_csv(cfg["input_csv"])
    in_normmap = norm_col_map(list(df_in.columns))
    n_detect_in = detect_point_count(list(df_in.columns))
    n = min(cfg["n_points"], n_detect_in)
    if n == 0:
        raise RuntimeError("입력 CSV 좌표 컬럼(x1..yN 또는 X1..YN)이 없습니다.")

    # 출력에 쓸 size 컬럼 이름 결정: 입력에 있으면 그대로, 없으면 'Size' 사용
    size_col_in = get_existing_name(in_normmap, ["size", "Size"])
    out_size_col = size_col_in if size_col_in is not None else "Size"

    out_rows = []

    for _, row in df_in.iterrows():
        # 원본 220 좌표 행(사용자가 원한다면 원본도 다시 저장)
        # -> 입력 자체가 이미 220이므로, 그대로 출력에 복사(좌표는 원본 그대로 유지)
        base_row = row.to_dict()

        # 예측용 전처리(역순 읽기 → heel-first → 정규화)
        xy220_raw = row_to_xy_descending(row, n_detect_in, in_normmap)
        if cfg["invert_y_input"]:
            xy220_raw = np.column_stack([xy220_raw[:,0], -xy220_raw[:,1]])
        L_in = xy220_raw[:,1].max() - xy220_raw[:,1].min()


        xy220_norm = shape_preprocess(
            xy220_raw if not cfg["invert_y_input"] else np.column_stack([xy220_raw[:,0], -xy220_raw[:,1]]),
            n_points=n,
            invert_y=False
        )

        # 예측
        pred_norm, diag = predict_270_for(
            xy220_norm, models,
            mode=mode, k=k,
            residual_weight=residual_weight,
            residual_weight_auto=cfg["residual_weight_auto"],
            auto_lambda_grid=cfg["auto_lambda_grid"],
            consensus_mm=cfg["consensus_mm"],
            var_thresh=var_thresh
        )

        # 후처리
        pred_norm = heel_resnap(pred_norm)
        pred_norm = area_resnap(pred_norm, area_ratio=(270.0/220.0)**2)
        pred_norm = length_snap_relative(pred_norm, input_len=L_in, scale_ratio=(270.0/220.0))
        # ---- 출력 행 구성 ----
        # (A) 원본 220 행: 입력행 그대로(좌표도 원본 그대로 유지)
        out_rows.append(base_row)

        # (B) 270 예측 행: 입력 메타데이터를 복사하고 좌표만 예측 좌표로 교체
        pred_row = base_row.copy()
        pred_row[out_size_col] = CONFIG["pred_size_value"]  # Size=270 같은 값으로 세팅
        pred_row.update(xy_to_row_dict(pred_norm))
        out_rows.append(pred_row)


    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(cfg["output_csv"], index=False)
    print(f"[DONE] Wrote: {cfg['output_csv']}")


if __name__ == "__main__":
    main()