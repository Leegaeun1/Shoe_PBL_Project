#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_series_enhanced.py

- K-최근접 이웃(K-NN) 블렌딩을 통해 예측 정확도 향상.
- PCHIP 보간을 사용하여 사이즈 간 부드러운 형태 변화 생성.
- argparse를 통해 주요 설정을 명령줄에서 제어.
- matplotlib을 이용한 결과 시각화 기능 추가.

입력:
  --train_csv TRAIN_CSV     # 학습 데이터 (ctrl_points.csv)
  --new230_csv NEW230_CSV   # 새로운 230 데이터 (new_230.csv)
출력:
  --save_pred SAVE_PRED     # 예측 결과 저장 (pred_series.csv)
"""

import os
import re
import csv
import argparse
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

# --------------------------------------------------
# 필요 라이브러리:
# pip install numpy scipy matplotlib
# --------------------------------------------------


# ---------- 강인한 파일 리더 (CSV/BOM/CP949 등) ----------
def _read_text(path, encodings=("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1")):
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception as e:
            last_err = e
            continue
    raise last_err

_NUM = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][\+\-]?\d+)?$')
def _num_tokens(line):
    toks = [t for t in line.replace(",", " ").replace(";", " ").split() if t]
    out = []
    for t in toks:
        t = t.strip().lstrip("\ufeff")
        if _NUM.match(t):
            out.append(float(t))
    return out

def load_train_rows(path):
    text = _read_text(path)
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        vals = _num_tokens(ln)
        if len(vals) < 3: continue
        size = int(round(vals[0]))
        xy = np.array(vals[1:], float)
        if len(xy) % 2 == 1:
            xy = xy[:-1]
        P = xy.reshape(-1, 2)
        if 230 <= size <= 290:
            rows.append((size, P))
    if not rows:
        raise RuntimeError(f"No train rows in 230..290 found in {path}")
    return rows

def load_new230_any(path):
    text = _read_text(path)
    first = None
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        vals = _num_tokens(ln)
        if len(vals) >= 2:
            first = np.array(vals, float)
            break
    if first is None:
        raise RuntimeError(f"No numeric row found in {path}")
    if 200.0 <= first[0] <= 320.0 and (len(first) - 1) >= 2:
        first = first[1:]
    if len(first) % 2 == 1:
        first = first[:-1]
    if len(first) < 4 or len(first) % 2 != 0:
        raise ValueError("Input CSV must contain an even number of coordinates (>=4).")
    return first.reshape(-1, 2)

# ---------------- 기하 유틸 ----------------
def chordlen_resample(P, n):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.zeros(len(P)); u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-9: return np.repeat(P[:1], n, axis=0)
    u /= L
    s = np.linspace(0, 1, n, endpoint=True)
    x = np.interp(s, u, P[:, 0]); y = np.interp(s, u, P[:, 1])
    return np.stack([x, y], axis=1)

def _align_score(P, Q):
    return float(np.sum((P - Q)**2))

def cyclic_align(P, Q):
    n = len(P)
    best = (None, 1e30, 0, False)
    for rev in [False, True]:
        R = Q[::-1].copy() if rev else Q.copy()
        for k in range(n):
            Rk = np.roll(R, -k, axis=0)
            sc = _align_score(P, Rk)
            if sc < best[1]:
                best = (Rk, sc, k, rev)
    return best

def tangents_normals(P):
    N = len(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    Nvec = np.stack([-T[:, 1], T[:, 0]], axis=1)
    return T, Nvec

# -------- 트랙 선택 로직 (특정 base 기준) --------
def select_track_from_base(rows_std, base):
    sizes_all = sorted(set(s for s, _ in rows_std if s >= 230))
    track = []
    for s in sizes_all:
        cand = [P for (ss, P) in rows_std if ss == s]
        if not cand: continue
        best_s = (None, 1.0e30)
        for P in cand:
            Q_best, sc, _, _ = cyclic_align(base, P)
            if sc < best_s[1]:
                best_s = (Q_best, sc)
        track.append((s, best_s[0]))
    track.sort(key=lambda x: x[0])
    return track

# -------- 시각화 함수 --------
def visualize_results(p_new, best_track, predictions):
    plt.figure(figsize=(10, 16))
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title("Input (Red) and Best Matched Track (Gray)")
    for s, p in best_track:
        ax1.plot(p[:, 0], p[:, 1], 'k-', alpha=0.3, lw=1)
    ax1.plot(p_new[:, 0], p_new[:, 1], 'r-', lw=2.5, label='New 230 (Input)')
    ax1.legend()
    ax1.set_aspect('equal', 'box')

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title("Prediction Series from Input")
    ax2.plot(p_new[:, 0], p_new[:, 1], 'r--', lw=1.5, alpha=0.6, label='New 230 (Input)')
    colors = plt.cm.viridis(np.linspace(0, 1, len(predictions)))
    for i, (s, p) in enumerate(predictions):
        ax2.plot(p[:, 0], p[:, 1], '-', color=colors[i], label=f'Pred {s}')
    ax2.legend()
    ax2.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.show()

# ----------------------------- 메인 로직 -----------------------------
def run_prediction(args):
    # 1. 데이터 로드
    rows = load_train_rows(args.train_csv)
    P_new_orig = load_new230_any(args.new230_csv)

    # 2. 모든 데이터를 고정된 포인트 수로 표준화
    L = args.points
    P_new = chordlen_resample(P_new_orig, L)
    rows_std = [(s, chordlen_resample(P, L)) for (s, P) in rows]

    # 3. new_230과 가장 가까운 K개의 230 후보(base) 찾기
    cand_230 = [(s, P) for (s, P) in rows_std if s == 230]
    if not cand_230:
        raise RuntimeError("No size 230 found in training data.")
    if len(cand_230) < args.k:
        print(f"[WARN] Requested k={args.k} but only found {len(cand_230)} size 230s. Using k={len(cand_230)}.")
        args.k = len(cand_230)

    distances = []
    for _, P_train230 in cand_230:
        _, score, _, _ = cyclic_align(P_new, P_train230)
        distances.append({'score': score, 'P_train': P_train230})
    
    distances.sort(key=lambda x: x['score'])
    
    best_k_bases = [d['P_train'] for d in distances[:args.k]]
    scores = np.array([d['score'] for d in distances[:args.k]])
    weights = 1.0 / (scores + 1e-9)
    weights /= np.sum(weights)

    print(f"[INFO] Using {args.k}-NN blending. Best match score: {scores[0]:.4f}")

    # 4. 각 K개의 트랙에 대해 변형장 예측 후 가중 평균
    all_dt_preds, all_dn_preds = [], []
    best_track_for_viz = None

    for i, base in enumerate(best_k_bases):
        track = select_track_from_base(rows_std, base)
        if i == 0: best_track_for_viz = track # 시각화를 위해 최고 트랙 저장
        
        sizes_train = np.array([s for s, _ in track])
        Ps = np.array([P for _, P in track])
        
        T_base, N_base = tangents_normals(base)
        D = Ps - base # 변위 벡터 (M, L, 2)
        
        # dt, dn 계산 (벡터 내적 활용)
        Ydt = np.einsum('ijk,jk->ij', D, T_base)
        Ydn = np.einsum('ijk,jk->ij', D, N_base)
        
        sizes_target = np.arange(235, 295, 5)
        
        # Pchip 보간으로 부드러운 예측
        dt_pred = PchipInterpolator(sizes_train, Ydt, axis=0)(sizes_target)
        dn_pred = PchipInterpolator(sizes_train, Ydn, axis=0)(sizes_target)
        all_dt_preds.append(dt_pred)
        all_dn_preds.append(dn_pred)

    # 가중 평균으로 최종 변형장 계산
    final_dt_pred = np.sum(np.stack(all_dt_preds) * weights[:, None, None], axis=0)
    final_dn_pred = np.sum(np.stack(all_dn_preds) * weights[:, None, None], axis=0)

    # 5. new_230의 접선/법선 벡터 계산
    Tn, Nn = tangents_normals(P_new)
    
    # 6. 최종 예측 결과 합성 및 저장
    predictions_for_viz = []
    with open(args.save_pred, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, s in enumerate(sizes_target):
            # 변형 적용: P_pred = P_new + (dt * T_new) + (dn * N_new)
            P = P_new + Tn * final_dt_pred[i][:, None] + Nn * final_dn_pred[i][:, None]
            predictions_for_viz.append((s, P))
            row = [int(s)] + [f"{v:.6f}" for v in P.reshape(-1)]
            w.writerow(row)
    print(f"[OK] Prediction saved -> {args.save_pred}")

    # 7. (선택) 시각화
    if args.plot:
        print("[INFO] Displaying visualization...")
        visualize_results(P_new, best_track_for_viz, predictions_for_viz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict shoe last control point series using K-NN blending.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--train_csv', type=str, default="control_points_master_20250930.csv", 
                        help="Path to training data CSV")
    parser.add_argument('--new230_csv', type=str, default="new_230.csv", 
                        help="Path to new 230 data CSV")
    parser.add_argument('--save_pred', type=str, default="pred_series.csv", 
                        help="Path to save prediction CSV")
    parser.add_argument('-k', '--k', type=int, default=3, 
                        help="Number of nearest tracks to blend (K for K-NN)")
    parser.add_argument('-p', '--points', type=int, default=100, 
                        help="Number of points for resampling all curves")
    parser.add_argument('--plot', action='store_true', 
                        help="Visualize the input, best matched track, and prediction results")
    
    args = parser.parse_args()
    
    try:
        run_prediction(args)
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")