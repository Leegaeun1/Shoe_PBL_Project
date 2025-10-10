#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_series_from_new_230_trackselect.py

- ctrl_points.csv 안에 여러 사람(모델)의 230~290 행이 섞여 있어도,
  new_230에 가장 가까운 230을 기준으로 '한 트랙'을 자동 선택해
  235,240,...,290 컨트롤 포인트를 안정적으로 예측합니다.

입력:
  TRAIN_CSV = "ctrl_points.csv"  # 행: size, x1,y1,x2,y2,...
  NEW230_CSV = "new_230.csv"     # 한 줄: (선택) size, x1,y1,...
출력:
  SAVE_PRED = "pred_series.csv"  # 행: size, x1,y1,...

필요: numpy (matplotlib는 미리보기 없어서 불필요)
"""

'''

1. 데이터 로딩: ctrl_points.csv (학습 데이터)와 new_230.csv (새로운 기준 데이터) 파일을 읽어옵니다.

2. 최적의 '트랙' 선택: 학습 데이터에 있는 수많은 230mm 디자인 중에서, 새로 입력된 230mm 디자인과 가장 닮은 것을 찾아냅니다. 그리고 그 '닮은꼴'과 같은 그룹에 속하는 다른 사이즈(235, 240, ...)들을 하나의 '트랙'으로 선택합니다.

3. 데이터 표준화: 선택된 트랙의 모든 디자인과 새로운 230 디자인의 제어점(control point) 개수를 동일하게 맞추고, 시작점을 정렬하여 비교하기 쉽게 만듭니다.

4. '변형 규칙' 분석: 선택된 트랙이 사이즈가 커질 때 각 제어점이 어떻게 움직이는지(곡선의 접선 방향으로 얼마나, 법선 방향으로 얼마나)를 분석하여 '변형장(deformation field)'을 구축합니다.

5. 새로운 사이즈 예측: 위에서 분석한 '변형 규칙'을 새로운 230mm 디자인에 적용하여, 235mm부터 290mm까지의 형태를 예측(합성)합니다.

6. 결과 저장: 예측된 각 사이즈의 제어점 좌표들을 pred_series.csv 파일에 저장합니다.'''

import os, re, csv
import numpy as np

# -------------------- 경로 설정 --------------------
TRAIN_CSV = "control_points_master_20251010.csv" # 학습 데이터(230,240,...290)
NEW230_CSV = "new_230.csv" # 새로운 230 데이터
SAVE_PRED  = "pred_series.csv" # 저장되는 파일
# --------------------------------------------------

# ---------- 강인한 파일 리더 (CSV/BOM/CP949 등) ----------
def _read_text(path, encodings=("utf-8-sig","utf-8","cp949","euc-kr","latin-1")):
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
        if len(xy) % 2 == 1:  # 꼬임 방지
            xy = xy[:-1]
        P = xy.reshape(-1,2)
        if 230 <= size <= 290:
            rows.append((size, P))
    if not rows:
        raise RuntimeError("No train rows in 230..290 found in ctrl_points.csv")
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

    # 맨 앞 값이 사이즈(mm)면 드랍
    if 200.0 <= first[0] <= 320.0 and (len(first)-1) >= 2:
        first = first[1:]

    if len(first) % 2 == 1:
        # 마지막 꼬임 제거
        first = first[:-1]

    if len(first) < 4 or len(first) % 2 != 0:
        raise ValueError("new_230.csv must contain an even number of coordinates (>=4).")

    return first.reshape(-1,2)

# ---------------- 기하 유틸 ----------------
def chordlen_resample(P, n):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.zeros(len(P)); u[1:] = np.cumsum(seg)
    L = u[-1]
    if L <= 1e-9: return np.repeat(P[:1], n, axis=0)
    u /= L
    s = np.linspace(0,1,n,endpoint=True)
    x = np.interp(s, u, P[:,0]); y = np.interp(s, u, P[:,1])
    return np.stack([x,y], axis=1)

def _align_score(P, Q):
    return float(np.sum((P-Q)**2))

def cyclic_align(P, Q):
    """P와 Q를 순환 시프트로 최적 정렬 (정방향/역방향 모두 시도)."""
    n = len(P)
    best = (None, 10**30, 0, False)  # (Q_best, score, shift, reversed?)
    for rev in [False, True]:
        R = Q[::-1].copy() if rev else Q.copy()
        for k in range(n):
            Rk = np.roll(R, -k, axis=0)
            sc = _align_score(P, Rk)
            if sc < best[1]:
                best = (Rk, sc, k, rev)
    return best  # Q_best, score, shift, reversed?

def tangents_normals(P):
    N = len(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0]  = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    Nvec = np.stack([-T[:,1], T[:,0]], axis=1)
    return T, Nvec

# -------- 트랙 선택 & 변형장 구축 (모노토닉 보간) --------
def select_track(rows, P_new_resampled):
    """
    rows: [(size, P)], 여러 사람 데이터 섞여 있음
    1) size==230 후보들 중에서 P_new와 가장 가까운 230을 기준으로 선택
    2) 각 size s에 대해 그 기준 230에 가장 가까운 행을 1개씩 채택
    반환: sorted track [(s, P_s_aligned)], base_230
    """
    # 모든 행을 new_230 길이에 맞춤
    L = len(P_new_resampled)
    rows_std = [(s, chordlen_resample(P, L)) for (s,P) in rows]

    # 1) 230 후보 탐색
    cand_230 = [(s,P) for (s,P) in rows_std if s == 230]
    if not cand_230:
        raise RuntimeError("No size 230 row in training data.")
    # new_230와 가장 가까운 230 찾기
    best = (None, 10**30, None)
    for _, P in cand_230:
        Q_best, sc, _, _ = cyclic_align(P_new_resampled, P)
        if sc < best[1]:
            best = (P, sc, Q_best)
    base_train230, base_dist, base_aligned = best
    # base_aligned는 new_230에 맞춘 정렬 버전이지만,
    # 트랙 구성은 '원본 base' 기준으로 각 s의 가장 가까운 행을 뽑는다.
    base = base_train230

    # 2) 각 사이즈별로 base와 가장 가까운 행 선택
    sizes_all = sorted(set(s for s,_ in rows_std if s >= 230))
    track = []
    for s in sizes_all:
        cand = [P for (ss,P) in rows_std if ss == s]
        if not cand: continue
        # base 기준 정렬 후 최소 거리 채택
        best_s = (None, 10**30)
        for P in cand:
            Q_best, sc, _, _ = cyclic_align(base, P)
            if sc < best_s[1]:
                best_s = (Q_best, sc)
        track.append((s, best_s[0]))

    # 정렬
    track.sort(key=lambda x: x[0])
    return track, base

def interp_piecewise(s_train, Y, s_targets):
    """모노토닉(구간 선형) 보간. s_targets가 s_train 중 하나면 정확히 그 값을 반환."""
    s_train = np.array(s_train, float)
    out = np.zeros((len(s_targets), Y.shape[1]), float)
    for i, st in enumerate(s_targets):
        if st <= s_train[0]:
            a, b = 0, 1 if len(s_train)>1 else 0
        elif st >= s_train[-1]:
            a, b = max(0,len(s_train)-2), len(s_train)-1
        else:
            idx = np.searchsorted(s_train, st)
            a, b = idx-1, idx
        denom = (s_train[b]-s_train[a]) if b!=a else 1.0
        t = (st - s_train[a]) / (denom + 1e-12)
        out[i] = (1-t)*Y[a] + t*Y[b]
    return out

# ----------------------------- 메인 -----------------------------
def main():
    # 0) 로드
    rows = load_train_rows(TRAIN_CSV)         # [(size, P)]
    P_new = load_new230_any(NEW230_CSV)       # (N0,2)

    # 1) 트랙 선택 (new_230와 가장 가까운 230을 기준)
    track, base = select_track(rows, P_new)
    sizes_train = [s for s,_ in track]
    Ps = [P for _,P in track]

    # 2) new_230 길이에 맞춰 모두 동일 포인트 수로 (이미 맞췄지만 안전 차원)
    L = len(P_new)
    Ps = [chordlen_resample(P, L) for P in Ps]
    base = chordlen_resample(base, L)
    P_new = chordlen_resample(P_new, L)

    # 3) 각 P_s를 base에 정렬
    aligned = []
    for s, P in zip(sizes_train, Ps):
        Q_best, _, _, _ = cyclic_align(base, P)
        aligned.append((s, Q_best))
    sizes_train = [s for s,_ in aligned]
    Ps = [P for _,P in aligned]

    # 4) 변형장(dt,dn) 구축 (base 기준)
    T, Nvec = tangents_normals(base)
    Ydt, Ydn = [], []
    for P in Ps:
        d = P - base
        Ydt.append((d*T).sum(axis=1))
        Ydn.append((d*Nvec).sum(axis=1))
    Ydt = np.stack(Ydt, axis=0)  # (M, L)
    Ydn = np.stack(Ydn, axis=0)  # (M, L)

    # 5) new_230의 접선/법선 기반 합성
    Tn, Nn = tangents_normals(P_new)
    sizes_target = np.arange(235, 295, 5, dtype=int)  # 235..290

    dt_pred = interp_piecewise(sizes_train, Ydt, sizes_target.astype(float))
    dn_pred = interp_piecewise(sizes_train, Ydn, sizes_target.astype(float))

    # 6) 합성 & 저장
    with open(SAVE_PRED, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, s in enumerate(sizes_target):
            P = P_new + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None]
            row = [int(s)] + [f"{v:.6f}" for v in P.reshape(-1)]
            w.writerow(row)
    print(f"[OK] saved → {SAVE_PRED}")

    # 디버그: 선택된 트랙 정보
    print(f"[INFO] chosen train sizes: {sizes_train[:10]}{'...' if len(sizes_train)>10 else ''}")
    # 만약 new_230 == 그 트랙의 230과 거의 동일하면, 예측은 그 트랙의 235/240/...을 재현합니다.

if __name__ == "__main__":
    main()
