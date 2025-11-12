#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3_predict_series_from_new_mm_SVR_V-GE.py

- [NEW] SVR(ε-지원 벡터 회귀)를 사용해 사이즈→변형량(dt, dn) 관계를 학습합니다.
- GPR/KRR 버전과 동일하게 'Type'을 자동 선택하고, 최소 사이즈 형상을 base로 삼아
  (P_s - base)의 접선/법선 성분을 예측한 뒤 new_230에 합성합니다.
- [V-GE] 예측 형상은 PCA(PC1) 길이가 s+tol_mm를 넘지 않도록 캡 및 단조 보정합니다.
"""

import os, re, csv
import numpy as np

# -------------------- 경로 설정 (PCA/KRR 스크립트와 동일 스타일) --------------------
Data_DIR = "control_points_master_test_Q"

TRAIN_CSV = os.path.join(Data_DIR, "control_points_master_L_20251104.csv")
NEW_CSV   = os.path.join(Data_DIR, "control_points_master_test_Q.csv")
SAVE_PRED = os.path.join(Data_DIR, "pred_Data_230_280_ge_SVR.csv")
# --------------------------------------------------------------------

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

# -------- 공통 파서들 (PCA/KRR 파일과 동일한 규칙) --------
def load_train_rows(path):
    """
    학습 CSV에서 (type, size, P) 목록을 읽는다.
    CSV 형식: type, side, size, x1, y1, ...
    """
    text = _read_text(path)
    rows = []
    header_skipped = False
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if not header_skipped:
            ln_lower = ln.lower()
            if "size" in ln_lower and "x1" in ln_lower and ("type" in ln_lower or "Type" in ln):
                header_skipped = True
                continue
        toks = [t.strip() for t in re.split(r'[\,\s]+', ln) if t.strip()]
        if len(toks) < 5:
            continue
        try:
            type_str = toks[0]
            if not _NUM.match(toks[2]):
                continue
            size = int(round(float(toks[2])))
            xy_vals_str = toks[3:]
            xy = np.array([float(v) for v in xy_vals_str if _NUM.match(v)], float)
            if len(xy) < 4 or len(xy) % 2 != 0:
                continue
            P = xy.reshape(-1, 2)
            rows.append((type_str, size, P))
        except Exception:
            continue
    if not rows:
        raise RuntimeError("No valid (Type, size, P) rows found in training CSV.")
    return rows


def load_new230_any(path):
    text = _read_text(path)
    first = None
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        vals = _num_tokens(ln)
        if len(vals) >= 2:
            first = np.array(vals, float)
            break
    if first is None:
        raise RuntimeError(f"No numeric row found in {path}")
    # 맨 앞 값이 사이즈(mm)면 드랍
    if 100.0 <= first[0] <= 500.0 and (len(first)-1) >= 2:
        first = first[1:]
    if len(first) % 2 == 1:
        first = first[:-1]
    if len(first) < 4 or len(first) % 2 != 0:
        raise ValueError("new_230.csv must contain an even number of coordinates (>=4).")
    return first.reshape(-1,2)


def parse_side_from_filename(path, default="N/A"):
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    base_lower = base.lower()
    if "_l_" in base_lower or base_lower.endswith("_l"):
        return "L"
    if "_r_" in base_lower or base_lower.endswith("_r"):
        return "R"
    tokens = re.findall(r'[a-zA-Z]+', base)
    for t in tokens:
        if t.upper() == 'L':
            return 'L'
        if t.upper() == 'R':
            return 'R'
    return default

# -------- 기하 유틸 (기존 파일들과 동일 가정) --------
def chordlen_resample(P, L):
    """호장 기반 균등 리샘플링. 기존 스크립트와 동일한 로직을 사용한다고 가정."""
    P = np.asarray(P, float)
    d = np.sqrt(((np.roll(P,-1,axis=0)-P)**2).sum(axis=1))
    t = np.concatenate([[0.0], np.cumsum(d)[:-1]])
    if t[-1] <= 0:
        return P.copy()
    t /= t[-1]
    ts = np.linspace(0,1, L)
    Q = np.empty((L,2), float)
    for j,u in enumerate(ts):
        i = np.searchsorted(t, u, side='right')-1
        i = np.clip(i, 0, len(P)-2)
        w = (u - t[i]) / max(t[i+1]-t[i], 1e-12)
        Q[j] = (1-w)*P[i] + w*P[i+1]
    return Q


def cyclic_align(A, B):
    """A,B (L,2). A에 대해 B를 원형 시프트+반전으로 정렬 (간단 버전)."""
    L = len(A)
    best = (1e30, None)
    for rev in (False, True):
        C = B[::-1] if rev else B
        for k in range(L):
            D = np.roll(C, -k, axis=0)
            sc = np.sqrt(((A-D)**2).sum(axis=1)).mean()
            if sc < best[0]:
                best = (sc, (D, k, rev))
    D, k, rev = best[1]
    return D, best[0], k, rev


def tangents_normals(P):
    P = np.asarray(P, float)
    L = len(P)
    d = np.roll(P,-1,axis=0) - np.roll(P,1,axis=0)
    T = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    N = np.stack([-T[:,1], T[:,0]], axis=1)
    return T, N

# -------- PCA 주축 길이 및 단조 캡 (PCA/GPR 스크립트 동일 로직) --------
def pca_major_axis(P):
    C = P - P.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    v1 = Vt[0]; v2 = Vt[1]
    z1 = (P @ v1)
    heel_idx = int(np.argmin(z1))
    L = float(z1.max() - z1.min())
    return v1, v2, heel_idx, L


def shrink_along_pc1(P, target_L, eps=1e-9):
    v1, v2, heel_idx, L = pca_major_axis(P)
    if L <= target_L + eps:
        return P
    heel = P[heel_idx]
    R = P - heel
    r1 = R @ v1
    r2 = R @ v2
    L_current = float(r1.max() - r1.min())
    if L_current < eps:
        return P
    alpha = target_L / L_current
    r1_new = r1 * alpha
    P_new = heel + np.outer(r1_new, v1) + np.outer(r2, v2)
    return P_new


def enforce_size_caps_monotone(P_list, sizes, tol_mm=1.0):
    n = len(P_list)
    L_pred = []
    for P in P_list:
        _, _, _, L = pca_major_axis(P)
        L_pred.append(L)
    L_pred = np.array(L_pred, float)
    U = np.array(sizes, float) + float(tol_mm)
    L_cap = np.minimum(L_pred, U)
    L_adj = L_cap.copy()
    for i in range(n-2, -1, -1):
        L_adj[i] = min(L_adj[i], L_adj[i+1])
    P_adj_list = []
    for P, Lp, La in zip(P_list, L_pred, L_adj):
        if Lp <= La + 1e-9:
            P_adj_list.append(P)
        else:
            P_adj_list.append(shrink_along_pc1(P, La))
    return P_adj_list, L_pred, L_adj

# ======================= SVR 안전 예측(멀티타깃) =======================
# KRR/GPR 스크립트의 꼬리 가드(국소 선형 + 경계 블렌딩)를 그대로 사용

def _linear_fit_multi(x, Y):
    x = np.asarray(x, float)
    Y = np.asarray(Y, float)
    X = np.vstack([x, np.ones_like(x)]).T
    # 최소제곱 해(a,b) -> y ≈ a*x + b
    a = np.linalg.lstsq(X, Y, rcond=None)[0][0]
    b = np.linalg.lstsq(X, Y, rcond=None)[0][1]
    return a, b


def _linear_predict_multi(a, b, x):
    return a * float(x) + b


def _blend_to_boundary(Y_linear, Y_boundary, dist_mm, tau_mm):
    gamma = np.exp(-float(dist_mm) / max(float(tau_mm), 1e-6))  # 1이면 경계, 0이면 선형외삽
    return gamma * Y_boundary + (1.0 - gamma) * Y_linear


def linear_piecewise_predict(s_train, Y, s_targets):
    s_train = np.array(s_train, float)
    Y = np.asarray(Y, float)
    out = np.zeros((len(s_targets), Y.shape[1]), float)
    order = np.argsort(s_train)
    s_train = s_train[order]; Y = Y[order]
    for i, st in enumerate(s_targets):
        if st <= s_train[0]:
            a, b = 0, min(1, len(s_train)-1)
        elif st >= s_train[-1]:
            a, b = max(0, len(s_train)-2), len(s_train)-1
        else:
            idx = np.searchsorted(s_train, st)
            a, b = idx-1, idx
        denom = (s_train[b]-s_train[a]) if b!=a else 1.0
        t = (st - s_train[a]) / (denom + 1e-12)
        out[i] = (1-t)*Y[a] + t*Y[b]
    return out


def svr_fit_predict_safe(x_train, Y, x_test,
                         C=100.0, epsilon=0.1, length_scale=20.0,
                         tail_k_local=3, tail_tau_mm=1.0):
    """SVR(RBF) + 경계 가드(국소 선형 외삽 + 경계 블렌딩)
    - x_train: (M,) 사이즈
    - Y: (M, D) 다변량 목표 (포인트별 dt 또는 dn)
    - x_test: (K,)
    """

    # SVR 라이브러리 로드
    try:
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
    except Exception as e:
        raise ImportError("scikit-learn이 필요합니다. `pip install scikit-learn`") from e

    # 데이터 배열 변환 및 정렬
    x_train = np.asarray(x_train, float)
    Y = np.asarray(Y, float)
    x_test = np.asarray(x_test, float)

    order = np.argsort(x_train)
    x_train = x_train[order]
    Y = Y[order]

    # 학습 데이터(사이즈) 개수 검사. 2개 미만이면 학습 불가능함! -> 선형으로 예측함
    if x_train.shape[0] < 2:
        return linear_piecewise_predict(x_train, Y, x_test)

    # ------SVR 모델 학습------------
    # RBF gamma 설정: 1/(2*ls^2)
    gamma = 1.0 / (2.0 * (float(length_scale)**2) + 1e-12)
    # RBF 커널 사용, C(규제 강도)와 epsilon(마진 폭) 적용
    model = MultiOutputRegressor(SVR(kernel='rbf', C=float(C), epsilon=float(epsilon), gamma=gamma))

    # 내부구간 예측용 핏, X(사이즈)를 입력으로 Y(변형량) 출력으로 모델 학습!
    X = x_train.reshape(-1,1) # -1부분은 알아서 계산해달라는 뜻. 행의 개수, 1은 열의 개수가 1개여야한다. => M X 1 행렬.
    model.fit(X, Y)
    #--------------------------------

    # 학습 데이터의 최소/최대 사이즈
    xmin, xmax = x_train[0], x_train[-1]
    # shape[0]은 행의 수. 사이즈의 개수, [1]은 열의 수. 컨트롤포인트 수
    K, D = len(x_test), Y.shape[1]
    out = np.zeros((K, D), float)

    # 경계값(Y[0], Y[-1]) 및 국소 선형 계수
    Y_left_boundary  = Y[0] # 최소 사이즈에서의 변형량 값. 외삽 시 블렌딩될 최종 목표값
    Y_right_boundary = Y[-1] # 최대 사이즈에서의 변형량 값. 외삽 시 블렌딩될 최종 목표값
    kL = min(max(int(tail_k_local), 2), len(x_train)) # 국소 선형 회귀에 사용할 데이터 포인트 개수(3개)

    # 왼/오 국소선형 a,b (다변량)
    def _ab(xseg, Yseg):
        Xseg = np.vstack([xseg, np.ones_like(xseg)]).T
        coef, _, _, _ = np.linalg.lstsq(Xseg, Yseg, rcond=None)
        a = coef[0]; b = coef[1]
        return a, b
    aL, bL = _ab(x_train[:kL], Y[:kL])
    aR, bR = _ab(x_train[-kL:], Y[-kL:])

    for i, st in enumerate(x_test):
        if st < xmin:
            y_lin = _linear_predict_multi(aL, bL, st)
            out[i] = _blend_to_boundary(y_lin, Y_left_boundary, dist_mm=(xmin - st), tau_mm=tail_tau_mm)
        elif st > xmax:
            y_lin = _linear_predict_multi(aR, bR, st)
            out[i] = _blend_to_boundary(y_lin, Y_right_boundary, dist_mm=(st - xmax), tau_mm=tail_tau_mm)
        else:
            out[i] = model.predict(np.array([[st]]) )[0]
    return out

# -------- 트랙 선택 (PCA/GPR/KRR와 동일 로직) --------
def find_best_track_and_base(all_rows, P_new_resampled):
    tracks_by_type = {}
    for type_str, size, P in all_rows:
        tracks_by_type.setdefault(type_str, []).append((size, P))
    if not tracks_by_type:
        raise RuntimeError("No tracks found after grouping by Type.")
    L = len(P_new_resampled)
    best_match = (None, 10**30, None)
    for type_str, track_list in tracks_by_type.items():
        if not track_list:
            continue
        min_size_row = min(track_list, key=lambda x: x[0])
        base_P_for_type = min_size_row[1]
        base_P_resampled = chordlen_resample(base_P_for_type, L)
        _, sc, _, _ = cyclic_align(P_new_resampled, base_P_resampled)
        if sc < best_match[1]:
            best_match = (type_str, sc, track_list)
    best_type, best_score, best_track_list = best_match
    if best_type is None:
        raise RuntimeError("Failed to find any matching type track.")
    base_row = min(best_track_list, key=lambda x: x[0])
    base_P = base_row[1]
    track_filtered = [(s, P) for s, P in best_track_list if 230 <= s <= 290]
    if not track_filtered:
        track_filtered = best_track_list
        if not track_filtered:
            raise RuntimeError(f"Type '{best_type}' was selected but contains no data.")
    track_filtered.sort(key=lambda x: x[0])
    return track_filtered, base_P, best_type


# ----------------------------- 메인 -----------------------------
def main():
    # 0) 로드
    all_rows = load_train_rows(TRAIN_CSV)
    P_new_raw = load_new230_any(NEW_CSV)

    # 1) Type 기반 트랙/베이스 선택
    L = len(P_new_raw)
    P_new_resampled = chordlen_resample(P_new_raw, L)
    track, base, best_type = find_best_track_and_base(all_rows, P_new_resampled)

    sizes_train = [s for s,_ in track]
    Ps = [P for _,P in track]
    if not Ps:
        raise RuntimeError(f"선택된 트랙(Type: {best_type})에 230~290 구간 샘플이 없습니다.")

    # 2) 동일 포인트수 리샘플 + Y축 원점 정렬
    L = len(P_new_raw)
    Ps   = [chordlen_resample(P, L) for P in Ps]
    base = chordlen_resample(base, L)
    P_new= chordlen_resample(P_new_raw, L)

    all_P = [P_new, base] + Ps
    min_y = min(P.min(axis=0)[1] for P in all_P)
    if min_y < 0.0:
        dy = -min_y
        P_new[:,1] += dy
        base[:,1]  += dy
        for P in Ps:
            P[:,1] += dy
        print(f"[INFO] Y-axis adjusted by {dy:.3f} to align minimum Y=0.")

    # 3) base 정렬
    aligned = []
    for s, P in zip(sizes_train, Ps):
        Q_best, _, _, _ = cyclic_align(base, P)
        aligned.append((s, Q_best))
    sizes_train = [s for s,_ in aligned]
    Ps = [P for _,P in aligned]

    # 4) 변형장(dt, dn)
    T, Nvec = tangents_normals(base)
    Ydt, Ydn = [], []
    for P in Ps:
        d = P - base
        Ydt.append((d*T).sum(axis=1))
        Ydn.append((d*Nvec).sum(axis=1))
    Ydt = np.stack(Ydt, axis=0)
    Ydn = np.stack(Ydn, axis=0)

    # 5) SVR로 dt(s), dn(s) 예측 (하이퍼파라미터는 소표본 친화적으로 보수 설정)
    sizes_target = np.arange(230, 295, 5, dtype=int)
    DT_TAU = 2.0
    DN_TAU = 0.2

    # 기본값 제안: C=5, epsilon=0.02, length_scale=20mm
    dt_pred = svr_fit_predict_safe(
        sizes_train, Ydt, sizes_target.astype(float),
        C=100.0, epsilon=0.1, length_scale=20.0,
        tail_k_local=3, tail_tau_mm=DT_TAU
    )
    dn_pred = svr_fit_predict_safe(
        sizes_train, Ydn, sizes_target.astype(float),
        C=100.0, epsilon=0.1, length_scale=20.0,
        tail_k_local=3, tail_tau_mm=DN_TAU
    )

    # 6) 합성 → 길이(PC1) 단조 캡
    Tn, Nn = tangents_normals(P_new)
    pred_shapes = []
    for i, s in enumerate(sizes_target):
        P = P_new + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None]
        pred_shapes.append(P)

    pred_shapes_adj, L_before, L_after = enforce_size_caps_monotone(
        pred_shapes, sizes_target.tolist(), tol_mm=1.0
    )

    # 7) 저장 (헤더: Type, side, size, x1,y1,...)
    side_str = parse_side_from_filename(TRAIN_CSV, default="N/A")
    print(f"[INFO] Inferred side='{side_str}' from training data path '{TRAIN_CSV}'")

    with open(SAVE_PRED, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        num_points = P_new.shape[0]
        header = ["Type", "side", "size"]
        for i in range(1, num_points + 1):
            header.append(f"x{i}"); header.append(f"y{i}")
        w.writerow(header)
        for s, P in zip(sizes_target, pred_shapes_adj):
            row = [best_type, side_str, int(s)] + [f"{v:.6f}" for v in P.reshape(-1)]
            w.writerow(row)

    print(f"\n[OK] Saved predictions to -> {SAVE_PRED}")
    print(f"[INFO] Based on new sample's match with Type: '{best_type}'")
    print("[INFO] PC1 lengths before/after cap (mm):")
    print("      sizes :", sizes_target.tolist())
    print("      before:", [round(x,3) for x in L_before.tolist()])
    print("      after :",  [round(x,3) for x in L_after.tolist()])


if __name__ == "__main__":
    main()
