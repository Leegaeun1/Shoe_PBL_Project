# 이건 PCA랑 RATIO랑 같게나옴...아..................원래 안이랬는데.......


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_series_from_new_mm_PCA_V-GE_TYPE.py

- [MODIFIED] TEST_TYPE과 TEST_SIZE_MM을 홀드아웃하여 P_new로 사용.
- 학습 데이터(TEST_TYPE 제외)에서 P_new와 가장 가까운 'Type'을 자동 선택.
- 선택된 Type의 '최소 사이즈'를 베이스로 삼아,
  해당 Type의 (P_s - base) 변형 벡터(Yd)를 PCA로 학습.
- PCA 주성분(Z)과 사이즈(s)의 관계를 선형회귀(LR)로 학습.
- LR -> PCA 역변환으로 dt, dn을 예측하고 new_sample에 합성.
- [V-GE] 예측 형상은 PCA(PC1) 길이가 s+tol_mm를 넘지 않도록 캡 및 단조 보정
"""

import os, re, csv
import numpy as np
import time
from datetime import datetime

Data_DIR = "Fin_Excel_Data_CTRL40"

# ★ 테스트에 사용할 타입(hold-out 타입)과 사이즈
TEST_TYPE 	 = "Type03" 	# 예측 시리즈의 기준 타입
TEST_SIZE_MM = 230 			# 예측 시리즈의 기준 사이즈 (P_new)

MASTER_CSV =os.path.join(
	Data_DIR,
	"control_points_master_L_20251118.csv"
)

TRAIN_CSV = MASTER_CSV
NEW_CSV 	= MASTER_CSV # P_new 로드에 사용 (경로만 동일)

SAVE_PRED =os.path.join(
	Data_DIR,
	f"pred_Data_230_280_PCA_{TEST_TYPE}.csv"
)
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

# --- Type 인식 로더 (Type-Aware Holdout 버전으로 교체) ---
def load_train_and_new_from_same_csv(path, test_type, test_size):
	"""
	path 하나에서 TEST_TYPE의 TEST_SIZE를 P_new로 분리하고,
	TEST_TYPE의 모든 데이터를 학습용(train_rows)에서 제외한다.
	"""
	text = _read_text(path)
	train_rows = []
	new_rows 	 = []
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
		
		toks = [t.strip() for t in re.split(r"[,\s]+", ln) if t.strip()]
		if len(toks) < 5:
			continue

		try:
			type_str = toks[0]
			side_str = toks[1]

			if not _NUM.match(toks[2]): continue

			size = int(round(float(toks[2])))

			xy_vals_str = toks[3:]
			xy = np.array([float(v) for v in xy_vals_str if _NUM.match(v)], float)

			if len(xy) < 4 or len(xy) % 2 != 0: continue

			P = xy.reshape(-1, 2)

		except Exception:
			continue
			
		# ★ 홀드아웃 로직: TEST_TYPE, TEST_SIZE 행만 new_rows로 분리
		if type_str == test_type and size == test_size:
			new_rows.append((P, side_str))
		# ★ TEST_TYPE과 동일한 모든 행을 학습 데이터에서 제외 (홀드아웃)
		elif type_str != test_type:
			train_rows.append((type_str, size, P))
			
	if not train_rows:
		raise RuntimeError(f"No train rows left after excluding type='{test_type}'.")
	if not new_rows:
		raise RuntimeError(f"No row found for test_type='{test_type}', size={test_size}.")

	P_new, side_new = new_rows[0]
	return train_rows, P_new, side_new

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

# ---------------- 기하 유틸 -----------------
def chordlen_resample(P, n):
	P = np.asarray(P, float)
	seg = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.array([])
	u = np.zeros(len(P))
	if len(P) > 1:
		u[1:] = np.cumsum(seg)
	L = u[-1]
	if L <= 1e-9:
		return np.repeat(P[:1], n, axis=0)
	u /= L
	s = np.linspace(0,1,n,endpoint=True)
	x = np.interp(s, u, P[:,0]); y = np.interp(s, u, P[:,1])
	return np.stack([x,y], axis=1)

def _align_score(P, Q):
	return float(np.sum((P-Q)**2))

def cyclic_align(P, Q):
	n = len(P)
	best = (None, 10**30, 0, False) 	# (Q_best, score, shift, reversed?)
	for rev in [False, True]:
		R = Q[::-1].copy() if rev else Q.copy()
		for k in range(n):
			Rk = np.roll(R, -k, axis=0)
			sc = _align_score(P, Rk)
			if sc < best[1]:
				best = (Rk, sc, k, rev)
	return best 	# Q_best, score, shift, reversed?

def tangents_normals(P):
	N = len(P)
	T = np.zeros_like(P)
	if N >= 2:
		T[1:-1] = P[2:] - P[:-2]
		T[0] 	= P[1] - P[0]
		T[-1] = P[-1] - P[-2]
	denom = np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
	T = T / denom
	Nvec = np.stack([-T[:,1], T[:,0]], axis=1)
	return T, Nvec

# ★★★ [NEW] 헬퍼 함수 추가 (GPR/KRR 버전과 동일) ★★★
def parse_side_from_filename(path, default="N/A"):
	"""
	'control_points_master_L_...csv' 같은 파일명에서 'L' 또는 'R'을 추론합니다.
	"""
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

# -------- 트랙 선택 (Type-Aware PCA 최적화 버전) --------
def find_best_track_and_base(all_rows, P_new_resampled, alpha_pca=0.5):
	"""
	all_rows: [(type, size, P)] (TEST_TYPE 제외)
	P_new_resampled: (L, 2) 새로운 형상 샘플
	
	PCA 방향성 일치도(alpha_pca 가중치)를 추가하여 가장 유사한 Type을 선택한다.
	"""
	tracks_by_type = {}
	for type_str, size, P in all_rows:
		if type_str not in tracks_by_type:
			tracks_by_type[type_str] = []
		tracks_by_type[type_str].append((size, P))

	if not tracks_by_type:
		raise RuntimeError("No tracks found after grouping by Type.")

	L_points = len(P_new_resampled)
	best_match = (None, 10**30, None)
	
	# P_new의 PC1 벡터 미리 계산 (기준)
	# pca_major_axis는 v1, v2, heel_idx, L1을 반환함
	v1_new, _, _, L1_new = pca_major_axis(P_new_resampled) 

	print(f"Finding best match for new sample among {len(tracks_by_type)} Types (alpha_pca={alpha_pca})...")
	
	for type_str, track_list in tracks_by_type.items():
		if not track_list: continue
		
		min_size_row = min(track_list, key=lambda x: x[0])
		base_P_for_type = min_size_row[1]
		
		base_P_resampled = chordlen_resample(base_P_for_type, L_points)
		_, sc_dist, _, _ = cyclic_align(P_new_resampled, base_P_resampled)
		
		# --- ★★★ PCA 방향성 비교 로직 ★★★
		v1_base, _, _, L1_base = pca_major_axis(base_P_resampled)
		dir_similarity = np.abs(np.dot(v1_new, v1_base))
		sc_pca_penalty = alpha_pca * (1.0 - dir_similarity)
		sc_total = sc_dist + sc_pca_penalty
		
		print(f" 	- Type '{type_str}' (base size {min_size_row[0]}): Dist={sc_dist:.2f}, PCA_Pen={sc_pca_penalty:.2f}, Total = {sc_total:.2f}")
		
		if sc_total < best_match[1]:
			best_match = (type_str, sc_total, track_list)

	best_type, best_score, best_track_list = best_match
	if best_type is None:
		raise RuntimeError("Failed to find any matching type track.")

	print(f"==> Best Match Found: Type '{best_type}' (Total Score: {best_score:.2f})")

	base_row = min(best_track_list, key=lambda x: x[0])
	base_P = base_row[1]
	
	# 학습에 사용할 트랙 데이터 필터링 (230-290 범위)
	track_filtered = [(s, P) for s, P in best_track_list if 230 <= s <= 290]
	
	if not track_filtered:
		print(f"Warning: No samples in 230-290 range for '{best_type}'. Using all {len(best_track_list)} samples.")
		track_filtered = best_track_list
		if not track_filtered:
			raise RuntimeError(f"Type '{best_type}' was selected but contains no data.")
	track_filtered.sort(key=lambda x: x[0])
	return track_filtered, base_P, best_type

# -------- PCA 길이 보정 함수 (GPR 버전과 동일) --------
def pca_major_axis(P):
	"""
	P: (N,2)
	반환: v1(PC1 단위벡터), v2(PC2 단위벡터), heel_idx(PC1 최소 투영점 인덱스), L(PC1 길이)
	"""
	C = P - P.mean(axis=0, keepdims=True)
	U, S, Vt = np.linalg.svd(C, full_matrices=False)
	v1 = Vt[0]
	v2 = Vt[1]
	z1 = (P @ v1)
	heel_idx = int(np.argmin(z1))
	L = float(z1.max() - z1.min())
	return v1, v2, heel_idx, L # L1만 반환

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

# ----------------------------- 메인 -----------------------------
def main():
	try:
		from sklearn.decomposition import PCA
		from sklearn.linear_model import LinearRegression
	except Exception as e:
		raise ImportError("scikit-learn이 필요합니다. `pip install scikit-learn`") from e
		
	# 0) 로드: TEST_TYPE, TEST_SIZE_MM을 홀드아웃하여 P_new와 학습 데이터(all_rows) 분리
	all_rows, P_new_raw, side_str = load_train_and_new_from_same_csv(
		TRAIN_CSV,
		test_type=TEST_TYPE,
		test_size=TEST_SIZE_MM,
	) # all_rows: [(type,size,P)...], P_new_raw: (N0,2), side_str: "L"/"R" 등

	# 1) 'Type' 기반 트랙 선택 (PCA 최적화 포함)
	L = len(P_new_raw)
	P_new_resampled = chordlen_resample(P_new_raw, L)
	
	# alpha_pca=0.5로 설정 (필요에 따라 조정)
	# Type 매칭 시 PCA 방향성 일치도(0.5 가중치)를 고려하여 가장 유사한 Type을 선택
	track, base, best_type = find_best_track_and_base(all_rows, P_new_resampled, alpha_pca=0.5) 
	
	sizes_train = [s for s,_ in track]
	Ps = [P for _,P in track]
	if not Ps:
		raise RuntimeError(f"선택된 트랙(Type: {best_type})에 230~290 구간 샘플이 없습니다.")

	# 2) 동일 포인트 수로 리샘플
	Ps = [chordlen_resample(P, L) for P in Ps] 
	base = chordlen_resample(base, L)
	P_new = P_new_resampled # P_new도 L개로 통일

	# 3) base 정렬
	aligned = []
	for s, P in zip(sizes_train, Ps):
		Q_best, _, _, _ = cyclic_align(base, P)
		aligned.append((s, Q_best))
	sizes_train = [s for s,_ in aligned]
	Ps = [P for _,P in aligned]

	# 4) 변형장(d = P_s - base) 구축
	Yd = []
	for P in Ps:
		d = P - base
		Yd.append(d.reshape(-1)) # (L, 2) -> (L*2,)
	Yd = np.stack(Yd, axis=0) # (M, L*2)
	
	# --- 5) PCA + LR 학습 ---
	# 5a. PCA: 변형장(Yd)을 저차원(n_components)으로 압축
	n_samples = Yd.shape[0]
	n_features = Yd.shape[1]
	
	n_components = min(n_samples - 1, n_features, 20) # 최대 20개
	if n_components < 1:
		raise RuntimeError(f"학습 샘플 부족 (n_samples={n_samples}) PCA 학습 불가. (최소 2개 필요)")
		
	print(f"[INFO] Running PCA(n_components={n_components}) on {n_samples} deformation samples...")
	pca_model = PCA(n_components=n_components, whiten=True, random_state=0)
	
	Z = pca_model.fit_transform(Yd) # Yd (M, L*2) -> Z (M, k)
	
	# 5b. LR: 사이즈(s) -> 주성분(Z) 관계 학습
	X = np.array(sizes_train, float).reshape(-1, 1)
	lr_model = LinearRegression()
	lr_model.fit(X, Z)

	# 6) 예측
	sizes_target = np.arange(230, 295, 5, dtype=int)
	Xt = sizes_target.reshape(-1, 1)
	
	Z_pred = lr_model.predict(Xt) # Xt (K, 1) -> Z_pred (K, k)
	
	# Z_pred (K, k) -> Yd_pred (K, L*2)
	Yd_pred = pca_model.inverse_transform(Z_pred)
	
	# 7) 합성
	pred_shapes = []
	for i, s in enumerate(sizes_target):
		d_pred_flat = Yd_pred[i] # (L*2,)
		d_pred = d_pred_flat.reshape(L, 2) # (L, 2)
		P = P_new + d_pred # P_new에 변형량 합성
		pred_shapes.append(P)

	# 8) PC1 길이 가드(<= s+1mm & 단조)
	pred_shapes_adj, L_before, L_after = enforce_size_caps_monotone(
		pred_shapes, sizes_target.tolist(), tol_mm=1.0 
	)

	# 9) 헤더 및 Type/Side 포함하여 저장
	print(f"[INFO] Held-out Type = '{TEST_TYPE}', size = {TEST_SIZE_MM}, side = '{side_str}'")
	print(f"[INFO] Inferred base track Type = '{best_type}' from training data")

	with open(SAVE_PRED, "w", encoding="utf-8", newline="") as f:
		w = csv.writer(f)
		
		num_points = P_new.shape[0] # (L)
		header = ["Type", "side", "size"]
		for i in range(1, num_points + 1):
			header.append(f"x{i}")
			header.append(f"y{i}")
		w.writerow(header) # 헤더 쓰기
		
		for s, P in zip(sizes_target, pred_shapes_adj):
			row = [
				TEST_TYPE, 	
				side_str, 	
				int(s) 	 	
			] + [f"{v:.6f}" for v in P.reshape(-1)]
			
			w.writerow(row)

	print(f"\n[OK] saved -> {SAVE_PRED}")
	print(f"[INFO] Based on new sample's match with Type: '{best_type}'") # best_type 로깅
	print("[INFO] PC1 Lengths before/after cap (mm):")
	print(" 	 sizes :", sizes_target.tolist())
	print(" 	 before:", [round(x,3) for x in L_before.tolist()])
	print(" 	 after :", [round(x,3) for x in L_after.tolist()])

	tr_min, tr_max = (min(sizes_train), max(sizes_train)) if sizes_train else (None, None)
	ext_note = ""
	if tr_min is not None and (sizes_target[0] < tr_min or sizes_target[-1] > tr_max):
		ext_note = " (경고: 일부 타겟 사이즈는 학습 범위를 벗어나 외삽입니다)"
	print(f"[INFO] chosen train sizes from type '{best_type}': {sizes_train[:10]}{'...' if len(sizes_train)>10 else ''}")
	print(f"[INFO] PCA/LR train range: [{tr_min}, {tr_max}] -> predict 230..290{ext_note}")

if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"\n[ERROR] Failed to run prediction: {e}")