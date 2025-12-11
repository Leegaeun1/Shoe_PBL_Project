import os, re, csv, time
import numpy as np
import joblib

# 머신러닝 라이브러리
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, DotProduct
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

# =========================================================
# [클래스] ShapePredictorEnv: "Legacy Logic" 
# =========================================================
class ShapePredictorEnv:
    # -----------------------------------------------------
    # 0. 초기화
    # -----------------------------------------------------
    def __init__(self, master_csv_path):
        '''데이터 로드 후 존재하는 모든 Type목록 저장'''
        self.master_csv_path = master_csv_path
        self.full_data = self._load_full_master_data(master_csv_path)
        self.all_types = sorted([t for t in self.full_data.keys() if t.startswith("Type")])
        print(f"[Env] 총 {len(self.all_types)}개의 Type을 로드했습니다.")

    # -----------------------------------------------------
    # 1. 데이터 로드
    # -----------------------------------------------------
    def _read_text(self, path, encodings=("utf-8-sig","utf-8","cp949","latin-1")):
        '''한글 깨짐 또는 에러 방지'''
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f: return f.read()
            except: continue
        raise ValueError("Encoding Error")

    def _load_full_master_data(self, path):
        '''데이터를 Type: { Size : (좌표 배열, Side문자열)} 형태의 딕셔너리로 구조화하여 반환'''
        text = self._read_text(path)
        data_dict = {}
        _NUM = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][\+\-]?\d+)?$')
        header_skipped = False
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            if not header_skipped:
                if "size" in ln.lower() and "x1" in ln.lower():
                    header_skipped = True; continue
            toks = [t.strip() for t in re.split(r"[,\s]+", ln) if t.strip()]
            if len(toks) < 5: continue
            try:
                type_str, side_str = toks[0], toks[1]
                if not _NUM.match(toks[2]): continue
                size = int(round(float(toks[2])))
                xy = np.array([float(v) for v in toks[3:] if _NUM.match(v)], float)
                if len(xy) < 4: continue
                P = xy.reshape(-1, 2)
                if type_str not in data_dict: data_dict[type_str] = {}
                data_dict[type_str][size] = (P, side_str)
            except: continue
        return data_dict

    # -----------------------------------------------------
    # 2. 기하 유틸리티
    # -----------------------------------------------------
    def chordlen_resample(self, P, n):
        '''입력된 좌표 P를 n개의 점으로 다시 찍음
           점 사이의 거리를 기준으로 등 간격으로 점을 배치'''
        P = np.asarray(P, float)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P)>1 else np.array([])
        u = np.zeros(len(P)); 
        if len(P)>1: u[1:] = np.cumsum(seg)
        L = u[-1]
        if L <= 1e-9: return np.repeat(P[:1], n, axis=0)
        u /= L
        s = np.linspace(0,1,n,endpoint=True)
        return np.stack([np.interp(s, u, P[:,0]), np.interp(s, u, P[:,1])], axis=1)

    def cyclic_align(self, P, Q):
        '''P와 가장 오차가 적은 배치를 찾아낸다.
           Q의 점들을 한칸씩 밀거나 뒤집음'''
        n = len(P)
        best = (None, 1e30, 0, False)
        for rev in [False, True]:
            R = Q[::-1].copy() if rev else Q.copy()
            for k in range(n):
                Rk = np.roll(R, -k, axis=0)
                dist = np.sum((P-Rk)**2)
                if dist < best[1]: best = (Rk, dist, k, rev)
        return best

    def tangents_normals(self, P):
        '''접선 벡터와 법선 벡터 계산
           길이 늘리기(접선)
           볼/두께 방향(법선)'''
        T = np.zeros_like(P)
        if len(P) >= 2:
            T[1:-1] = P[2:] - P[:-2]
            T[0], T[-1] = P[1] - P[0], P[-1] - P[-2]
        denom = np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        T /= denom
        Nvec = np.stack([-T[:,1], T[:,0]], axis=1)
        return T, Nvec
    
    def pca_major_axis(self, P):
        '''PCA를 이용해 형상의 주축(가장 긴 방향)을 찾음. 
           => 신발의 실제 길이(L) 계산'''
        C = P - P.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
        v1 = Vt[0]
        z1 = (P @ v1)
        heel_idx = int(np.argmin(z1))
        L = float(z1.max() - z1.min())
        return v1, None, heel_idx, L

    def shrink_along_pc1(self, P, target_L):
        '''형상 P를 주축 방향으로 강제로 줄이거나 늘려서 목표 길이 target_L에 맞춤'''
        v1, _, heel_idx, L_curr = self.pca_major_axis(P)
        if L_curr <= target_L + 1e-9: return P
        heel = P[heel_idx]
        R = P - heel
        r1 = R @ v1
        P_ortho = R - np.outer(r1, v1)
        alpha = target_L / L_curr
        return heel + np.outer(r1 * alpha, v1) + P_ortho

    def enforce_size_caps_monotone(self, P_list, sizes):
        '''작은 사이즈보다 큰 사이즈의 형상이 더 짧게 예측 -> 강제로 길이 보정하여 모순 없앰'''
        n = len(P_list)
        L_pred = []
        for P in P_list:
            _, _, _, L = self.pca_major_axis(P)
            L_pred.append(L)
        
        L_adj = np.array(L_pred, float)
        for i in range(n-2, -1, -1):
            L_adj[i] = min(L_adj[i], L_adj[i+1])
            
        P_adj_list = []
        for P, Lp, La in zip(P_list, L_pred, L_adj):
            if Lp <= La + 1e-9: P_adj_list.append(P)
            else: P_adj_list.append(self.shrink_along_pc1(P, La))
        return P_adj_list

    # -----------------------------------------------------
    # 3. Best Match
    # -----------------------------------------------------
    def find_best_track(self, train_dict, P_target):
        '''예측 대상(P_target)과 가장 닮은 훈련 데이터를 찾음'''
        L = len(P_target)
        best_match = (None, 1e30, None)
        for t_type, s_map in train_dict.items():
            if not s_map: continue
            min_s = min(s_map.keys())
            base_P, _ = s_map[min_s]
            base_res = self.chordlen_resample(base_P, L)
            _, sc, _, _ = self.cyclic_align(P_target, base_res)
            if sc < best_match[1]:
                track = sorted([(s, p) for s, (p, _) in s_map.items()], key=lambda x: x[0])
                best_match = (t_type, sc, track)
        if best_match[0] is None: raise ValueError("No matching track found.")
        return best_match[2], best_match[0]

    # =====================================================
    # Safe Prediction Logic
    # =====================================================
    def _linear_fit_multi(self, x, Y):
        '''단순 선형 회귀 수행.'''
        x, Y = np.asarray(x, float), np.asarray(Y, float)
        X = np.stack([x, np.ones_like(x)], axis=1)
        XtX = X.T @ X + 1e-12 * np.eye(2)
        beta = np.linalg.inv(XtX) @ (X.T @ Y)
        return beta[0], beta[1]

    def _linear_predict_multi(self, a, b, x): return a * float(x) + b
    '''단순 선형 회귀 수행.'''

    def _blend_to_boundary(self, Y_linear, Y_boundary, dist_mm, tau_mm=8.0):
        '''블렌딩 함수 : 학습 데이터 경계에서 멀어질수록 머신러닝 값 대신 선형 예측 값을 사용하도록 가중치를 섞음.
           dist_mm가 멀어질수록 선형 모델의 비중을 높임'''
        gamma = np.exp(-dist_mm / max(tau_mm, 1e-6))
        return gamma * Y_boundary + (1.0 - gamma) * Y_linear

    def linear_piecewise_predict(self, s_train, Y, s_targets):
        '''데이터 점이 너무 적을때(2개 미만) 사용하는 단순 선간 보간법'''
        s_train, Y = np.array(s_train, float), np.asarray(Y, float)
        out = np.zeros((len(s_targets), Y.shape[1]), float)
        order = np.argsort(s_train)
        s_train, Y = s_train[order], Y[order]
        for i, st in enumerate(s_targets):
            if st <= s_train[0]: a, b = 0, min(1, len(s_train)-1)
            elif st >= s_train[-1]: a, b = max(0, len(s_train)-2), len(s_train)-1
            else:
                idx = np.searchsorted(s_train, st)
                a, b = idx-1, idx
            denom = (s_train[b]-s_train[a]) if b!=a else 1.0
            t = (st - s_train[a]) / (denom + 1e-12)
            out[i] = (1-t)*Y[a] + t*Y[b]
        return out

    # -----------------------------------------------------
    # 4. 모델별 Fit & Predict
    # -----------------------------------------------------
    def fit_predict_safe(self, model_name, x_train, Y, x_test, override_tau=None):
        """핵심 예측 엔진
           1. 데이터 정렬 및 2차원 변환
           2. 모델에 따라 GPR, SVR, KRR 객체를 생성하고 학습(fit)시킴
           3. 학습 범위 내 : 머신러닝 모델 결과 사용, 학습 범위 밖 : 데이터 끝부분의 선형 기울기를 구해 선형으로 예측하되,
              경계면에서는 부드럽게 이어줌 => 안정적으로 예측"""
        x_train, Y, x_test = np.asarray(x_train, float), np.asarray(Y, float), np.asarray(x_test, float)
        order = np.argsort(x_train)
        x_train, Y = x_train[order], Y[order]
        X_train_2d = x_train.reshape(-1, 1)

        if len(x_train) < 2: 
            return self.linear_piecewise_predict(x_train, Y, x_test), None

        model = None
        
        if "GPR" in model_name:
            kernel = C(1.0)*RBF(70.0) + WhiteKernel(1e-3) + C(1.0)*DotProduct()
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=0, normalize_y=True)
            
        elif "SVR" in model_name:
            gamma = 1.0 / (2.0 * (20.0**2) + 1e-12)
            svr_single = SVR(kernel='rbf', C=100.0, epsilon=0.1, gamma=gamma)
            model = MultiOutputRegressor(svr_single)
            
        elif "KRR" in model_name:
            gamma = 1.0 / (2.0 * (20.0**2) + 1e-12)
            def kernel_callable(A, B):
                A = np.asarray(A)
                B = np.asarray(B)
                if A.ndim == 1: A = A.reshape(-1, 1)
                if B.ndim == 1: B = B.reshape(-1, 1)
                return 1.0*rbf_kernel(A, B, gamma=gamma) + 1.0*linear_kernel(A, B)
            model = KernelRidge(alpha=0.01, kernel=kernel_callable)

        if model: model.fit(X_train_2d, Y)

        tail_k = 3
        if override_tau is not None: tail_tau = override_tau
        elif "GPR" in model_name: tail_tau = 1.0 
        elif "SVR" in model_name: tail_tau = 2.0
        elif "KRR" in model_name: tail_tau = 8.0
        else: tail_tau = 2.0

        xmin, xmax = x_train[0], x_train[-1]
        out = np.zeros((len(x_test), Y.shape[1]), float)
        
        kL = min(max(tail_k, 2), len(x_train))
        aL, bL = self._linear_fit_multi(x_train[:kL], Y[:kL])
        aR, bR = self._linear_fit_multi(x_train[-kL:], Y[-kL:])

        for i, st in enumerate(x_test):
            if st < xmin:
                y_lin = self._linear_predict_multi(aL, bL, st)
                out[i] = self._blend_to_boundary(y_lin, Y[0], (xmin - st), tail_tau)
            elif st > xmax:
                y_lin = self._linear_predict_multi(aR, bR, st)
                out[i] = self._blend_to_boundary(y_lin, Y[-1], (st - xmax), tail_tau)
            else:
                if "GPR" in model_name and hasattr(model, 'predict'):
                    out[i] = model.predict(np.array([[st]]))[0]
                else:
                    out[i] = model.predict(np.array([[st]]))[0]
        return out, model

    # -----------------------------------------------------
    # PCA Hybrid용 Factory
    # -----------------------------------------------------
    def _get_sklearn_regressor(self, model_name):
        if "LINEAR" in model_name:
            return LinearRegression()
        elif "GPR" in model_name:
            kernel = C(1.0)*RBF(70.0) + WhiteKernel(1e-3) + C(1.0)*DotProduct()
            return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, random_state=0)
        elif "SVR" in model_name:
            return MultiOutputRegressor(SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale'))
        elif "KRR" in model_name:
            return KernelRidge(alpha=0.01, kernel='rbf', gamma=0.001)
        raise ValueError(f"Unknown Regressor: {model_name}")

    # -----------------------------------------------------
    # 메인 처리 로직
    # -----------------------------------------------------
    def process_single_type(self, target_type, model_type, target_sizes):
        '''하나의 타입을 처리하는 함수
           1. 타겟 데이터 로드 및 리샘플링
           2. 유사한 참조 모델 찾기
           3. 참조 모델의 형상 정렬
           4. 모델링 분기 : PCA 또는 일반
           5. 결과 보정 후 CSV 행 데이터 생성'''
        target_map = self.full_data[target_type]
        min_size = min(target_map.keys())
        P_new, side_str = target_map[min_size]
        train_dict = {k:v for k,v in self.full_data.items() if k != target_type}
        
        L = len(P_new)
        P_new_res = self.chordlen_resample(P_new, L)
        try:
            track, matched_type = self.find_best_track(train_dict, P_new_res)
        except: return [], "Match Fail", 0

        sizes_train = np.array([s for s,_ in track])
        Ps_train = [self.chordlen_resample(p, L) for _,p in track]
        base_P = Ps_train[0]
        
        aligned_Ps = []
        for P in Ps_train:
            Q, _, _, _ = self.cyclic_align(base_P, P)
            aligned_Ps.append(Q)
        P_new_aligned, _, _, _ = self.cyclic_align(base_P, P_new_res)

        if "PCA" in model_type:
            Y_flat = np.stack([(P - base_P).reshape(-1) for P in aligned_Ps])
            n_comp = min(len(Y_flat)-1, 20)
            if n_comp < 1: n_comp = 1
            pca = PCA(n_components=n_comp, whiten=True, random_state=0)
            Z = pca.fit_transform(Y_flat)
            
            regressor = self._get_sklearn_regressor(model_type)
            regressor.fit(sizes_train.reshape(-1,1), Z)
            
            Z_pred = regressor.predict(target_sizes.reshape(-1,1))
            Y_pred_flat = pca.inverse_transform(Z_pred)
            
            results = [P_new_aligned + Y_pred_flat[i].reshape(L,2) for i in range(len(target_sizes))]
            model_obj = (pca, regressor)

        else:
            T, Nvec = self.tangents_normals(base_P)
            Ydt = np.stack([((P-base_P)*T).sum(axis=1) for P in aligned_Ps])
            Ydn = np.stack([((P-base_P)*Nvec).sum(axis=1) for P in aligned_Ps])
            
            override_tau_dn = None
            if "GPR" in model_type: override_tau_dn = 0.1
            if "SVR" in model_type: override_tau_dn = 0.2
            
            dt_pred, m_dt = self.fit_predict_safe(model_type, sizes_train, Ydt, target_sizes)
            dn_pred, m_dn = self.fit_predict_safe(model_type, sizes_train, Ydn, target_sizes, override_tau=override_tau_dn)
            
            Tn, Nn = self.tangents_normals(P_new_aligned)
            results = []
            for i in range(len(target_sizes)):
                P_final = P_new_aligned + Tn*dt_pred[i][:,None] + Nn*dn_pred[i][:,None]
                results.append(P_final)
            model_obj = (m_dt, m_dn)

        results = self.enforce_size_caps_monotone(results, target_sizes)

        csv_rows = []
        for s, P in zip(target_sizes, results):
            row = [target_type, side_str, int(s)] + [f"{v:.6f}" for v in P.reshape(-1)]
            csv_rows.append(row)
        return csv_rows, matched_type, model_obj

    def run_prediction_all_types(self, model_type, target_sizes, save_path=None, save_model=False):
        all_results = []
        summary = []
        start_total = time.perf_counter()
        
        for t_type in self.all_types:
            rows, matched, model = self.process_single_type(t_type, model_type, target_sizes)
            if rows:
                all_results.extend(rows)
                summary.append({"Type": t_type, "Match": matched})
            else:
                print(f"  [Fail] {t_type}")

        elapsed = time.perf_counter() - start_total
        print(f"  [Done] Total Time: {elapsed:.2f}s")
            
        if save_path and all_results:
            n_pts = (len(all_results[0]) - 3) // 2
            header = ["Type", "side", "size"] + [f"{ax}{i}" for i in range(1, n_pts+1) for ax in ("x","y")]
            with open(save_path, "w", newline="") as f:
                csv.writer(f).writerow(header)
                csv.writer(f).writerows(all_results)
            print(f"  [Save] {save_path}")
            
            sum_path = save_path.replace(".csv", "_summary.csv")
            with open(sum_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["Type", "Match"])
                w.writeheader()
                w.writerows(summary)