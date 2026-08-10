import os, re, csv, time
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 머신러닝 라이브러리
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, DotProduct
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

# 클러스터링 라이브러리
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

class ShapePredictorEnv:
    def __init__(self, master_csv_path):
        self.master_csv_path = master_csv_path
        self.full_data = self._load_full_master_data(master_csv_path)
        self.all_types = sorted([t for t in self.full_data.keys() if t.startswith("Type")])
        print(f"[Env] 총 {len(self.all_types)}개의 Type을 로드했습니다.")
        self.global_cluster_map = {} 

    # -----------------------------------------------------
    # 1. 데이터 로드 및 유틸리티
    # -----------------------------------------------------
    def _read_text(self, path, encodings=("utf-8-sig","utf-8","cp949","latin-1")):
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f: return f.read()
            except: continue
        raise ValueError("Encoding Error")

    def _load_full_master_data(self, path):
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

    def chordlen_resample(self, P, n):
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
        T = np.zeros_like(P)
        if len(P) >= 2:
            T[1:-1] = P[2:] - P[:-2]
            T[0], T[-1] = P[1] - P[0], P[-1] - P[-2]
        denom = np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        T /= denom
        Nvec = np.stack([-T[:,1], T[:,0]], axis=1)
        return T, Nvec
    
    def pca_major_axis(self, P):
        C = P - P.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
        v1 = Vt[0]
        z1 = (P @ v1)
        heel_idx = int(np.argmin(z1))
        L = float(z1.max() - z1.min())
        return v1, None, heel_idx, L

    def shrink_along_pc1(self, P, target_L):
        v1, _, heel_idx, L_curr = self.pca_major_axis(P)
        if L_curr <= target_L + 1e-9: return P
        heel = P[heel_idx]
        R = P - heel
        r1 = R @ v1
        P_ortho = R - np.outer(r1, v1)
        alpha = target_L / L_curr
        return heel + np.outer(r1 * alpha, v1) + P_ortho

    def enforce_size_caps_monotone(self, P_list, sizes):
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
    # 클러스터 분석 (전체 맵 생성용)
    # -----------------------------------------------------
    def analyze_type_clusters(self, save_dir, threshold=None, n_clusters=4):
        print("\n[Cluster] 전체 데이터셋 구조 분석 시작...")
        
        base_shapes = []
        valid_types = []
        compare_n = 100 

        for t in self.all_types:
            s_map = self.full_data[t]
            if not s_map: continue
            min_s = min(s_map.keys())
            P, _ = s_map[min_s]
            P_res = self.chordlen_resample(P, compare_n)
            P_res -= P_res.mean(axis=0) 
            base_shapes.append(P_res)
            valid_types.append(t)
            
        n = len(base_shapes)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                _, dist_sq, _, _ = self.cyclic_align(base_shapes[i], base_shapes[j])
                dist = np.sqrt(dist_sq / compare_n)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                
        condensed_dist = squareform(dist_matrix)
        Z = linkage(condensed_dist, method='ward')
        
        if n_clusters is not None:
            cluster_ids = fcluster(Z, t=n_clusters, criterion='maxclust')
            cut_mode = f"Fixed Count ({n_clusters})"
        else:
            t_val = threshold if threshold else 10.0
            cluster_ids = fcluster(Z, t=t_val, criterion='distance')
            cut_mode = f"Distance Threshold ({t_val})"

        self.global_cluster_map = {t: int(cid) for t, cid in zip(valid_types, cluster_ids)}
        
        # 결과 저장 (시각화용)
        os.makedirs(save_dir, exist_ok=True)
        self._save_cluster_results(save_dir, valid_types, dist_matrix, cluster_ids, Z, cut_mode)
        print(f"  [Result] {cut_mode} -> Created {len(set(cluster_ids))} Groups.")

    def _save_cluster_results(self, save_dir, valid_types, dist_matrix, cluster_ids, Z, cut_mode):
        with open(os.path.join(save_dir, "Type_Distance_Matrix.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + valid_types)
            for i, row in enumerate(dist_matrix):
                w.writerow([valid_types[i]] + [f"{v:.4f}" for v in row])
        with open(os.path.join(save_dir, "Type_Cluster_Info.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Type", "Cluster_ID"])
            for t, cid in zip(valid_types, cluster_ids):
                w.writerow([t, cid])
        plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=valid_types, leaf_rotation=90)
        plt.title(f"Hierarchical Clustering Dendrogram\n(Mode: {cut_mode})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Type_Dendrogram.png"), dpi=150)
        plt.close()

    # -----------------------------------------------------
    # Best Match Finder 
    # -----------------------------------------------------
    def _find_smart_match(self, target_type, target_shape_res, n_clusters=4):
        
        # 1. DB 준비 (Target 제외)
        db_types = [t for t in self.all_types if t != target_type]
        if not db_types: return [], "No DB (Self)", 999.0

        db_shapes = []
        for t in db_types:
            s_map = self.full_data[t]
            min_s = min(s_map.keys())
            P, _ = s_map[min_s]
            P_res = self.chordlen_resample(P, len(target_shape_res))
            P_res -= P_res.mean(axis=0) # Centering
            db_shapes.append(P_res)

        target_centered = target_shape_res - target_shape_res.mean(axis=0)

        # --- 전략 A: Cluster Centroid Matching (평균 비교) ---
        best_cluster_members = []
        cluster_rmse = 999.0
        best_cluster_id = -1
        
        if len(db_shapes) >= n_clusters:
            # Clustering
            n = len(db_shapes)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    _, dist_sq, _, _ = self.cyclic_align(db_shapes[i], db_shapes[j])
                    dist = np.sqrt(dist_sq / len(target_shape_res))
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
            
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method='ward')
            cluster_ids = fcluster(Z, t=n_clusters, criterion='maxclust')
            
            # Centroid Search
            unique_clusters = set(cluster_ids)
            for cid in unique_clusters:
                indices = [i for i, c in enumerate(cluster_ids) if c == cid]
                members = [db_shapes[i] for i in indices]
                
                # Calculate Centroid
                ref = members[0]
                avg_shape = np.zeros_like(ref)
                for m in members:
                    aligned_m, _, _, _ = self.cyclic_align(ref, m)
                    avg_shape += aligned_m
                centroid = avg_shape / len(members)
                
                # Compare with Target
                _, dist_sq, _, _ = self.cyclic_align(target_centered, centroid)
                rmse = np.sqrt(dist_sq / len(target_centered))
                
                if rmse < cluster_rmse:
                    cluster_rmse = rmse
                    best_cluster_id = cid
                    best_cluster_members = [db_types[i] for i in indices]

        # --- 전략 B: Individual Global Matching (개별 비교) ---
        best_individual_member = []
        individual_rmse = 999.0
        best_individual_name = ""
        
        for i, db_s in enumerate(db_shapes):
            _, dist_sq, _, _ = self.cyclic_align(target_centered, db_s)
            rmse = np.sqrt(dist_sq / len(target_centered))
            
            if rmse < individual_rmse:
                individual_rmse = rmse
                best_individual_member = [db_types[i]]
                best_individual_name = db_types[i]

        # --- 승자 결정 (로그 메시지 강화) ---
        final_members = []
        final_info = ""
        final_rmse = 0.0
        print(f"Strategy A는 {cluster_rmse} 이고, Strategy B는 {individual_rmse}입니다")
        if cluster_rmse < individual_rmse:
            # 전략 A 승리
            final_members = best_cluster_members
            final_info = f"[Strategy A] Cluster Avg (Group {best_cluster_id}, RMSE: {cluster_rmse:.4f})"
            final_rmse = cluster_rmse
        else:
            # 전략 B 승리
            final_members = best_individual_member
            final_info = f"[Strategy B] Individual Best ({best_individual_name}, RMSE: {individual_rmse:.4f})"
            final_rmse = individual_rmse
            
        return final_members, final_info, final_rmse

    # =====================================================
    # Safe Prediction Logic
    # =====================================================
    def _linear_fit_multi(self, x, Y):
        x, Y = np.asarray(x, float), np.asarray(Y, float)
        X = np.stack([x, np.ones_like(x)], axis=1)
        XtX = X.T @ X + 1e-12 * np.eye(2)
        beta = np.linalg.inv(XtX) @ (X.T @ Y)
        return beta[0], beta[1]

    def _linear_predict_multi(self, a, b, x): return a * float(x) + b

    def _blend_to_boundary(self, Y_linear, Y_boundary, dist_mm, tau_mm=8.0):
        gamma = np.exp(-dist_mm / max(tau_mm, 1e-6))
        return gamma * Y_boundary + (1.0 - gamma) * Y_linear

    def linear_piecewise_predict(self, s_train, Y, s_targets):
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

    def fit_predict_safe(self, model_name, x_train, Y, x_test, override_tau=None):
        x_train, Y, x_test = np.asarray(x_train, float), np.asarray(Y, float), np.asarray(x_test, float)
        order = np.argsort(x_train)
        x_train, Y = x_train[order], Y[order]
        X_train_2d = x_train.reshape(-1, 1)

        if len(x_train) < 2: 
            return self.linear_piecewise_predict(x_train, Y, x_test), None

        model = None
        
        if "GPR" in model_name:
            kernel = C(1.0)*RBF(70.0) + WhiteKernel(1e-3) + C(1.0)*DotProduct()
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=0)
            
        elif "SVR" in model_name:
            gamma = 1.0 / (2.0 * (20.0**2) + 1e-12)
            svr_single = SVR(kernel='rbf', C=100.0, epsilon=0.1, gamma=gamma)
            model = MultiOutputRegressor(svr_single)
            
        elif "KRR" in model_name:
            gamma = 1.0 / (2.0 * (20.0**2) + 1e-12)
            def kernel_callable(A, B):
                A = np.asarray(A); B = np.asarray(B)
                if A.ndim == 1: A = A.reshape(-1, 1)
                if B.ndim == 1: B = B.reshape(-1, 1)
                return 1.0*rbf_kernel(A, B, gamma=gamma) + 1.0*linear_kernel(A, B)
            model = KernelRidge(alpha=0.01, kernel=kernel_callable)

        if model: 
            model.fit(X_train_2d, Y)

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

    def _get_sklearn_regressor(self, model_name):
        if "LINEAR" in model_name: return LinearRegression()
        elif "GPR" in model_name:
            kernel = C(1.0)*RBF(70.0) + WhiteKernel(1e-3) + C(1.0)*DotProduct()
            return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, random_state=0)
        elif "SVR" in model_name: return MultiOutputRegressor(SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale'))
        elif "KRR" in model_name: return KernelRidge(alpha=0.01, kernel='rbf', gamma=0.001)
        raise ValueError(f"Unknown Regressor: {model_name}")

    # -----------------------------------------------------
    # 메인 처리 로직 (수정됨: 로그 출력 추가)
    # -----------------------------------------------------
    def process_single_type(self, target_type, model_type, target_sizes):
        target_map = self.full_data[target_type]
        min_size = min(target_map.keys())
        P_new, side_str = target_map[min_size]
        
        L = len(P_new)
        P_new_res = self.chordlen_resample(P_new, L)

        # 1. 하이브리드 매칭 (클러스터 vs 개별) -> match_info에 "Cluster Match"인지 "Individual Match"인지 적혀있음
        selected_members, match_info, rmse_val = self._find_smart_match(target_type, P_new_res, n_clusters=4)
        
        # 2. 안전장치 (Threshold Check)
        THRESHOLD_RMSE = 3.5
        
        track = []
        if rmse_val > THRESHOLD_RMSE:
            # [Case 3] Ratio (Fallback)
            strategy_log = f"🛑 [Fallback: Ratio] RMSE({rmse_val:.2f}) > {THRESHOLD_RMSE} (No similar data)"
            matched_type = strategy_log
            track = sorted([(s, p) for s, (p, _) in target_map.items()], key=lambda x: x[0])
            
            # 콘솔에 로그 출력
            print(f"    -> {strategy_log}")
            
        else:
            # [Case 1 or 2] ML (Cluster or Individual)
            # match_info 예시: "Cluster Match (RMSE: 1.2)" 또는 "Individual Match (Type04, RMSE: 1.1)"
            strategy_log = f"✅ [ML Selected] {match_info}"
            matched_type = strategy_log
            
            for t in selected_members:
                s_map = self.full_data[t]
                for s, (p, _) in s_map.items():
                    track.append((s, p))
            track.sort(key=lambda x: x[0])
            
            # 콘솔에 로그 출력
            print(f"    -> {strategy_log}")

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
                summary.append({"Type": t_type, "Matched_With": matched})
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
                w = csv.DictWriter(f, fieldnames=["Type", "Matched_With"])
                w.writeheader()
                w.writerows(summary)