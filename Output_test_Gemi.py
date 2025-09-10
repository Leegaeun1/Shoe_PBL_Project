import pandas as pd
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from itertools import product
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=RuntimeWarning)

# -----------------------------------------------------------------------------
# 0. 유틸리티 및 데이터 로딩 함수
# -----------------------------------------------------------------------------

def umeyama_similarity(X, Y):
    """
    두 점 집합 X와 Y 사이의 최적 유사 변환(크기, 회전, 이동)을 찾습니다.
    (Umeyama, 1991) 알고리즘을 기반으로 합니다.

    Returns:
        s (float): 스케일 팩터
        R (np.ndarray): 회전 행렬 (2x2)
        t (np.ndarray): 이동 벡터 (2x1)
    """
    m, n = X.shape
    assert m == 2, "점 집합은 2xN 형태여야 합니다."

    # 중심 계산
    mean_X = np.mean(X, axis=1, keepdims=True)
    mean_Y = np.mean(Y, axis=1, keepdims=True)

    # 중심화
    X_c = X - mean_X
    Y_c = Y - mean_Y

    # 공분산 행렬
    S_XY = np.dot(Y_c, X_c.T) / n

    # SVD
    U, D, Vt = np.linalg.svd(S_XY)
    V = Vt.T

    # 회전 행렬 R 계산
    S = np.eye(m)
    if np.linalg.det(S_XY) < 0:
        S[-1, -1] = -1

    R = np.dot(U, np.dot(S, Vt))

    # 스케일 s 계산
    var_X = np.var(X, axis=1).sum()
    s = np.trace(np.diag(D) @ S) / var_X

    # 이동 t 계산
    t = mean_Y - s * np.dot(R, mean_X)

    return s, R, t

def polygon_area(x, y):
    """다각형의 면적을 계산합니다."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# -----------------------------------------------------------------------------
# 1. 전처리 및 정규화 함수
# -----------------------------------------------------------------------------

def heel_index_max_y(P):
    """가장 큰 y값을 가진 점을 뒤꿈치 점으로 간주하고 인덱스를 반환합니다."""
    return np.argmax(P[:, 1])

def closed_bspline_resample(P, n_points=40):
    """폐곡선 B-Spline을 사용하여 점들을 균일한 간격으로 다시 샘플링합니다."""
    tck, u = splprep([P[:, 0], P[:, 1]], s=0, per=True)
    u_new = np.linspace(u.min(), u.max(), n_points)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.vstack((x_new, y_new)).T

def heel_align(P):
    """뒤꿈치 점(최대 y)을 원점으로 이동시키고, 배열을 회전시켜 뒤꿈치가 마지막 점이 되도록 합니다."""
    heel_idx = heel_index_max_y(P)
    P_rotated = np.roll(P, -heel_idx - 1, axis=0)
    P_aligned = P_rotated - P_rotated[-1, :]
    return P_aligned

def preprocess_shape(P):
    """전처리 파이프라인: B-Spline 리샘플링 후 뒤꿈치 정렬"""
    P_resampled = closed_bspline_resample(P)
    P_aligned = heel_align(P_resampled)
    return P_aligned

# -----------------------------------------------------------------------------
# 2. 도너 쌍 모델링 함수
# -----------------------------------------------------------------------------

def fit_pair_model(P220_norm, P270_norm):
    """
    정규화된 220mm와 270mm 도너 쌍으로부터 변환 모델을 학습합니다.
    """
    # 1. Umeyama 유사 변환으로 전역 변환(s, R, t) 찾기
    s, R, t = umeyama_similarity(P220_norm.T, P270_norm.T)

    # 2. 전역 변환 적용
    P220_transformed = (s * R @ P220_norm.T + t).T

    # 3. 잔차(Residual) 계산
    residuals = P270_norm - P220_transformed

    # 4. 법선(Normal) 벡터 계산 (근사치)
    normals = np.zeros_like(P220_norm)
    normals[:-1] = P220_norm[1:] - P220_norm[:-1]
    normals[-1] = P220_norm[0] - P220_norm[-1]
    normals = np.array([-normals[:, 1], normals[:, 0]]).T
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # 5. 법선 방향으로의 잔차 스케일(s_res) 계산
    s_res = np.einsum('ij,ij->i', residuals, normals) # 각 점별 내적

    model = {'s': s, 'R': R, 't': t, 's_res': s_res, 'normals_base': normals}
    return model

# -----------------------------------------------------------------------------
# 3. 예측 모델 적용 함수
# -----------------------------------------------------------------------------

def predict_from_model(P220_norm, model, residual_weight=1.0):
    """학습된 단일 모델을 사용하여 270mm 모양을 예측합니다."""
    # 전역 변환 적용
    P_pred_global = (model['s'] * model['R'] @ P220_norm.T + model['t']).T

    # 잔차 적용
    residuals_pred = (model['s_res'] * residual_weight)[:, np.newaxis] * model['normals_base']
    P_pred_final = P_pred_global + residuals_pred
    return P_pred_final

def predict_270_for(P220_target_norm, donor_models, P220_donors_norm,
                    mode='donor_avg', k=5, residual_weight=1.0,
                    pca_model=None, Z_donors=None,
                    residual_weight_auto=False, auto_lambda_grid=None, consensus_mm=1.0):
    """
    주어진 220mm 타겟에 대해 다양한 모드로 270mm 모양을 예측합니다.
    """
    # --- A. scale_only 모드 ---
    if mode == 'scale_only':
        return P220_target_norm * (270 / 220)

    # --- K-최근접 이웃(KNN) 찾기 ---
    # Procrustes 거리를 사용하여 P220 타겟과 가장 유사한 K개의 P220 도너를 찾습니다.
    distances = [procrustes(P220_target_norm, P_donor)[2] for P_donor in P220_donors_norm]
    knn_indices = np.argsort(distances)[:k]
    knn_distances = np.array([distances[i] for i in knn_indices])

    # 거리에 반비례하는 가중치 계산
    weights = 1 / (knn_distances + 1e-9)
    weights /= np.sum(weights)

    # --- B. donor_avg 모드 ---
    if mode == 'donor_avg':
        predictions = []
        final_lambda = residual_weight

        # V3: 적응형 잔차 가중치
        if residual_weight_auto and auto_lambda_grid:
            spreads = []
            for lam in auto_lambda_grid:
                lam_preds = [predict_from_model(P220_target_norm, donor_models[i], residual_weight=lam) for i in knn_indices]
                # 컨센서스 확산 계산 (예측 결과들의 점별 표준편차의 평균)
                consensus_spread = np.mean(np.std(np.array(lam_preds), axis=0))
                spreads.append(consensus_spread)

            best_lambda_idx = np.argmin(spreads)
            min_spread = spreads[best_lambda_idx]

            # 컨센서스 기반 폴백
            if min_spread > consensus_mm:
                return P220_target_norm * (270 / 220) # scale_only로 폴백
            final_lambda = auto_lambda_grid[best_lambda_idx]

        # K개 모델로 예측 후 가중 평균
        for i, idx in enumerate(knn_indices):
            pred = predict_from_model(P220_target_norm, donor_models[idx], residual_weight=final_lambda)
            predictions.append(pred * weights[i])

        return np.sum(predictions, axis=0)

    # --- C. pca_knn 모드 ---
    if mode == 'pca_knn':
        # 전역 변환의 가중 평균
        avg_s = np.sum([donor_models[i]['s'] * weights[j] for j, i in enumerate(knn_indices)])
        avg_R = np.sum([donor_models[i]['R'] * weights[j] for j, i in enumerate(knn_indices)], axis=0)
        avg_t = np.sum([donor_models[i]['t'] * weights[j] for j, i in enumerate(knn_indices)], axis=0)

        P_pred_global = (avg_s * avg_R @ P220_target_norm.T + avg_t).T

        # 잔차의 PCA 점수(Z) 가중 평균
        Z_knn = Z_donors[knn_indices, :]
        z_bar_pred = np.sum(Z_knn * weights[:, np.newaxis], axis=0)

        # 예측된 PCA 점수로부터 잔차 재구성
        s_res_pred = pca_model.inverse_transform(z_bar_pred.reshape(1, -1)).flatten()
        normals_base = np.mean([donor_models[i]['normals_base'] for i in knn_indices], axis=0)
        residuals_pred = (s_res_pred * residual_weight)[:, np.newaxis] * normals_base

        return P_pred_global + residuals_pred

    raise ValueError(f"알 수 없는 모드: {mode}")

# -----------------------------------------------------------------------------
# 4. 후처리 함수
# -----------------------------------------------------------------------------

def heel_resnap(P_pred, P_target_raw):
    """
    예측된 모양의 뒤꿈치를 원본 타겟의 뒤꿈치 위치로 이동시킵니다.
    """
    heel_idx_raw = heel_index_max_y(P_target_raw)
    heel_pos_raw = P_target_raw[heel_idx_raw]
    
    heel_idx_pred = heel_index_max_y(P_pred)
    heel_pos_pred = P_pred[heel_idx_pred]
    
    translation = heel_pos_raw - heel_pos_pred
    return P_pred + translation

def area_resnap(P_pred, P220_target_norm):
    """

    예측된 270mm 모양의 면적을 이론적인 타겟 면적에 맞게 보정합니다.
    """
    area_220_norm = polygon_area(P220_target_norm[:, 0], P220_target_norm[:, 1])
    target_area_270 = area_220_norm * (270/220)**2
    
    current_area_270 = polygon_area(P_pred[:, 0], P_pred[:, 1])
    
    if current_area_270 < 1e-6: return P_pred # 면적이 0에 가까우면 스케일링 방지
    
    scale_factor = np.sqrt(target_area_270 / current_area_270)
    
    centroid = np.mean(P_pred, axis=0)
    return (P_pred - centroid) * scale_factor + centroid

# -----------------------------------------------------------------------------
# 5. 데이터 로딩 및 자동 튜닝 함수
# -----------------------------------------------------------------------------

def load_and_prepare_data(filepath):
    """
    CSV 파일에서 데이터를 로드하고 도너 쌍을 준비합니다.
    
    *** 중요 ***
    이 함수는 220mm와 270mm 데이터가 쌍으로 존재한다고 가정합니다.
    실제 데이터 구조에 맞게 이 부분을 수정해야 합니다.
    """
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Size', 'PairID']) # 메타데이터 없는 행 제거
    
    # 40개의 좌표 (x1, y1, ..., x40, y40)를 (40, 2) 형태의 배열로 변환
    def shape_to_array(row):
        coords = []
        for i in range(1, 41):
            coords.append([row[f'x{i}'], row[f'y{i}']])
        return np.array(coords)

    df['shape_array'] = df.apply(shape_to_array, axis=1)

    # --- TODO: 이 부분을 실제 데이터 구조에 맞게 수정하세요 ---
    # 예시: 'PairID'를 기준으로 220과 270 사이즈를 짝짓습니다.
    # 현재 파일에는 270mm 데이터가 없으므로, 임시로 220mm를 스케일링하여
    # 가상의 270mm 도너 데이터를 생성합니다.
    df_220 = df[df['Size'] == 220].copy()
    
    # 실제 데이터가 있다면 아래와 같이 270mm 데이터를 로드하고 merge 해야 합니다.
    # df_270 = df[df['Size'] == 270].copy()
    # donor_df = pd.merge(df_220, df_270, on='PairID', suffixes=('_220', '_270'))
    
    # 임시 가상 데이터 생성
    print("경고: 270mm 도너 데이터가 없어 220mm 데이터를 스케일링하여 가상 도너 쌍을 생성합니다.")
    df_220['shape_array_270_dummy'] = df_220['shape_array'].apply(lambda p: p * (270/220))
    donor_df = df_220.rename(columns={'shape_array': 'shape_array_220'})
    donor_df = donor_df.rename(columns={'shape_array_270_dummy': 'shape_array_270'})
    # --- 여기까지 수정 ---
    
    P220_donors_raw = list(donor_df['shape_array_220'])
    P270_donors_raw = list(donor_df['shape_array_270'])
    
    return P220_donors_raw, P270_donors_raw

def auto_tune(P220_donors_raw, P270_donors_raw, param_grid):
    """
    Leave-One-Out 교차 검증을 통해 최적의 하이퍼파라미터를 찾습니다.
    """
    best_params = None
    best_rmse = float('inf')

    # 모든 파라미터 조합에 대해 테스트
    param_combinations = list(product(*param_grid.values()))
    
    for params_tuple in param_combinations:
        params = dict(zip(param_grid.keys(), params_tuple))
        
        loo = LeaveOneOut()
        total_rmse = 0
        
        # 교차 검증
        for train_idx, test_idx in loo.split(P220_donors_raw):
            test_idx = test_idx[0]
            
            # --- 학습 데이터 준비 ---
            P220_train_raw = [P220_donors_raw[i] for i in train_idx]
            P270_train_raw = [P270_donors_raw[i] for i in train_idx]
            
            P220_train_norm = [preprocess_shape(p) for p in P220_train_raw]
            P270_train_norm = [preprocess_shape(p) for p in P270_train_raw]
            
            donor_models = [fit_pair_model(p220, p270) for p220, p270 in zip(P220_train_norm, P270_train_norm)]
            
            # PCA 모델 (필요시)
            pca_model, Z_donors = None, None
            if params['mode'] == 'pca_knn':
                s_res_matrix = np.array([m['s_res'] for m in donor_models])
                n_components = int(s_res_matrix.shape[1] * params.get('var_thresh', 0.95))
                pca_model = PCA(n_components=n_components)
                Z_donors = pca_model.fit_transform(s_res_matrix)

            # --- 테스트 데이터 준비 ---
            P220_test_raw = P220_donors_raw[test_idx]
            P270_test_gt = P270_donors_raw[test_idx] # Ground Truth
            P220_test_norm = preprocess_shape(P220_test_raw)
            
            # --- 예측 ---
            P270_pred_norm = predict_270_for(
                P220_test_norm, donor_models, P220_train_norm,
                mode=params['mode'], k=params['k'], residual_weight=params['residual_weight'],
                pca_model=pca_model, Z_donors=Z_donors
            )
            
            # --- 후처리 ---
            P270_pred_final = heel_resnap(
                area_resnap(P270_pred_norm, P220_test_norm),
                P220_test_raw
            )
            
            # --- 평가 ---
            rmse = np.sqrt(np.mean((P270_pred_final - P270_test_gt)**2))
            total_rmse += rmse
            
        avg_rmse = total_rmse / len(P220_donors_raw)
        
        print(f"테스트 파라미터: {params}, 평균 RMSE: {avg_rmse:.4f}")
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params
            
    return best_params, best_rmse

# -----------------------------------------------------------------------------
# 6. 메인 실행 블록
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # --- 1. 데이터 로드 ---
    # 실제 220mm-270mm 쌍 데이터가 필요합니다.
    P220_donors_raw, P270_donors_raw = load_and_prepare_data('insole_viewer_output.csv')
    
    print(f"\n총 {len(P220_donors_raw)}개의 도너 쌍 데이터를 로드했습니다.")
    
    # --- 2. 자동 튜닝 (--auto_tune) ---
    print("\n--- 최적 하이퍼파라미터 자동 튜닝 시작 ---")
    param_grid = {
        'mode': ['donor_avg', 'pca_knn', 'scale_only'],
        'k': [3, 5, 7],
        'residual_weight': [0.5, 1.0, 1.5],
        'var_thresh': [0.95] # for pca_knn
    }
    
    best_params, best_rmse = auto_tune(P220_donors_raw, P270_donors_raw, param_grid)
    
    print(f"\n--- 튜닝 완료 ---")
    print(f"최적 파라미터: {best_params}")
    print(f"최소 평균 RMSE: {best_rmse:.4f}")
    
    # --- 3. 최적 파라미터로 최종 모델 학습 및 예측 ---
    # 여기서는 예시로 첫 번째 220mm 데이터를 새로운 타겟으로 가정하고 예측합니다.
    target_idx = 0
    P220_target_raw = P220_donors_raw[target_idx]
    
    # 타겟을 제외한 나머지로 학습
    P220_train_raw = P220_donors_raw[:target_idx] + P220_donors_raw[target_idx+1:]
    P270_train_raw = P270_donors_raw[:target_idx] + P270_donors_raw[target_idx+1:]
    
    P220_train_norm = [preprocess_shape(p) for p in P220_train_raw]
    P270_train_norm = [preprocess_shape(p) for p in P270_train_raw]
    
    all_donor_models = [fit_pair_model(p220, p270) for p220, p270 in zip(P220_train_norm, P270_train_norm)]
    
    pca, Z = None, None
    if best_params['mode'] == 'pca_knn':
        s_res_mat = np.array([m['s_res'] for m in all_donor_models])
        n_comp = int(s_res_mat.shape[1] * best_params.get('var_thresh', 0.95))
        pca = PCA(n_components=n_comp)
        Z = pca.fit_transform(s_res_mat)
        
    P220_target_norm = preprocess_shape(P220_target_raw)
    
    final_prediction_norm = predict_270_for(
        P220_target_norm, all_donor_models, P220_train_norm, **best_params,
        pca_model=pca, Z_donors=Z
    )
    
    final_prediction = heel_resnap(
        area_resnap(final_prediction_norm, P220_target_norm),
        P220_target_raw
    )
    
    # 예측 결과를 CSV로 저장
    pred_df = pd.DataFrame(final_prediction, columns=['x', 'y'])
    pred_df.to_csv('predicted_270mm_shape.csv', index=False)
    
    print(f"\n최적 파라미터로 첫 번째 220mm 데이터에 대한 270mm 예측을 완료했습니다.")
    print("결과가 'predicted_270mm_shape.csv' 파일로 저장되었습니다.")