import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
import warnings
from itertools import product
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union

warnings.filterwarnings('ignore')

class InsolePredictionSystem:
    def __init__(self):
        self.donor_pairs = {}
        self.donor_models = {}
        self.pca_model = None
        self.scaler = None
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """CSV 파일에서 데이터 로드"""
        df = pd.read_csv(csv_path)
        return df
    
    def extract_coordinates(self, row: pd.Series) -> np.ndarray:
        """행에서 x, y 좌표 추출"""
        coords = []
        for i in range(1, 41):  # x1,y1 ~ x40,y40
            x_col = f'x{i}'
            y_col = f'y{i}'
            if x_col in row and y_col in row:
                coords.append([row[x_col], row[y_col]])
        return np.array(coords)
    
    def heel_index_max_y(self, coords: np.ndarray) -> int:
        """가장 큰 y좌표를 가진 점(뒤꿈치)의 인덱스 찾기"""
        return np.argmax(coords[:, 1])
    
    def rotate_to_heel_last(self, coords: np.ndarray) -> np.ndarray:
        """뒤꿈치 점이 마지막이 되도록 좌표 회전"""
        heel_idx = self.heel_index_max_y(coords)
        if heel_idx == len(coords) - 1:
            return coords.copy()
        return np.vstack([coords[heel_idx+1:], coords[:heel_idx+1]])
    
    def closed_bspline_resample(self, coords: np.ndarray, num_points: int = 40) -> np.ndarray:
        """B-spline을 사용한 폐곡선 리샘플링"""
        coords_closed = np.vstack([coords, coords[0]])  # 곡선을 닫음
        
        # 매개변수화
        distances = np.sqrt(np.sum(np.diff(coords_closed, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        t = cumulative_distances / cumulative_distances[-1]
        
        try:
            # B-spline 피팅
            tck, _ = splprep([coords_closed[:, 0], coords_closed[:, 1]], u=t, s=0, per=1)
            
            # 균등한 간격으로 리샘플링
            u_new = np.linspace(0, 1, num_points, endpoint=False)
            resampled = np.array(splev(u_new, tck)).T
            
            return resampled
        except:
            # B-spline이 실패하면 선형 보간 사용
            return self._linear_resample(coords, num_points)
    
    def _linear_resample(self, coords: np.ndarray, num_points: int) -> np.ndarray:
        """선형 보간을 사용한 리샘플링"""
        coords_closed = np.vstack([coords, coords[0]])
        distances = np.sqrt(np.sum(np.diff(coords_closed, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative_distances[-1]
        
        # 균등한 간격으로 새로운 점들 생성
        new_distances = np.linspace(0, total_length, num_points, endpoint=False)
        resampled = []
        
        for new_dist in new_distances:
            # 현재 거리에 해당하는 점 찾기
            idx = np.searchsorted(cumulative_distances, new_dist)
            if idx == 0:
                resampled.append(coords_closed[0])
            elif idx >= len(cumulative_distances) - 1:
                resampled.append(coords_closed[-2])  # 마지막 점 전 점 사용
            else:
                # 선형 보간
                t = (new_dist - cumulative_distances[idx-1]) / (cumulative_distances[idx] - cumulative_distances[idx-1])
                point = coords_closed[idx-1] + t * (coords_closed[idx] - coords_closed[idx-1])
                resampled.append(point)
        
        return np.array(resampled)
    
    def heel_align(self, coords: np.ndarray) -> np.ndarray:
        """뒤꿈치를 원점 근처로 정렬"""
        heel_idx = self.heel_index_max_y(coords)
        heel_point = coords[heel_idx]
        return coords - heel_point
    
    def preprocess_shape(self, coords: np.ndarray) -> np.ndarray:
        """전체 전처리 파이프라인"""
        coords = self.rotate_to_heel_last(coords)
        coords = self.closed_bspline_resample(coords, 40)
        coords = self.heel_align(coords)
        return coords
    
    def umeyama_similarity(self, P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Umeyama 유사 변환 계산"""
        # 중심점 계산
        mu_P = np.mean(P, axis=0)
        mu_Q = np.mean(Q, axis=0)
        
        # 중심점으로 이동
        P_centered = P - mu_P
        Q_centered = Q - mu_Q
        
        # 스케일 계산
        sigma_P2 = np.sum(P_centered**2) / len(P)
        
        if sigma_P2 < 1e-10:
            # P가 모든 점이 같은 경우
            s = 1.0
            R = np.eye(2)
            t = mu_Q - mu_P
        else:
            # 공분산 행렬
            Sigma = (P_centered.T @ Q_centered) / len(P)
            
            # SVD
            U, D, Vt = np.linalg.svd(Sigma)
            
            # 회전 행렬
            S = np.eye(2)
            if np.linalg.det(U @ Vt) < 0:
                S[-1, -1] = -1
            R = U @ S @ Vt
            
            # 스케일
            s = np.trace(np.diag(D) @ S) / sigma_P2
            
            # 평행이동
            t = mu_Q - s * R @ mu_P
        
        return s, R, t
    
    def compute_normals(self, coords: np.ndarray) -> np.ndarray:
        """각 점에서의 법선 벡터 계산"""
        n = len(coords)
        normals = np.zeros_like(coords)
        
        for i in range(n):
            # 이전 점과 다음 점
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            # 접선 벡터
            tangent = coords[next_idx] - coords[prev_idx]
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 1e-10:
                tangent = tangent / tangent_norm
                # 법선 벡터 (시계 반대방향으로 90도 회전)
                normal = np.array([-tangent[1], tangent[0]])
                normals[i] = normal
            else:
                normals[i] = np.array([0, 1])  # 기본 법선
        
        return normals
    
    def fit_pair_model(self, P220: np.ndarray, P270: np.ndarray) -> Dict:
        """도너 쌍 모델 생성"""
        # Umeyama 변환
        s, R, t = self.umeyama_similarity(P220, P270)
        
        # 변환 적용
        P220_transformed = s * (P220 @ R.T) + t
        
        # 잔차 계산
        residual = P270 - P220_transformed
        
        # 법선 방향으로 잔차 투영
        normals = self.compute_normals(P220_transformed)
        s_res = np.array([np.dot(residual[i], normals[i]) for i in range(len(residual))])
        
        return {
            's': s,
            'R': R,
            't': t,
            's_res': s_res
        }
    
    def procrustes_distance(self, P1: np.ndarray, P2: np.ndarray) -> float:
        """Procrustes 거리 계산"""
        s, R, t = self.umeyama_similarity(P1, P2)
        P1_aligned = s * (P1 @ R.T) + t
        return np.sqrt(np.mean(np.sum((P1_aligned - P2)**2, axis=1)))
    
    def apply_model(self, model: Dict, P220: np.ndarray, residual_weight: float = 1.0) -> np.ndarray:
        """모델을 새로운 220mm 모양에 적용"""
        # 전역 변환 적용
        P_global = model['s'] * (P220 @ model['R'].T) + model['t']
        
        # 법선 방향으로 잔차 적용
        normals = self.compute_normals(P_global)
        residual_vectors = normals * model['s_res'].reshape(-1, 1) * residual_weight
        
        return P_global + residual_vectors
    
    def predict_scale_only(self, P220: np.ndarray, target_size: float = 270, source_size: float = 220) -> np.ndarray:
        """단순 스케일링 예측"""
        scale_factor = target_size / source_size
        return P220 * scale_factor
    
    def predict_donor_avg(self, P220: np.ndarray, k: int = 3, residual_weight: float = 1.0) -> np.ndarray:
        """도너 평균 기반 예측"""
        if not self.donor_models:
            raise ValueError("도너 모델이 없습니다. 먼저 학습을 수행하세요.")
        
        # 거리 계산
        distances = []
        for pair_id, (donor_220, _) in self.donor_pairs.items():
            dist = self.procrustes_distance(P220, donor_220)
            distances.append((dist, pair_id))
        
        # k개 최근접 이웃 선택
        distances.sort()
        selected = distances[:k]
        
        # 가중치 계산 (거리의 역수)
        weights = []
        for dist, _ in selected:
            weight = 1.0 / (dist + 1e-10)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 정규화
        
        # 가중 평균 예측
        predictions = []
        for (_, pair_id), weight in zip(selected, weights):
            model = self.donor_models[pair_id]
            pred = self.apply_model(model, P220, residual_weight)
            predictions.append(pred)
        
        # 가중 평균
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_pred += pred * weight
        
        return weighted_pred
    
    def fit_pca_models(self, var_thresh: float = 0.95):
        """PCA 모델 학습"""
        if not self.donor_models:
            raise ValueError("도너 모델이 없습니다.")
        
        # 모든 잔차 수집
        residuals = []
        for model in self.donor_models.values():
            residuals.append(model['s_res'])
        
        residuals = np.array(residuals)
        
        # 표준화
        self.scaler = StandardScaler()
        residuals_scaled = self.scaler.fit_transform(residuals)
        
        # PCA
        self.pca_model = PCA()
        self.pca_model.fit(residuals_scaled)
        
        # 분산 임계값으로 성분 수 결정
        cumsum_var = np.cumsum(self.pca_model.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= var_thresh) + 1
        
        # 선택된 성분으로 다시 학습
        self.pca_model = PCA(n_components=n_components)
        self.pca_model.fit(residuals_scaled)
    
    def predict_pca_knn(self, P220: np.ndarray, k: int = 3, residual_weight: float = 1.0) -> np.ndarray:
        """PCA KNN 기반 예측"""
        if self.pca_model is None:
            raise ValueError("PCA 모델이 학습되지 않았습니다.")
        
        # k개 최근접 이웃 찾기
        distances = []
        for pair_id, (donor_220, _) in self.donor_pairs.items():
            dist = self.procrustes_distance(P220, donor_220)
            distances.append((dist, pair_id))
        
        distances.sort()
        selected = distances[:k]
        
        # 가중치 계산
        weights = []
        for dist, _ in selected:
            weight = 1.0 / (dist + 1e-10)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # PCA 점수 가중 평균
        pca_scores = []
        global_transforms = []
        
        for (_, pair_id), weight in zip(selected, weights):
            model = self.donor_models[pair_id]
            
            # 잔차를 PCA 공간으로 변환
            s_res_scaled = self.scaler.transform([model['s_res']])
            z = self.pca_model.transform(s_res_scaled)[0]
            pca_scores.append(z)
            
            # 전역 변환도 저장
            global_transforms.append((model['s'], model['R'], model['t']))
        
        # 가중 평균
        z_avg = np.zeros(self.pca_model.n_components_)
        for z, weight in zip(pca_scores, weights):
            z_avg += z * weight
        
        # PCA로부터 잔차 재구성
        s_res_scaled_pred = self.pca_model.inverse_transform([z_avg])
        s_res_pred = self.scaler.inverse_transform(s_res_scaled_pred)[0]
        
        # 전역 변환 가중 평균
        s_avg = sum(s * w for (s, _, _), w in zip(global_transforms, weights))
        R_avg = np.zeros((2, 2))
        t_avg = np.zeros(2)
        
        for (_, R, t), w in zip(global_transforms, weights):
            R_avg += R * w
            t_avg += t * w
        
        # 직교화 (Gram-Schmidt)
        R_avg = self._orthogonalize_matrix(R_avg)
        
        # 예측 적용
        P_global = s_avg * (P220 @ R_avg.T) + t_avg
        normals = self.compute_normals(P_global)
        residual_vectors = normals * s_res_pred.reshape(-1, 1) * residual_weight
        
        return P_global + residual_vectors
    
    def _orthogonalize_matrix(self, M: np.ndarray) -> np.ndarray:
        """행렬을 직교화"""
        U, _, Vt = np.linalg.svd(M)
        return U @ Vt
    
    def compute_consensus_spread(self, P220: np.ndarray, k: int, residual_weight: float) -> float:
        """컨센서스 확산 계산"""
        if k >= len(self.donor_pairs):
            k = len(self.donor_pairs)
        
        # k개 예측 생성
        distances = []
        for pair_id, (donor_220, _) in self.donor_pairs.items():
            dist = self.procrustes_distance(P220, donor_220)
            distances.append((dist, pair_id))
        
        distances.sort()
        selected = distances[:k]
        
        predictions = []
        for _, pair_id in selected:
            model = self.donor_models[pair_id]
            pred = self.apply_model(model, P220, residual_weight)
            predictions.append(pred)
        
        if len(predictions) < 2:
            return 0.0
        
        # 점별 표준편차의 평균
        predictions = np.array(predictions)  # (k, 40, 2)
        point_stds = np.std(predictions, axis=0)  # (40, 2)
        point_distances = np.sqrt(np.sum(point_stds**2, axis=1))  # (40,)
        
        return np.mean(point_distances)
    
    def predict_270_for(self, P220: np.ndarray, mode: str = 'donor_avg', k: int = 3, 
                       residual_weight: float = 1.0, var_thresh: float = 0.95,
                       residual_weight_auto: bool = False, auto_lambda_grid: List[float] = None,
                       consensus_mm: float = 2.0) -> np.ndarray:
        """270mm 예측 메인 함수"""
        
        if mode == 'scale_only':
            return self.predict_scale_only(P220)
        
        elif mode == 'donor_avg':
            if residual_weight_auto and auto_lambda_grid:
                # 자동 잔차 가중치 선택
                best_lambda = residual_weight
                min_spread = float('inf')
                
                for lambda_val in auto_lambda_grid:
                    spread = self.compute_consensus_spread(P220, k, lambda_val)
                    if spread < min_spread:
                        min_spread = spread
                        best_lambda = lambda_val
                
                # 컨센서스가 너무 크면 scale_only로 폴백
                if min_spread > consensus_mm:
                    return self.predict_scale_only(P220)
                
                residual_weight = best_lambda
            
            return self.predict_donor_avg(P220, k, residual_weight)
        
        elif mode == 'pca_knn':
            if self.pca_model is None:
                self.fit_pca_models(var_thresh)
            return self.predict_pca_knn(P220, k, residual_weight)
        
        else:
            raise ValueError(f"알 수 없는 모드: {mode}")
    
    def heel_resnap(self, coords: np.ndarray) -> np.ndarray:
        """뒤꿈치 재정렬"""
        heel_idx = self.heel_index_max_y(coords)
        heel_point = coords[heel_idx]
        return coords - heel_point
    
    def area_resnap(self, P220: np.ndarray, P270_pred: np.ndarray, 
                   target_size: float = 270, source_size: float = 220) -> np.ndarray:
        """면적 기반 재보정"""
        # 이론적 면적 비율
        area_ratio_target = (target_size / source_size) ** 2
        
        # 실제 면적 계산 (Shoelace formula)
        def polygon_area(coords):
            x, y = coords[:, 0], coords[:, 1]
            return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
        
        area_220 = polygon_area(P220)
        area_270_pred = polygon_area(P270_pred)
        
        if area_220 < 1e-10:
            return P270_pred
        
        actual_ratio = area_270_pred / area_220
        correction_factor = np.sqrt(area_ratio_target / actual_ratio)
        
        # 중심점 기준으로 스케일링
        centroid = np.mean(P270_pred, axis=0)
        return (P270_pred - centroid) * correction_factor + centroid
    
    def train(self, df: pd.DataFrame):
        """도너 쌍으로 모델 학습"""
        print("도너 쌍 학습 중...")
        
        # 쌍별로 데이터 그룹화
        pairs = {}
        for _, row in df.iterrows():
            pair_id = row['PairID']
            size = row['Size']
            coords = self.extract_coordinates(row)
            coords = self.preprocess_shape(coords)
            
            if pair_id not in pairs:
                pairs[pair_id] = {}
            pairs[pair_id][size] = coords
        
        # 완전한 쌍만 선택
        complete_pairs = {}
        for pair_id, data in pairs.items():
            if 220 in data and 270 in data:
                complete_pairs[pair_id] = (data[220], data[270])
        
        print(f"완전한 도너 쌍 수: {len(complete_pairs)}")
        
        self.donor_pairs = complete_pairs
        
        # 각 쌍에 대해 모델 학습
        for pair_id, (P220, P270) in complete_pairs.items():
            model = self.fit_pair_model(P220, P270)
            self.donor_models[pair_id] = model
        
        print("학습 완료!")
    
    def evaluate_model(self, mode: str, k: int, residual_weight: float, var_thresh: float) -> float:
        """Leave-One-Out CV로 모델 평가"""
        if len(self.donor_pairs) < 2:
            return float('inf')
        
        rmse_scores = []
        
        for test_pair_id, (test_220, test_270) in self.donor_pairs.items():
            # 테스트 쌍을 제외한 임시 모델 생성
            temp_pairs = {pid: data for pid, data in self.donor_pairs.items() if pid != test_pair_id}
            temp_models = {pid: model for pid, model in self.donor_models.items() if pid != test_pair_id}
            
            # 임시로 데이터 교체
            orig_pairs = self.donor_pairs
            orig_models = self.donor_models
            orig_pca = self.pca_model
            orig_scaler = self.scaler
            
            self.donor_pairs = temp_pairs
            self.donor_models = temp_models
            self.pca_model = None
            self.scaler = None
            
            try:
                # 예측 수행
                if mode == 'pca_knn':
                    if len(temp_pairs) >= 2:  # PCA를 위한 최소 데이터
                        self.fit_pca_models(var_thresh)
                        pred_270 = self.predict_270_for(test_220, mode, min(k, len(temp_pairs)), residual_weight, var_thresh)
                    else:
                        pred_270 = self.predict_scale_only(test_220)
                else:
                    pred_270 = self.predict_270_for(test_220, mode, min(k, len(temp_pairs)), residual_weight)
                
                # RMSE 계산
                rmse = np.sqrt(np.mean(np.sum((pred_270 - test_270)**2, axis=1)))
                rmse_scores.append(rmse)
                
            except Exception as e:
                print(f"평가 중 오류 발생 (쌍 {test_pair_id}): {e}")
                rmse_scores.append(float('inf'))
            finally:
                # 원래 데이터 복원
                self.donor_pairs = orig_pairs
                self.donor_models = orig_models
                self.pca_model = orig_pca
                self.scaler = orig_scaler
        
        return np.mean(rmse_scores) if rmse_scores else float('inf')
    
    def auto_tune(self) -> Dict:
        """자동 하이퍼파라미터 튜닝"""
        print("자동 튜닝 시작...")
        
        # 탐색 공간 정의
        modes = ['scale_only', 'donor_avg', 'pca_knn']
        k_values = [1, 2, 3, 5] if len(self.donor_pairs) >= 5 else [1, 2, 3]
        residual_weights = [0.0, 0.5, 1.0, 1.5, 2.0]
        var_thresholds = [0.90, 0.95, 0.99]
        
        best_score = float('inf')
        best_params = {}
        
        total_combinations = len(modes) * len(k_values) * len(residual_weights) * len(var_thresholds)
        current_combination = 0
        
        for mode in modes:
            for k in k_values:
                for residual_weight in residual_weights:
                    for var_thresh in var_thresholds:
                        current_combination += 1
                        
                        # pca_knn이 아닌 경우 var_thresh는 무관함
                        if mode != 'pca_knn' and var_thresh != 0.95:
                            continue
                        
                        # scale_only인 경우 k와 residual_weight는 무관함
                        if mode == 'scale_only' and (k != 3 or residual_weight != 1.0):
                            continue
                        
                        print(f"테스트 중 ({current_combination}/{total_combinations}): "
                              f"mode={mode}, k={k}, residual_weight={residual_weight}, var_thresh={var_thresh}")
                        
                        try:
                            score = self.evaluate_model(mode, k, residual_weight, var_thresh)
                            
                            if score < best_score:
                                best_score = score
                                best_params = {
                                    'mode': mode,
                                    'k': k,
                                    'residual_weight': residual_weight,
                                    'var_thresh': var_thresh,
                                    'cv_rmse': score
                                }
                                print(f"새로운 최고 성능: RMSE = {score:.4f}")
                        
                        except Exception as e:
                            print(f"오류 발생: {e}")
                            continue
        
        print(f"자동 튜닝 완료. 최고 성능: RMSE = {best_score:.4f}")
        print(f"최적 파라미터: {best_params}")
        
        return best_params
    
    def predict_and_save(self, df: pd.DataFrame, output_path: str, **kwargs):
        """예측 수행 및 결과 저장"""
        print("예측 시작...")
        
        # 220mm 데이터만 필터링
        df_220 = df[df['Size'] == 220].copy()
        
        predictions = []
        
        for idx, row in df_220.iterrows():
            print(f"예측 중: {row['PairID']} ({idx+1}/{len(df_220)})")
            
            # 좌표 추출 및 전처리
            coords_220 = self.extract_coordinates(row)
            coords_220 = self.preprocess_shape(coords_220)
            
            # 예측
            try:
                pred_270 = self.predict_270_for(coords_220, **kwargs)
                
                # 후처리
                pred_270 = self.heel_resnap(pred_270)
                pred_270 = self.area_resnap(coords_220, pred_270)
                
                # 결과 행 생성
                pred_row = row.copy()
                pred_row['Size'] = 270
                
                # 좌표 업데이트
                for i in range(40):
                    pred_row[f'x{i+1}'] = pred_270[i, 0]
                    pred_row[f'y{i+1}'] = pred_270[i, 1]
                
                predictions.append(pred_row)
                
            except Exception as e:
                print(f"예측 실패 {row['PairID']}: {e}")
                continue
        
        # 결과 병합 및 저장
        df_result = pd.concat([df, pd.DataFrame(predictions)], ignore_index=True)
        df_result = df_result.sort_values(['PairID', 'Size'])
        df_result.to_csv(output_path, index=False)
        
        print(f"예측 완료! 결과가 {output_path}에 저장되었습니다.")
        print(f"총 {len(predictions)}개의 270mm 인솔이 예측되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='인솔 크기 예측 시스템')
    parser.add_argument('input_csv', help='입력 CSV 파일 경로')
    parser.add_argument('output_csv', help='출력 CSV 파일 경로')
    parser.add_argument('--mode', choices=['scale_only', 'donor_avg', 'pca_knn'], 
                       default='donor_avg', help='예측 모드')
    parser.add_argument('--k', type=int, default=3, help='K-최근접 이웃 수')
    parser.add_argument('--residual_weight', type=float, default=1.0, help='잔차 가중치')
    parser.add_argument('--var_thresh', type=float, default=0.95, help='PCA 분산 임계값')
    parser.add_argument('--auto_tune', action='store_true', help='자동 하이퍼파라미터 튜닝')
    parser.add_argument('--residual_weight_auto', action='store_true', 
                       help='자동 잔차 가중치 조정 (donor_avg 모드에서만)')
    parser.add_argument('--auto_lambda_grid', nargs='+', type=float,
                       default=[0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
                       help='자동 람다 선택을 위한 그리드')
    parser.add_argument('--consensus_mm', type=float, default=2.0, 
                       help='컨센서스 임계값 (mm)')
    parser.add_argument('--visualize', action='store_true', help='결과 시각화')
    
    args = parser.parse_args()
    
    # 시스템 초기화
    system = InsolePredictionSystem()
    
    # 데이터 로드
    print(f"데이터 로드 중: {args.input_csv}")
    df = system.load_data(args.input_csv)
    print(f"총 {len(df)}개의 레코드 로드됨")
    
    # 모델 학습
    system.train(df)
    
    # 자동 튜닝
    if args.auto_tune:
        best_params = system.auto_tune()
        
        # 최적 파라미터로 업데이트
        args.mode = best_params['mode']
        args.k = best_params['k']
        args.residual_weight = best_params['residual_weight']
        args.var_thresh = best_params['var_thresh']
        
        print(f"자동 튜닝 결과 사용: {best_params}")
    
    # 예측 파라미터 설정
    predict_kwargs = {
        'mode': args.mode,
        'k': args.k,
        'residual_weight': args.residual_weight,
        'var_thresh': args.var_thresh,
        'residual_weight_auto': args.residual_weight_auto,
        'auto_lambda_grid': args.auto_lambda_grid,
        'consensus_mm': args.consensus_mm
    }
    
    # 예측 및 저장
    system.predict_and_save(df, args.output_csv, **predict_kwargs)
    
    # 시각화
    if args.visualize:
        visualize_results(args.output_csv)

def visualize_results(csv_path: str, sample_pairs: int = 3):
    """결과 시각화"""
    df = pd.read_csv(csv_path)
    
    # 완전한 쌍(220mm, 270mm 모두 있는 것) 찾기
    pair_counts = df.groupby('PairID')['Size'].nunique()
    complete_pairs = pair_counts[pair_counts == 2].index[:sample_pairs]
    
    if len(complete_pairs) == 0:
        print("시각화할 완전한 쌍이 없습니다.")
        return
    
    fig, axes = plt.subplots(1, len(complete_pairs), figsize=(5*len(complete_pairs), 5))
    if len(complete_pairs) == 1:
        axes = [axes]
    
    system = InsolePredictionSystem()
    
    for idx, pair_id in enumerate(complete_pairs):
        ax = axes[idx]
        
        # 해당 쌍의 데이터 가져오기
        pair_data = df[df['PairID'] == pair_id]
        
        for _, row in pair_data.iterrows():
            coords = system.extract_coordinates(row)
            coords = system.preprocess_shape(coords)
            
            # 닫힌 곡선으로 그리기
            coords_closed = np.vstack([coords, coords[0]])
            
            if row['Size'] == 220:
                ax.plot(coords_closed[:, 0], coords_closed[:, 1], 'b-', linewidth=2, 
                       label='220mm', alpha=0.7)
            else:
                ax.plot(coords_closed[:, 0], coords_closed[:, 1], 'r-', linewidth=2, 
                       label='270mm', alpha=0.7)
        
        ax.set_title(f'Pair {pair_id}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()

def analyze_dataset(csv_path: str):
    """데이터셋 분석"""
    df = pd.read_csv(csv_path)
    
    print("=== 데이터셋 분석 ===")
    print(f"총 레코드 수: {len(df)}")
    print(f"유니크 PairID 수: {df['PairID'].nunique()}")
    print(f"크기별 분포:")
    print(df['Size'].value_counts().sort_index())
    
    print(f"\n형태별 분포:")
    print(df['Shape'].value_counts())
    
    # 완전한 쌍 분석
    pair_counts = df.groupby('PairID')['Size'].nunique()
    complete_pairs = (pair_counts == 2).sum()
    incomplete_pairs = (pair_counts == 1).sum()
    
    print(f"\n쌍 분석:")
    print(f"완전한 쌍 (220mm + 270mm): {complete_pairs}")
    print(f"불완전한 쌍 (220mm만): {incomplete_pairs}")
    
    # 220mm만 있는 쌍들
    only_220_pairs = pair_counts[pair_counts == 1].index
    df_only_220 = df[df['PairID'].isin(only_220_pairs)]
    print(f"예측 대상 (220mm만 있는 쌍): {len(df_only_220)}")

if __name__ == "__main__":
    import sys
    
    # 간단한 분석 모드
    if len(sys.argv) == 2 and sys.argv[1].endswith('.csv'):
        analyze_dataset(sys.argv[1])
    else:
        main()

# 사용 예시 및 테스트 함수
def example_usage():
    """사용 예시"""
    print("=== 인솔 예측 시스템 사용 예시 ===")
    print()
    
    print("1. 기본 사용법:")
    print("python insole_prediction.py input.csv output.csv")
    print()
    
    print("2. 자동 튜닝 사용:")
    print("python insole_prediction.py input.csv output.csv --auto_tune")
    print()
    
    print("3. 고급 옵션:")
    print("python insole_prediction.py input.csv output.csv --mode pca_knn --k 5 --residual_weight 1.2")
    print()
    
    print("4. 자동 잔차 가중치 조정:")
    print("python insole_prediction.py input.csv output.csv --residual_weight_auto --consensus_mm 1.5")
    print()
    
    print("5. 데이터셋 분석만:")
    print("python insole_prediction.py input.csv")
    print()

def test_with_sample_data():
    """샘플 데이터로 테스트"""
    # 간단한 원형 데이터 생성
    def create_circle(center, radius, num_points=40):
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        points = np.column_stack([
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles)
        ])
        return points
    
    # 테스트 시스템
    system = InsolePredictionSystem()
    
    # 가짜 도너 쌍 생성
    system.donor_pairs = {
        'pair1': (create_circle([0, 0], 50), create_circle([0, 0], 61.36)),  # 270/220 비율
        'pair2': (create_circle([10, 5], 48), create_circle([12, 6], 59.5))
    }
    
    # 모델 학습
    for pair_id, (p220, p270) in system.donor_pairs.items():
        model = system.fit_pair_model(p220, p270)
        system.donor_models[pair_id] = model
    
    # 테스트 예측
    test_220 = create_circle([5, 2], 49)
    pred_270 = system.predict_270_for(test_220, mode='donor_avg', k=2)
    
    print("테스트 완료!")
    print(f"입력 220mm 크기: {np.mean(np.linalg.norm(test_220, axis=1)):.2f}")
    print(f"예측 270mm 크기: {np.mean(np.linalg.norm(pred_270, axis=1)):.2f}")
    print(f"기대 비율: {270/220:.3f}, 실제 비율: {np.mean(np.linalg.norm(pred_270, axis=1))/np.mean(np.linalg.norm(test_220, axis=1)):.3f}")

if __name__ == "__test__":
    test_with_sample_data()