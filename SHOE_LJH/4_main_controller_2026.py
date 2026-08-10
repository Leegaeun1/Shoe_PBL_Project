import os
import numpy as np

# 라이브러리 import
try:
    from Fin_shape_prediction_lib_2026_V2 import ShapePredictorEnv
except ImportError:
    from Fin_shape_prediction_lib_2026 import ShapePredictorEnv

# 예측할 사이즈 범위
TARGET_SIZES = np.arange(230, 285, 5, dtype=int)

# 클러스터 개수 지정 (원하는 그룹 수)
TARGET_CLUSTER_COUNT = 4 

EXECUTE_MODELS = [
    "PCA_LINEAR",
    "PCA_SVR", 
    "PCA_KRR",
    "PURE_GPR", 
    "PURE_SVR", 
    "PURE_KRR"
]

SAVE_PRED_CSV = True      
SAVE_MODEL_PKL = False    

def main_process(target_ctrl_num, base_date="20260106"):
    # 경로 설정
    BASE_DIR = f"{base_date}/CTRL{target_ctrl_num}"
    MASTER_CSV_PATH = os.path.join(BASE_DIR, f"control_points_master_L_{base_date}.csv")

    print(f"[ML] Start Prediction for CTRL{target_ctrl_num}")
    print(f"[ML] Data Path: {MASTER_CSV_PATH}")

    if not os.path.exists(MASTER_CSV_PATH):
        print(f"[Error] CSV 파일을 찾을 수 없습니다: {MASTER_CSV_PATH}")
        return

    # 1. 환경 초기화
    env = ShapePredictorEnv(MASTER_CSV_PATH)
    
    # 1.5 클러스터 분석 (강제 4그룹 할당)
    cluster_save_dir = os.path.join(BASE_DIR, "Cluster_Analysis")
    
    # n_clusters 인자 사용
    env.analyze_type_clusters(save_dir=cluster_save_dir, n_clusters=TARGET_CLUSTER_COUNT)
    
    # 2. 모델별 실행
    for model_name in EXECUTE_MODELS:
        print(f"   >>> Model: {model_name}")
        
        output_dir = os.path.join(BASE_DIR, "Predictions", model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        output_csv_name = f"pred_Data_{model_name}_230_280.csv"
        output_path = os.path.join(output_dir, output_csv_name)
        
        env.run_prediction_all_types(
            model_type=model_name,
            target_sizes=TARGET_SIZES,
            save_path=output_path if SAVE_PRED_CSV else None,
            save_model=SAVE_MODEL_PKL
        )

    print(f"[ML] Done for CTRL{target_ctrl_num}")