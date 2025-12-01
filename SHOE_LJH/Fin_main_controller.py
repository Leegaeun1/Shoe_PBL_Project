import os
import numpy as np
from Fin_shape_prediction_lib import ShapePredictorEnv  # 2번 코드 임포트



# 예측할 사이즈 범위 (예: 230 ~ 280, 5mm 단위)
TARGET_SIZES = np.arange(230, 285, 5, dtype=int)

# =========================================================
# [설정 2] 모델 선택 (원하는 모델만 True로 설정)
# =========================================================
# available_models:
# 1. Global Deformation (PCA 기반): "PCA_LINEAR", "PCA_GPR", "PCA_SVR", "PCA_KRR"
# 2. Local Deformation (점 단위):   "PURE_GPR", "PURE_SVR", "PURE_KRR"

EXECUTE_MODELS = [
    "PCA_LINEAR",

    "PCA_GPR",     
    "PCA_SVR", 
    "PCA_KRR",

    "PURE_GPR", 
    "PURE_SVR", 
    "PURE_KRR"
]

# =========================================================
# [설정 3] 저장 옵션
# =========================================================
SAVE_PRED_CSV = True       # 예측 결과 CSV 저장 여부
SAVE_MODEL_PKL = False     # 학습된 모델 객체(pickle) 저장 여부 (필요시 True)

def main():

    for Num in range(10,80,10):
        # =========================================================
        # [설정 1] 경로 및 데이터 설정
        # =========================================================
        BASE_DIR = f"20251125/CTRL{Num}"
        MASTER_CSV_PATH = os.path.join(BASE_DIR, "control_points_master_L_20251124.csv")


        # 1. 환경 초기화 (데이터 로드)
        print(f"[Main] 데이터 로드 중... {MASTER_CSV_PATH}")
        if not os.path.exists(MASTER_CSV_PATH):
            print(f"[Error] 파일을 찾을 수 없습니다: {MASTER_CSV_PATH}")
            return

        env = ShapePredictorEnv(MASTER_CSV_PATH)
        
        # 2. 선택된 모델별로 순차 실행
        for model_name in EXECUTE_MODELS:
            print(f"\n==========================================")
            print(f" >>> 모델 실행: {model_name}")
            print(f"==========================================")
            
            # 저장 경로 자동 생성 (모델 이름별로 폴더 구분)
            output_dir = os.path.join(BASE_DIR, "Predictions", model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            output_csv_name = f"pred_Data_{model_name}_230_280.csv"
            output_path = os.path.join(output_dir, output_csv_name)
            
            # 3. 학습 및 예측 실행 (2번 코드의 핵심 함수 호출)
            # -> 내부적으로 각 Type별로 Best Match를 찾아서 학습함
            env.run_prediction_all_types(
                model_type=model_name,
                target_sizes=TARGET_SIZES,
                save_path=output_path if SAVE_PRED_CSV else None,
                save_model=SAVE_MODEL_PKL
            )

        print(f"\n{Num} 모든 작업이 완료되었습니다.")

    print("\n[Main] 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()