import os
import importlib
import sys

IMAGE_DIR_PATH = r"C:\Users\user\Documents\GitHub\Shoe_PBL_Project\SHOE_LJH\output_outlines_1030\L" 
BASE_DATE = "20260106"  # 결과가 저장될 폴더 날짜

# 실행 범위 설정 (20개 ~ 50개)
START_N = 20
END_N = 50 
STEP = 1

def run_automation():
    print(f">>> [Auto] Start Loop: Control Points {START_N} ~ {END_N}")

    try:
        mod_counter = importlib.import_module("2_CounterToExcel_Vis_2026")
        mod_ml_ctrl = importlib.import_module("4_main_controller_2026")
        mod_ratio   = importlib.import_module("4_ratio_point_2026")
        mod_visual  = importlib.import_module("5_Result_Visual_V8_All_2026")
        mod_agg     = importlib.import_module("5_aggregate_results_2026")
    except ImportError as e:
        print(f"[Error] 파일을 찾을 수 없습니다. 파일명이 정확한지 확인해주세요: {e}")
        return

    # 1. 루프 실행 (개별 CTRL 폴더 생성 및 처리)
    for n_ctrl in range(START_N, END_N + 1, STEP):
        print(f"\n" + "="*60)
        print(f" >>> Processing Control Points: {n_ctrl}")
        print(f"="*60)
        
        # [Step 1] 제어점 CSV 생성 (2번 파일)
        print(f"\n[1/4] Generating Control Points CSV...")
        try:
            mod_counter.main_process(IMAGE_DIR_PATH, n_ctrl, BASE_DATE)
        except Exception as e:
            print(f"[Error] Step 1 Failed: {e}")
            continue # CSV 생성 실패 시 해당 루프 건너뜀

        # [Step 2] ML 모델 예측 (4번 메인 컨트롤러)
        print(f"\n[2/4] Running ML Predictions...")
        try:
            mod_ml_ctrl.main_process(n_ctrl, BASE_DATE)
        except Exception as e:
            print(f"[Error] Step 2 Failed: {e}")

        # [Step 3] Ratio 로직 실행 (4번 비율 파일)
        print(f"\n[3/4] Running Ratio Logic...")
        try:
            mod_ratio.main_process(n_ctrl, BASE_DATE)
        except Exception as e:
            print(f"[Error] Step 3 Failed: {e}")

        # [Step 4] 시각화 및 개별 평가 (5번 시각화 파일)
        print(f"\n[4/4] Visualizing & Evaluating...")
        try:
            mod_visual.main_process(n_ctrl, BASE_DATE)
        except Exception as e:
            print(f"[Error] Step 4 Failed: {e}")
            
    # 2. 루프 종료 후 최종 결과 집계
    print(f"\n" + "#"*60)
    print(f" ### All Cycles Finished. Aggregating Results...")
    print(f"#"*60)
    
    try:
        # 집계 대상 폴더 리스트 생성
        target_dirs = [f"CTRL{i}" for i in range(START_N, END_N + 1, STEP)]
        
        # 집계 모듈 변수 설정 후 실행
        mod_agg.BASE_ROOT = BASE_DATE
        mod_agg.TARGET_CTRL_DIRS = target_dirs
        mod_agg.OUTPUT_FILENAME = f"Final_Performance_Summary_CTRL{START_N}_{END_N}.csv"
        
        mod_agg.main()
        
    except Exception as e:
        print(f"[Error] Aggregation Failed: {e}")

if __name__ == "__main__":
    run_automation()