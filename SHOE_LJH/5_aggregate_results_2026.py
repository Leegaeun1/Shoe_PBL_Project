import os
import pandas as pd

# =========================================================
# [설정] 경로 및 대상
# =========================================================
BASE_ROOT = "20260106"
TARGET_CTRL_DIRS = [f"CTRL{i}" for i in range(10, 80, 10)] 
TARGET_FILENAME = "model_performance_average_V2.csv" 

# 기본값 
OUTPUT_FILENAME = "Final_Performance_Summary_All_CTRLs_V2.csv" 

def main():
    print(">>> [Start] Aggregating Performance Data...")
    
    all_data = []

    # 1. 각 CTRL 폴더 순회
    for ctrl_dir in TARGET_CTRL_DIRS:
        file_path = os.path.join(BASE_ROOT, ctrl_dir, TARGET_FILENAME)
        if not os.path.exists(file_path):
            print(f"  [Skip] Not found: {file_path}")
            continue
            
        print(f"  > Reading: {ctrl_dir}")
        
        try:
            df = pd.read_csv(file_path)
            
            # 'CTRL_Count' 컬럼 추가 (숫자만 추출)
            ctrl_num = int(ctrl_dir.replace("CTRL", ""))
            df.insert(0, "CTRL_Count", ctrl_num)
            
            all_data.append(df)
            
        except Exception as e:
            print(f"    [Error] Failed to read {ctrl_dir}: {e}")

    if not all_data:
        print("\n[Warning] No data found to aggregate.")
        return

    # 2. 데이터 병합
    final_df = pd.concat(all_data, ignore_index=True)

    # 3. 정렬 (모델명 -> 컨트롤 포인트 순)
    if "Model" in final_df.columns:
        final_df = final_df.sort_values(by=["Model", "CTRL_Count"])

    # 4. 저장 (확장자에 따라 분기 처리)
    output_path = os.path.join(BASE_ROOT, OUTPUT_FILENAME)
    
    try:
        # 파일명이 .csv로 끝나면 CSV로 저장
        if OUTPUT_FILENAME.lower().endswith(".csv"):
            final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"\n[Success] Aggregated CSV saved at:\n  -> {output_path}")
            
        # 그 외(.xlsx 등)는 엑셀로 저장
        else:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False, sheet_name="All_Summary")
            print(f"\n[Success] Aggregated Excel saved at:\n  -> {output_path}")

        print("\n[Preview]")
        print(final_df.head())
        
    except Exception as e:
        print(f"\n[Error] Save failed: {e}")

if __name__ == "__main__":
    main()