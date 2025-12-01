import os
import pandas as pd

## 하나로 합치는 부분 


# =========================================================
# [설정] 경로 및 대상
# =========================================================
BASE_ROOT = "20251126"
TARGET_CTRL_DIRS = [f"CTRL{i}" for i in range(10, 80, 10)] # CTRL10 ~ CTRL70
TARGET_FILENAME = "model_performance_average_V2.csv" # 각 폴더에서 가져올 파일명

# 최종 저장할 파일명
OUTPUT_FILENAME = "Final_Performance_Summary_All_CTRLs.xlsx" # 엑셀로 저장

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
        
        # CSV 읽기
        try:
            df = pd.read_csv(file_path)
            
            # 'CTRL_Count' 컬럼 추가 (구분자 역할)
            # 예: CTRL10 -> 10 (숫자만 추출)
            ctrl_num = int(ctrl_dir.replace("CTRL", ""))
            df.insert(0, "CTRL_Count", ctrl_num)
            
            all_data.append(df)
            
        except Exception as e:
            print(f"    [Error] Failed to read {ctrl_dir}: {e}")

    if not all_data:
        print("\n[Warning] No data found to aggregate.")
        return

    # 2. 데이터 병합 (세로로 합치기)
    final_df = pd.concat(all_data, ignore_index=True)

    # 3. 보기 좋게 정렬 (모델명 -> 컨트롤 포인트 순)
    # 예: PCA_LINEAR (10 -> 20 -> ...), PURE_GPR (10 -> 20 -> ...)
    if "Model" in final_df.columns:
        final_df = final_df.sort_values(by=["Model", "CTRL_Count"])

    # 4. 엑셀 저장
    output_path = os.path.join(BASE_ROOT, OUTPUT_FILENAME)
    
    try:
        # 엑셀Writer를 쓰면 시트 이름도 지정 가능
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False, sheet_name="All_Summary")
            
        print(f"\n[Success] Aggregated file saved at:\n  -> {output_path}")
        print("\n[Preview]")
        print(final_df.head())
        
    except Exception as e:
        print(f"\n[Error] Save failed: {e}")
        # 엑셀 저장이 안 되면 CSV로라도 백업 저장
        csv_backup = output_path.replace(".xlsx", ".csv")
        final_df.to_csv(csv_backup, index=False)
        print(f"  -> Saved as CSV instead: {csv_backup}")

if __name__ == "__main__":
    main()