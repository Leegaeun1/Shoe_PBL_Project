import pandas as pd
import numpy as np
import os
import time
import csv

# 타겟 사이즈
target_sizes = list(range(230, 281, 5)) 

def scale_master_csv_by_type_min(input_path, output_path, target_sizes):
    print(f"Loading Master Data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return

    df.columns = df.columns.str.strip()
    
    # 좌표 컬럼 찾기 (x1, y1, ...)
    coord_cols = [col for col in df.columns if col.startswith(('x', 'y')) and col[1:].isdigit()]
    n_points = len(coord_cols) // 2

    # 최소 사이즈 행 찾기
    base_sizes = df.groupby('type')['size'].min().reset_index()
    base_sizes.rename(columns={'size': 'base_size'}, inplace=True)
    df_merged = df.merge(base_sizes, on='type', how='left')
    base_df = df_merged[df_merged['size'] == df_merged['base_size']].copy()

    new_rows = []
    runtime_stats = [] 

    print(">>> Starting Ratio Scaling...")

    for index, base_row in base_df.iterrows():
        base_type = base_row['type']
        base_side = base_row['side']
        base_size = base_row['size']
        base_coords = base_row[coord_cols].values
        
        start_t = time.perf_counter()

        for target_size in target_sizes:
            ratio = target_size / base_size
            scaled_coords = base_coords * ratio

            new_row = {
                'type': base_type,
                'side': base_side,
                'size': target_size
            }
            for i, col in enumerate(coord_cols):
                new_row[col] = scaled_coords[i]
            new_rows.append(new_row)
        
        end_t = time.perf_counter()
        elapsed = end_t - start_t
        
        runtime_stats.append({
            "Type": base_type,
            "Base_Size": base_size,
            "Matched_Type": "Self (Ratio)", 
            "Time_sec": round(elapsed, 6),
            "Points": n_points
        })

    scaled_df = pd.DataFrame(new_rows)
    scaled_df = scaled_df[['type', 'side', 'size'] + coord_cols]
    scaled_df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"[SUCCESS] Saved Ratio CSV -> {output_path}")

    # 런타임 요약 저장
    summary_path = output_path.replace(".csv", "_summary.csv")
    if runtime_stats:
        fieldnames = ["Type", "Base_Size", "Matched_Type", "Time_sec", "Points"]
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(runtime_stats)

# 외부 호출용 함수
def main_process(target_ctrl_num, base_date="20260106"):
    # 동적 경로 설정
    DIR_NAME = f"{base_date}/CTRL{target_ctrl_num}"
    file_name = f"{DIR_NAME}/control_points_master_L_{base_date}.csv"
    output_file = f"{DIR_NAME}/Predictions/RATIO_CTRL/pred_Data_RATIO_CTRL_230_280.csv"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if os.path.exists(file_name):
        scale_master_csv_by_type_min(file_name, output_file, target_sizes)
    else:
        print(f"[Ratio] Master file missing: {file_name}")

# if __name__ == "__main__":
#     main_process(30)