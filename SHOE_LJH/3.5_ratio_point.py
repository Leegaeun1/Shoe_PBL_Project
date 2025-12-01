import pandas as pd
import numpy as np
import os
import time
import csv

# -----------------------
# 전역 설정
# -----------------------
DIR_NAME = "20251126/CTRL70"

file_name = f"{DIR_NAME}/control_points_master_R_20251126.csv"
output_file = f"{DIR_NAME}/Predictions/RATIO_CTRL/pred_Data_RATIO_CTRL_230_280.csv"

# 실행 시간 요약 저장 경로
summary_output_file = f"{DIR_NAME}/ratio_ctrl_runtime_summary.csv"

target_sizes = list(range(230, 281, 5)) # 230, 235, ..., 280

def scale_master_csv_by_type_min(input_path, output_path, target_sizes):
    """
    각 type별 최소 사이즈의 행을 찾아, 해당 행의 control point를
    target_sizes에 따라 비율대로 스케일링하여 저장합니다.
    (실행 시간 측정 기능 추가)
    """
    print(f"Loading Master Data from {input_path}...")
    
    # 1. Load the data and clean column names
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    
    # Define coordinate columns, 마지막 60 이부분은 컨트롤수에 맞게 고쳐야함!
    coord_cols = [col for col in df.columns if col.startswith(('x', 'y')) and col[1:].isdigit() and int(col[1:]) <= 100]
    n_points = len(coord_cols) // 2

    # 2. Determine Base Sizes (minimum size for each type)
    base_sizes = df.groupby('type')['size'].min().reset_index()
    base_sizes.rename(columns={'size': 'base_size'}, inplace=True)

    # 3. Identify Base Rows
    # Merge to get the base size for each type
    df_merged = df.merge(base_sizes, on='type', how='left')

    # Filter to keep only the rows corresponding to the minimum size for each type
    base_df = df_merged[df_merged['size'] == df_merged['base_size']].copy()

    print("Base Rows (Min Size for each Type):")
    print(base_df[['type', 'side', 'size']].head(10))
    print(f"\nTarget Sizes: {target_sizes}")

    # 4. Scaling and Assembly
    new_rows = []
    runtime_stats = [] # 실행 시간 저장용 리스트

    print("\n>>> Starting Ratio Scaling...")

    # 각 Type(Base Row)별로 순회하며 시간 측정
    for index, base_row in base_df.iterrows():
        base_type = base_row['type']
        base_side = base_row['side']
        base_size = base_row['size']
        base_coords = base_row[coord_cols].values
        
        # ★ 타이머 시작
        start_t = time.perf_counter()

        # 타겟 사이즈별 스케일링 수행
        for target_size in target_sizes:
            # Calculate the scaling ratio
            ratio = target_size / base_size

            # Scale the coordinates
            scaled_coords = base_coords * ratio

            # Create a new row (dictionary)
            new_row = {
                'type': base_type,
                'side': base_side,
                'size': target_size
            }

            # Add scaled coordinates
            for i, col in enumerate(coord_cols):
                new_row[col] = scaled_coords[i]

            new_rows.append(new_row)
        
        # ★ 타이머 종료
        end_t = time.perf_counter()
        elapsed = end_t - start_t
        
        print(f"  [DONE] Type: '{base_type}' (Base: {base_size}), Time: {elapsed:.5f}s")

        # ★ 통계 저장
        runtime_stats.append({
            "Type": base_type,
            "Base_Size": base_size,
            "Matched_Type": "Self (Ratio)", # Ratio 방식은 자기 자신을 참조함
            "Time_sec": round(elapsed, 6),
            "Points": n_points
        })

    # 5. Export: Concatenate all new scaled rows
    scaled_df = pd.DataFrame(new_rows)
    
    # Reorder columns to match the original
    scaled_df = scaled_df[['type', 'side', 'size'] + coord_cols]

    # 6. Save Prediction CSV
    scaled_df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"\n[SUCCESS] Scaled CSV saved to -> {output_path}")
    print(f"Total rows saved: {len(scaled_df)}")

    # 7. Save Runtime Summary CSV (★ 추가된 기능)
    if runtime_stats:
        os.makedirs(os.path.dirname(summary_output_file), exist_ok=True)
        fieldnames = ["Type", "Base_Size", "Matched_Type", "Time_sec", "Points"]
        
        with open(summary_output_file, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(runtime_stats)
            
        print(f"[SAVED] Runtime Summary  -> {summary_output_file}")


if __name__ == "__main__":
    scale_master_csv_by_type_min(file_name, output_file, target_sizes)