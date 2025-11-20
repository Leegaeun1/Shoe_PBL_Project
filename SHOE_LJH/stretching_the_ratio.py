import csv
import os

# -----------------------
# 전역 설정 (필요에 맞게 수정)
# -----------------------
CORRECT_SHOE_SIZE = 250
TARGET_SHOE_SIZE  = 270
ratio = TARGET_SHOE_SIZE / CORRECT_SHOE_SIZE  # 1.08 등

# 이미 만들어져 있는 마스터 CSV 경로
INPUT_CSV  = r"C:\Users\user\Documents\GitHub\Shoe_PBL_Project\SHOE_LJH\Fin_Excel_Data1\control_points_master_L_20251117_2.csv"
# 출력 CSV 경로 (원본 이름 + "_scaled_270mm" 같은 식)
base, ext  = os.path.splitext(INPUT_CSV)
OUTPUT_CSV = base + f"_scaled_{TARGET_SHOE_SIZE}mm" + ext


def scale_master_csv(input_path, output_path, ratio):
    """
    CORRECT_SHOE_SIZE와 일치하는 행만 찾아 control point를 ratio 배로 스케일링하고, 
    size 컬럼을 TARGET_SHOE_SIZE로 변경합니다. 스케일링된 행만 OUTPUT_CSV에 저장됩니다.
    """
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("[ERROR] 입력 CSV가 비어 있습니다.")
        return

    # --- 헤더/데이터 분리 ---
    header = None
    data_rows = rows

    first = [c.strip() for c in rows[0]]
    # 헤더 유무 및 형식 확인
    has_header = first[0].lower() in ("type", "size")
    has_type_side = (has_header and first[0].lower() == "type")

    if has_header:
        header = first
        data_rows = [ [c.strip() for c in row] for row in rows[1:] ]
    else:
        # 헤더가 없으면 전체를 데이터로 처리
        data_rows = [ [c.strip() for c in row] for row in rows ]

    out_rows = []
    scaled_count = 0

    # 헤더는 그대로 복사
    if header is not None:
        out_rows.append(header)

    # --- 각 행 스케일링 및 필터링 ---
    for row in data_rows:
        if not row or all(not c for c in row):
            continue  # 빈 줄 스킵

        current_size = None
        new_row = None # 스케일링 성공 시 할당됨

        try:
            if has_type_side:
                # 1. 현재 사이즈 파싱 (인덱스 2)
                current_size = int(float(row[2]))
            else:
                # 1. 현재 사이즈 파싱 (인덱스 0)
                current_size = int(float(row[0]))
            
        except ValueError:
            continue # size 컬럼 값이 숫자가 아니면 스킵
            
        # 2. 스케일링 조건 확인
        if current_size == CORRECT_SHOE_SIZE:
            
            if has_type_side:
                # 메타데이터 추출
                type_label = row[0]
                side       = row[1]
                coord_strs = row[3:]
            
            else:
                coord_strs = row[1:]
            
            try:
                # 좌표 스케일링
                coords = [float(v) for v in coord_strs]
            except ValueError:
                continue # 좌표가 숫자가 아니면 스킵
                    
            scaled = [v * ratio for v in coords]
            new_size_str = f"{TARGET_SHOE_SIZE:.0f}"

            # 새 행 구성
            if has_type_side:
                new_row = [type_label, side, new_size_str] \
                          + [f"{v:.6f}" for v in scaled]
            else:
                new_row = [new_size_str] + [f"{v:.6f}" for v in scaled]
            
            # ★★★ 스케일링에 성공한 행만 out_rows에 추가 ★★★
            out_rows.append(new_row)
            scaled_count += 1
            
        # 스케일링되지 않은 행은 out_rows에 추가되지 않음

    # --- 결과 쓰기 ---
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    print(f"[OK] Scaled CSV saved to -> {output_path}")
    print(f"     Only {scaled_count} rows (scaled from {CORRECT_SHOE_SIZE} to {TARGET_SHOE_SIZE}) were saved.")


if __name__ == "__main__":
    scale_master_csv(INPUT_CSV, OUTPUT_CSV, ratio)