import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# [1] 기존 코드의 핵심 함수들 (B-Spline + PCA 정렬)
# =========================================================

# 1-1. B-Spline 관련
def open_uniform_knot_vector(n_ctrl, degree):
    kv = np.concatenate([np.zeros(degree+1), np.arange(1, n_ctrl-degree), np.full(degree+1, n_ctrl-degree)])
    return kv / kv[-1]

def bspline_basis(i, k, knots, t):
    t = np.asarray(t)
    if k == 0:
        last = (i+1 == len(knots)-1)
        return np.where((knots[i] <= t) & ((t < knots[i+1]) | (last & np.isclose(t, knots[i+1]))), 1.0, 0.0)
    left_den, right_den = knots[i+k]-knots[i], knots[i+k+1]-knots[i+1]
    left = ((t-knots[i])/left_den)*bspline_basis(i, k-1, knots, t) if left_den > 0 else 0.0
    right = ((knots[i+k+1]-t)/right_den)*bspline_basis(i+1, k-1, knots, t) if right_den > 0 else 0.0
    return left + right

def bspline_curve(ctrl, degree=3, samples=1500, closed=False):
    ctrl = np.asarray(ctrl, float)
    if closed:
        ctrl = np.concatenate([ctrl, ctrl[:degree]], axis=0)
    n = len(ctrl)
    knots = open_uniform_knot_vector(n, degree)
    t = np.linspace(0, 1, samples, endpoint=True)
    basis = np.stack([bspline_basis(i, degree, knots, t) for i in range(n)], axis=1)
    return basis @ ctrl

# 1-2. 정렬(Alignment) 관련
def pca_major_axis(P):
    C = P - P.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    v1 = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    v2 = np.array([-v1[1], v1[0]]) 
    return v1, v2

def to_heel_up_frame(P, v1=None, v2=None, y0_shift=None):
    if v1 is None or v2 is None: v1, v2 = pca_major_axis(P)
    R = np.stack([v2, v1], axis=1)
    Pp = P @ R
    if y0_shift is None: y0_shift = Pp[:,1].min()
    Pp[:,1] -= y0_shift
    return Pp

# =========================================================
# [2] 이미지 생성 함수 (투명 배경 수정)
# =========================================================
def generate_aligned_shoe_image(csv_path, target_type, target_size, ModelName, ColorType, output_dir="output_images"):
    # 1. 데이터 로드
    if not os.path.exists(csv_path):
        print(f"Error: 파일을 찾을 수 없습니다. ({csv_path})")
        return
    df = pd.read_csv(csv_path)



    # 2. 데이터 필터링
    filtered_df = df[(df['Type'] == target_type) & (df['size'] == target_size)]
    if filtered_df.empty:
        print(f"Error: 데이터 없음 (Type: {target_type}, Size: {target_size})")
        return
    row = filtered_df.iloc[0]

    # 3. 제어점 추출
    points = []
    for i in range(1, 51):
        if f'x{i}' in row and f'y{i}' in row:
            points.append([row[f'x{i}'], row[f'y{i}']])
    control_points = np.array(points)

    # 4. B-Spline 곡선 생성
    raw_curve = bspline_curve(control_points, degree=3, samples=1500, closed=False)
    raw_curve = raw_curve[:-1] # 끝점 미세 보정
    
    # 5. PCA 정렬
    v1, v2 = pca_major_axis(raw_curve)
    aligned_curve = to_heel_up_frame(raw_curve, v1, v2)

    # 6. 시각화 및 저장 (투명 배경 적용)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(6, 10))
    


    print(ColorType)
    # 내부 채우기 (Skyblue) & 외곽선 (Blue)
    COLOR_MAP = {
        0: ('skyblue', 'navy', 0.5),
        1: ('lightgray', 'black', 0.6),
        2: ('#98FB98', '#006400', 0.5),
        3: ('#FFA07A', '#FF4500', 0.6),
        4: ('#E6E6FA', '#4B0082', 0.6),
        5: ('#FFFACD', '#DAA520', 0.7),
        6: ('#E0FFFF', '#008080', 0.6),
        7: ('#FFB6C1', '#C71585', 0.5),
        8: ('#708090', '#2F4F4F', 0.5),
        9: ('none',    'black',   1.0),
    }


    color, edgecolor, alpha = COLOR_MAP.get(ColorType, COLOR_MAP[0])

    plt.fill(aligned_curve[:, 0], aligned_curve[:, 1], 
             color=color, alpha=alpha, edgecolor=edgecolor # >>>>>>>>> 색 바꾸기
             , linewidth=2)  
    




    # [수정된 부분] 축과 그리드 제거
    plt.axis('equal') # 비율 유지
    plt.axis('off')   # 축, 눈금, 테두리 모두 제거
    
    save_name = f"{target_type}_{target_size}_{ModelName}_aligned_transparent.png"
    save_path = os.path.join(output_dir, save_name)
    
    # [수정된 부분] transparent=True 옵션 추가
    plt.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"[Success] 투명 배경 이미지가 저장되었습니다: {save_path}")

# =========================================================
# [3] 실행부
# =========================================================
if __name__ == "__main__":
    # 파일 경로 수정하세요

    state = "20251126" # 20251125 20251126   왼쪽 오른쪽

    CTRLPoint = "CTRL70"
    ModelName = "PCA_LINEAR" # PCA_GPR PCA_KRR PCA_LINEAR PCA_SVR PURE_GPR PURE_KRR PURE_SVR
    CSV_FILE = rf"C:\Users\ljhyu\OneDrive\Desktop\Shoe_Project\신발정리한거\{state}\{CTRLPoint}\Predictions\{ModelName}\pred_Data_{ModelName}_230_280.csv" 
    

    # ModelName = "Ori"
    # CSV_FILE = rf"C:\Users\ljhyu\OneDrive\Desktop\Shoe_Project\신발정리한거\{state}\{CTRLPoint}\control_points_master_R_20251126.csv" 
    
    
    ColorType = 9
    TARGET_TYPE = "Type00"
    TARGET_SIZE = 250
    
    generate_aligned_shoe_image(CSV_FILE, TARGET_TYPE, TARGET_SIZE,ModelName, ColorType)