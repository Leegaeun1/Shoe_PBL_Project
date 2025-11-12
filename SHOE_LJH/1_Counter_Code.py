import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import math
import os
import traceback

# ==============================================================================
# SECTION 1: 유틸리티 및 헬퍼 함수
# (이 섹션은 변경 사항 없음)
# ==============================================================================

def resize_for_display(image, max_width=1600):
    """(세로가 길면 가로로 돌린 후) 화면 표시에 맞게 이미지 크기를 조절하는 함수"""
    if image is None: return None
    
    h, w = image.shape[:2]

    # 세로(h)가 가로(w)보다 길면 (세로 이미지이면)
    if h > w:
        # 이미지를 시계 방향으로 90도 회전
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # 회전 후, 높이와 너비 값을 다시 계산
        h, w = image.shape[:2] 

    # 기존 리사이즈 로직
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        return cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)
    
    return image

def order_points(pts):
    """네 점을 시계방향 (좌상단, 우상단, 우하단, 좌하단)으로 정렬하는 함수"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_adjacencies(labels_map, num_clusters):
    """클러스터들의 인접 관계를 찾는 함수"""
    adj = {i: set() for i in range(num_clusters)}
    kernel = np.ones((3, 3), np.uint8)
    for i in range(num_clusters):
        mask = np.uint8(labels_map == i) * 255
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        boundary_mask = dilated_mask - mask
        neighbor_labels = np.unique(labels_map[boundary_mask > 0])
        for label in neighbor_labels:
            if label != i:
                adj[i].add(label)
                adj[label].add(i)
    return adj

# ==============================================================================
# SECTION 2: 이미지 처리 핵심 함수
# (이 섹션은 변경 사항 없음)
# ==============================================================================

def get_perspective_transformed_image(img):
    """그린스크린을 찾아 이미지를 반듯하게 원근 변환합니다."""
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    
    kernel = np.ones((7, 7), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
            print("INFO: 그린스크린 영역을 찾지 못했습니다. 검은색 배경을 탐색합니다.")
            screen_type = "Black"
            
            lower_black = np.array([0, 0, 0])
            # 밝기(V) 임계값을 100으로 설정, 채도(S) 임계값도 100으로 설정하여 회색도 포함
            upper_black = np.array([180, 100, 100]) # <-- S, V 임계값 조정
            
            black_mask = cv2.inRange(hsv_img, lower_black, upper_black)
            
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
            
            # ★★★ [추가] 마스크 외곽을 약간 확장하여 4각형 검출에 도움 ★★★
            black_mask = cv2.dilate(black_mask, np.ones((5,5),np.uint8), iterations=1) 
            
            contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("오류: 그린스크린과 검은색 배경을 모두 찾지 못했습니다.")
                return None, None
    

    screen_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(screen_contour)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    
# 만약 여전히 4개가 아니라면, 4개 꼭짓점이 있는 컨투어를 직접 찾아서 시도
    # 이 로직은 조금 더 복잡해지지만, 가장 견고한 방법일 수 있습니다.
    if len(approx) != 4:
        print(f"INFO: 초기 {screen_type} 스크린이 사각형이 아닙니다. 검출된 꼭짓점: {len(approx)}개. 4개짜리 컨투어를 재탐색합니다.")
        
        # 모든 컨투어를 순회하며 4개 꼭짓점의 컨투어 찾기
        four_point_contours = []
        for c in contours:
            # 면적이 너무 작은 컨투어는 무시 (진짜 배경이 아닐 가능성)
            if cv2.contourArea(c) < (img.shape[0] * img.shape[1] / 100): # 전체 이미지 면적의 1% 미만은 무시
                continue
            
            # 볼록 껍질을 다시 계산하여 좀 더 일반적인 형태를 얻음
            c_hull = cv2.convexHull(c)
            c_peri = cv2.arcLength(c_hull, True)
            c_approx = cv2.approxPolyDP(c_hull, 0.03 * c_peri, True) # 동일한 epsilon 사용
            
            if len(c_approx) == 4:
                four_point_contours.append(c_approx)
        
        if four_point_contours:
            # 4개 꼭짓점 컨투어 중 가장 큰 면적을 가진 것을 선택
            screen_contour = max(four_point_contours, key=cv2.contourArea)
            approx = screen_contour # 선택된 4개 꼭짓점 컨투어로 approx 업데이트
            print(f"INFO: {screen_type} 스크린에서 4개 꼭짓점 컨투어를 성공적으로 재탐색했습니다.")
        else:
            print(f"오류: {screen_type} 스크린에서 4개 꼭짓점 컨투어를 찾을 수 없습니다. 검출된 꼭짓점: {len(approx)}개")
            return None, None # 재탐색에도 실패하면 종료
    
    src_pts = order_points(approx.reshape(4, 2))
    (tl, tr, br, bl) = src_pts
    width_a, width_b = np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
    height_a, height_b = np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
    
    target_width = 2000
    max_width, max_height = max(int(width_a), int(width_b)), max(int(height_a), int(height_b))
    aspect_ratio = max_width / max_height if max_height != 0 else 1
    target_height = int(target_width / aspect_ratio)
    
    dst_pts = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (target_width, target_height))
    
    return warped, M

def get_object_mask_with_kmeans(warped_image, k=13):
    """K-Means와 배경 영역 확장으로 객체 마스크를 생성합니다."""
    # 1. K-Means 클러스터링 적용
    h, w = warped_image.shape[:2]
    processing_width = 300
    small_img = cv2.resize(warped_image, (processing_width, int(processing_width * h / w)), interpolation=cv2.INTER_NEAREST)
    lab_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2LAB)
    pixel_values = np.float32(lab_img.reshape((-1, 3)))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers_lab = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    labels_resized = cv2.resize(labels.reshape(small_img.shape[:2]), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 디버깅용 분할 이미지 생성
    centers_uint8 = np.uint8(centers_lab)
    segmented_image_small = centers_uint8[labels.flatten()].reshape(lab_img.shape)
    segmented_image_bgr = cv2.cvtColor(segmented_image_small, cv2.COLOR_LAB2BGR)
    segmented_image = cv2.resize(segmented_image_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. 배경 탐색 및 영역 확장 (Region Growing)
    num_clusters = len(centers_lab)
    centers_bgr = cv2.cvtColor(np.uint8(centers_lab).reshape(-1, 1, 3), cv2.COLOR_LAB2BGR)
    centers_hsv = cv2.cvtColor(centers_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    cluster_counts = np.bincount(labels_resized.flatten())
    background_seed_index = np.argmax(cluster_counts)
    
    adjacencies = find_adjacencies(labels_resized, num_clusters)
    
    screen_indices = {background_seed_index}
    queue = list(adjacencies[background_seed_index])
    visited = {background_seed_index} | set(queue)

    # HSV 유사도 임계값
    hue_threshold, saturation_threshold, value_threshold = 30, 110, 110
    main_h, main_s, main_v = centers_hsv[background_seed_index]

    while queue:
        current_node = queue.pop(0)
        current_h, current_s, current_v = centers_hsv[current_node]
        
        hue_diff = min(abs(int(current_h) - int(main_h)), 180 - abs(int(current_h) - int(main_h)))
        sat_diff = abs(int(current_s) - int(main_s))
        val_diff = abs(int(current_v) - int(main_v))

        if hue_diff <= hue_threshold and sat_diff <= saturation_threshold and val_diff <= value_threshold:
            screen_indices.add(current_node)
            for neighbor in adjacencies[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    print(f"배경 시작 클러스터 ID: {background_seed_index}, 배경 인식 클러스터: {sorted(list(screen_indices))}")

    # 3. 객체 마스크 생성
    obj_mask = np.zeros(labels_resized.shape, dtype=np.uint8)
    for i in range(num_clusters):
        if i not in screen_indices:
            obj_mask[labels_resized == i] = 255
            
    return obj_mask, segmented_image

def get_smoothed_contour(mask):
    """마스크에서 가장 큰 외곽선을 찾아 부드럽게 다듬습니다."""
    # 노이즈 제거 및 내부 구멍 채우기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    kernel = np.ones((15, 15), np.uint8)
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)

    # 블러링을 이용한 외곽선 스무딩
    blurred_mask = cv2.GaussianBlur(filled_mask, (31, 31), 0)
    _, blurred_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)
    
    contours_blurred, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_blurred:
        return None

    # 스플라인 보간법으로 최종 외곽선 생성
    base_contour = max(contours_blurred, key=cv2.contourArea)
    squeezed_contour = np.squeeze(base_contour)
    
    if len(squeezed_contour) < 4:
        return base_contour

    smoothing_factor = len(squeezed_contour) * 5.0
    tck, u = splprep([squeezed_contour[:, 0], squeezed_contour[:, 1]], s=smoothing_factor, per=True)
    x_new, y_new = splev(np.linspace(0, 1, 1000), tck)
    smoothed_contour = np.array([x_new, y_new]).T.reshape((-1, 1, 2)).astype(np.int32)
    
    print("부드러운 외곽선(contour)을 성공적으로 생성했습니다.")
    return smoothed_contour, filled_mask, blurred_mask

# ==============================================================================
# SECTION 3: 메인 처리 함수
# ==============================================================================

# ★★★ 변경점 1: 함수 시그니처 변경 ★★★
# output_folder 대신 최종 저장 경로인 output_path를 받도록 수정
def process_image(input_path, output_path):
    # file_name = os.path.basename(input_path) # <-- 이 로직은 main으로 이동
    # output_path = os.path.join(output_folder, f"outline_{file_name}") # <-- 불필요

    # ================================
    # STEP 1. 이미지 불러오기 및 전처리
    # ================================
    img = cv2.imread(input_path)
    if img is None:
        print(f"오류: '{input_path}' 파일을 찾을 수 없거나 열 수 없습니다.")
        return
    img = resize_for_display(img)

    # ================================
    # STEP 2. 원근 변환 적용
    # ================================
    warped, M = get_perspective_transformed_image(img)
    if warped is None:
        return

    # ================================
    # STEP 3. K-MEANS 기반 객체 마스크 추출 (★새로운 로직★)
    # ================================
    obj_mask, segmented_image = get_object_mask_with_kmeans(warped)
    
    # ================================
    # STEP 4. 외곽선 스무딩 (★새로운 로직★)
    # ================================
    result = get_smoothed_contour(obj_mask)
    if result is None:
        print(f"오류: '{os.path.basename(input_path)}'의 마스크에서 외곽선을 생성할 수 없습니다.")
        return
    analysis_contour, filled_mask, smoothed_mask = result
    
    # ================================
    # STEP 5. 최종 윤곽선 각도 보정 및 저장 (★기존 로직 유지★)
    # ================================
    h_warped, w_warped = warped.shape[:2]
    final_output_image = None
    
    if analysis_contour is not None:
        rect = cv2.minAreaRect(analysis_contour)
        (center_x, center_y), _, angle = rect
        box_w, box_h = rect[1]

        # 객체가 가로로 길게 누워있으면 90도 추가 회전
        if box_w > box_h:
            angle += 90

        # 임시 회전으로 상하 방향 결정
        M_temp_rot = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        rotated_contour = cv2.transform(analysis_contour, M_temp_rot)
        _, y_rc, _, h_rc = cv2.boundingRect(rotated_contour)
        
        contour_mask_for_check = np.zeros((y_rc + h_rc, 1), dtype=np.uint8)
        shifted_contour = rotated_contour - (0, y_rc)
        cv2.drawContours(contour_mask_for_check, [shifted_contour], -1, 255, -1)
        
        mid_y = h_rc // 2
        top_area = cv2.countNonZero(contour_mask_for_check[:mid_y])
        bottom_area = cv2.countNonZero(contour_mask_for_check[mid_y:])

        # 위쪽 면적이 더 크면 (뒤집어져 있으면) 180도 추가 회전
        if top_area > bottom_area:
            angle += 180
        
        # 최종 회전 행렬 계산 및 적용
        M_rot = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        abs_cos = abs(math.cos(math.radians(angle)))
        abs_sin = abs(math.sin(math.radians(angle)))
        w_rot = int(h_warped * abs_sin + w_warped * abs_cos)
        h_rot = int(h_warped * abs_cos + w_warped * abs_sin)
        M_rot[0, 2] += (w_rot / 2) - center_x
        M_rot[1, 2] += (h_rot / 2) - center_y
        
        outline_canvas = np.full((h_warped, w_warped, 3), 255, dtype=np.uint8)
        cv2.drawContours(outline_canvas, [analysis_contour], -1, (0, 0, 0), 3)
        rotated_canvas = cv2.warpAffine(outline_canvas, M_rot, (w_rot, h_rot), borderValue=(255, 255, 255))
        
        # 결과물 크롭핑
        rotated_gray = cv2.cvtColor(rotated_canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(rotated_gray, 250, 255, cv2.THRESH_BINARY_INV)
        contours_rot, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours_rot:
            main_contour_rot = max(contours_rot, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour_rot)
            padding = 20
            x_start, y_start = max(x - padding, 0), max(y - padding, 0)
            x_end, y_end = min(x + w + padding, w_rot), min(y + h + padding, h_rot)
            final_output_image = rotated_canvas[y_start:y_end, x_start:x_end]
        else:
            final_output_image = rotated_canvas
        
        # ★★★ 변경점 2: 인자로 받은 output_path를 그대로 사용 ★★★
        cv2.imwrite(output_path, final_output_image)
        print(f"'{output_path}' 파일에 회전 보정된 윤곽선 이미지를 저장했습니다.")
    else:
        print("물체를 찾지 못해 윤곽선 이미지를 생성할 수 없습니다.")

    # ================================
    # STEP 6. 결과 시각화
    # ================================
    warped_with_contour = warped.copy()
    if analysis_contour is not None:
        cv2.drawContours(warped_with_contour, [analysis_contour], -1, (0, 0, 255), 10)

    # 시각화 코드는 용도에 맞게 활성화/비활성화 하세요 (주석 처리)
    # cv2.imshow("Warped Image with Final Contour", resize_for_display(warped_with_contour, 800))
    # cv2.imshow("K-Means Segments", resize_for_display(segmented_image, 800))
    # cv2.imshow("Object Mask (Initial)", resize_for_display(obj_mask, 800))
    # cv2.imshow("Object Mask (Filled)", resize_for_display(filled_mask, 800))
    # cv2.imshow("Object Mask (Smoothed)", resize_for_display(smoothed_mask, 800))
    # cv2.imshow("Final Cropped Outline", resize_for_display(final_output_image, 800))
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# ★★★ 변경점 3: main 블록 전체 수정 ★★★
if __name__ == '__main__':
    base_input_folder = "ShoeAllData"  # 기본 입력 폴더 (e.g., /shoose_data)
    base_output_folder = "output_outlines_1030" # 기본 출력 폴더
    
    os.makedirs(base_output_folder, exist_ok=True)
    
    if not os.path.isdir(base_input_folder):
        print(f"오류: 입력 폴더 '{base_input_folder}'를 찾을 수 없습니다.")
    else:
        print(f"'{base_input_folder}'에서 스캔을 시작합니다...")
        
        # os.walk로 하위 디렉토리 순회
        for dirpath, dirnames, filenames in os.walk(base_input_folder):
            # dirpath는 'shoose_data/adidas/L'와 같은 형태가 됩니다.
            
            # 경로를 정규화하고 구분자로 분리
            norm_dirpath = os.path.normpath(dirpath)
            parts = norm_dirpath.split(os.sep)
            
            # 우리는 'shoose_data/brand/side' 3단계 깊이의 폴더만 원합니다.
            if len(parts) != 3:
                continue

            # [0]: base_input_folder, [1]: brand, [2]: side
            brand = parts[1]
            side = parts[2]
            
            # L 또는 R 폴더가 아니면 건너뛰기
            if side not in ('L', 'R'):
                continue
                
            print(f"--- [Brand: {brand}, Side: {side}] 폴더 처리 중 ---")
            
            # 최종 출력 폴더 생성 (e.g., 'output_outlines/L')
            final_output_folder = os.path.join(base_output_folder, side)
            os.makedirs(final_output_folder, exist_ok=True)
            
            # 해당 폴더 내의 이미지 파일 처리
            for image_file in filenames:
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue # 이미지 파일이 아니면 건너뛰기
                
                # 1. 원본 이미지 파일 경로
                # e.g., 'shoose_data/adidas/L/230L.png'
                input_path = os.path.join(dirpath, image_file)
                
                # 2. 새로운 출력 파일명 생성
                # e.g., 'adidas_230L.png'
                output_filename = f"{brand}_{image_file}"
                
                # 3. 최종 저장 파일 경로
                # e.g., 'output_outlines/L/adidas_230L.png'
                output_path = os.path.join(final_output_folder, output_filename)
                
                print(f"\n====================\n처리 시작: {input_path}\n====================")
                try:
                    # 변경된 process_image 함수 호출
                    process_image(input_path=input_path, output_path=output_path)
                except Exception as e:
                    print(f"!!! '{image_file}' 처리 중 치명적 오류 발생: {e} !!!")
                    traceback.print_exc()

        print("\n모든 작업이 완료되었습니다.")