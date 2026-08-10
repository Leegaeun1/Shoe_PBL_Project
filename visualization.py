import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch

# === 파일 경로 설정 ===
img_raw_path = "raw_sample.jpg"       # 원본 사진
img_contour_path = "contour_sample.png" # 1_Counter_Code 결과
img_result_path = "prediction_plot.png" # 5_Result_Visual 결과 (크롭된 그래프 추천)

def create_paper_figure():
    # 1. 캔버스 생성 (가로로 긴 비율)
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2]) # 결과 그래프를 조금 더 넓게

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # 2. 이미지 로드 및 표시
    # (a) Raw Input
    try:
        img1 = mpimg.imread(img_raw_path)
        ax1.imshow(img1)
        ax1.set_title("(a) Raw Image Input", fontsize=14, fontweight='bold', y=-0.15)
        ax1.axis('off')
    except: ax1.text(0.5, 0.5, "Image Not Found", ha='center')

    # (b) Extraction
    try:
        img2 = mpimg.imread(img_contour_path)
        # 흑백 반전 등 처리가 필요하면 여기서 cmap 조절
        ax2.imshow(img2, cmap='gray') 
        ax2.set_title("(b) Contour Extraction", fontsize=14, fontweight='bold', y=-0.15)
        ax2.axis('off')
    except: ax2.text(0.5, 0.5, "Image Not Found", ha='center')

    # (c) Prediction
    try:
        img3 = mpimg.imread(img_result_path)
        ax3.imshow(img3)
        ax3.set_title("(c) Prediction (PURE-SVR)", fontsize=14, fontweight='bold', y=-0.15)
        ax3.axis('off')
    except: ax3.text(0.5, 0.5, "Image Not Found", ha='center')

    # 3. 화살표 그리기 (Fancy Arrow)
    # 좌표는 (0,0)~(1,1) Figure 좌표계 기준
    # 시행착오로 위치를 조금씩 조절해야 할 수 있습니다.
    arrow_props = dict(arrowstyle='simple,head_length=0.7,head_width=0.7,tail_width=0.3', 
                       color='gray', alpha=0.6, mutation_scale=20)
    
    # Arrow 1 -> 2
    trans = fig.transFigure
    arrow1 = FancyArrowPatch((0.32, 0.55), (0.35, 0.55), transform=trans, **arrow_props)
    fig.patches.append(arrow1)

    # Arrow 2 -> 3
    arrow2 = FancyArrowPatch((0.64, 0.55), (0.67, 0.55), transform=trans, **arrow_props)
    fig.patches.append(arrow2)

    # 4. 저장
    plt.savefig("Figure_Pipeline_Final.png", dpi=300, bbox_inches='tight')
    print("저장 완료: Figure_Pipeline_Final.png")
    plt.show()

# 실행하려면 주석 해제
# create_paper_figure()